#include <cuda/atomic>
#include <cuda/ptx>
#include <cuda/std/complex>
#include <cuda/std/cstdint>

#include <algorithm>

#include <cuda_runtime_api.h>

namespace rle_impl
{
namespace ptx = cuda::ptx;
using u64     = cuda::std::uint64_t;

// key/output types: one translation unit per instantiation (mirrors CUB's per-type bench TUs).
// KeyT needs only operator== with CUB's equality semantics (floats: NaN breaks runs; complex:
// componentwise). LenT is the run-length output type; run-length arithmetic is tile-local int,
// widened at the store (valid while the longest run < 2^31, i.e. num_items < 2^31).
#ifndef RLE_KEY_T
#  define RLE_KEY_T int
#endif
#ifndef RLE_LEN_T
#  define RLE_LEN_T int
#endif
// num_runs output type (CUB's OffsetT axis, via choose_signed_offset_t). Internal tile/run
// arithmetic stays int: valid while num_items < 2^31 (the current launcher contract).
#ifndef RLE_NUM_RUNS_T
#  define RLE_NUM_RUNS_T int
#endif
// offset type (CUB's OffsetT): num_items and GLOBAL run indices/open lengths. Wide (i64) builds
// carry the folded prefix as a 16B {count, open} pair and index outputs 64-bit -- REAL support
// for num_items >= 2^31 (e.g. 2^32), not a widened parameter. Per-tile states are tile-bounded
// and stay [launch_gen:32|open:16|count:16] regardless.
#ifndef RLE_OFFSET_T
#  define RLE_OFFSET_T int
#endif
using KeyT     = RLE_KEY_T;
using LenT     = RLE_LEN_T;
using NumRunsT = RLE_NUM_RUNS_T;
using OffT     = RLE_OFFSET_T;
static_assert(sizeof(OffT) == 4 || sizeof(OffT) == 8, "OffT: int or long long");
// KeyT envelope: kSlotPad = 16/sizeof(KeyT) needs the size to DIVIDE the 16B TMA quantum (pow2
// <= 16), and the key ring is carved from the 16B-aligned dynamic smem base. Exotic key types
// (CUB accepts any trivially-copyable size via its untuned generic policy) must be routed to
// stock cub::Encode by the dispatch shell instead of instantiating this kernel.
// TODO(templates pass): these two asserts move inside the templated kernel body -- they are
// per-instantiation constraints on the KeyT template parameter, not TU-scope facts.
static_assert(16 % sizeof(KeyT) == 0, "KeyT size must be a power of two <= 16 (TMA quantum / kSlotPad math)");
static_assert(alignof(KeyT) <= 16, "KeyT alignment must fit the 16B-aligned dynamic smem carve");

#ifndef K_IPT
// size-class tile policy: the tile is capped at 8192 ELEMENTS by the ballot design (32 chunks x
// 32 lanes x kNumCompWarps warps); big keys shrink the tile to keep the 5-deep key ring inside
// the smem cap (8B keys: 5x4096x8=160KB; 16B: 5x2048x16=160KB; <=4B: the tuned 8192 geometry).
constexpr int kIPT = (sizeof(KeyT) >= 16) ? 8 : (sizeof(KeyT) == 8 ? 16 : 32);
#else
constexpr int kIPT = K_IPT; // elements/thread; tile = kNumCompWarps*kIPT*32 (small K_IPT: local-GPU debug)
#endif
#ifndef K_COMP_WARPS
#  define K_COMP_WARPS 8
#endif
constexpr int kNumCompWarps = K_COMP_WARPS; // warp-tiles stay 32*kIPT elems; cw grows the tile
#ifndef K_STORE_WARPS
// 8 beats 16 by +1.3 to +5 BWUtil pts everywhere except seg1 dense (-3.3) now that the drain path is
// cheap (v10); the old "16 balanced" verdict predates that. 16 = dense-leaning alternative.
#  define K_STORE_WARPS 8
#endif
constexpr int kNumStoreWarps = K_STORE_WARPS; // store warps; must divide or be a multiple of kNumCompWarps
#ifndef K_STAGES
#  define K_STAGES \
    5 // 5 (with posRing 3) beats 4 by +0.3..+1.9pt EVERYWHERE; v17b's "5 loses" was a
      // posRing-2 coupling confound. 5x keys + 3x pos = 212KB.
#endif
constexpr int kStages = K_STAGES; // pipeline depth (keys ring)
// positions ring depth: positions are written at staging and consumed by the drain ~2 pipeline_gens later,
// so the ring can be SHALLOWER than the keys ring -- freed smem buys bigger tiles / deeper stages.
// kPosStages < kStages adds a pos_free barrier (store drains release a pos slot for re-staging).
#ifndef K_POS_STAGES
#  define K_POS_STAGES \
    3 // shallower than the keys ring (positions live staging->drain only);
      // 2 strangles the depth benefit (pos_free gates staging on drains-2-back)
#endif
constexpr int kPosStages = K_POS_STAGES;
static_assert(kPosStages >= 2 && kPosStages <= kStages, "positions ring: 2..kStages");

#ifndef K_POLL_MLP
// 4 is measured optimal (5 and 8 both lose, 1-10%): the 2-round fold structure OVERLAPS folding the
// old (long-published) tiles with the frontier tiles' publish window; a single wider round serializes
// the whole fold behind the youngest predecessor's publish.
#  define K_POLL_MLP 4
#endif
constexpr int kPollMlp = K_POLL_MLP; // how many loads each poll lane keeps in flight

// drain-loop unroll (gather MLP): 2 is the measured default winner (+6.5pt dense, ~flat elsewhere);
// 4 gains less at dense and costs 1.3-1.5pt at seg8-16; 0 = no unroll. NOTE this knob's verdict
// FLIPPED twice as the surrounding design changed -- re-sweep it after any store-path change.

// head-flag words (one conflict-free STS.32 per lane; no scan, no peel, no pos-ring use) and the
// drain rank-selects head positions from them (5-shfl binary search + branchless nth-set-bit, pure
// registers, no LDS in the run loop). Dense warp-tiles keep the champion peel+positions path:
// decode cost scales with runs, the staging saving doesn't. Both sides derive the SAME predicate
// from the staged warp run count, so no mode flag is exchanged. Collects the measured peel-write
// conflict prize (v33: +0.4..+1.3 at seg4-16) and deletes staging work + pos_free coupling at mid.
#ifndef RLE_FW_THRESH
// re-swept 2026-07-06 after the nxt-ffs decode cheapening: 64 = +1.4 BWUtil pts at seg32, flat
// elsewhere (v33's "64/96 regress seg32" verdict predates cheap decode). 96 loses seg32 by -4.4
// (drags 65-96-run warp-tiles onto the run-scaling decode), 128 also craters seg16 (-12.6).
#  define RLE_FW_THRESH 64
#endif
// RLE_REGBUF: prefix-decoupled drain. Warp-tiles with run count <= RLE_REGBUF decode+gather into
// REGISTERS before the prefixed wait (only the output ADDRESS needs the prefix), then release the
// pos slot AND the key slot -- `empty` stops waiting on the prefix chain entirely for buffered
// warp-tiles; the final store burst trails the pipeline by the prefix latency and nothing
// downstream orders on it. Early `empty` breaks the old "poll never overwrites a live
// prefix_packed" ordering proof; prefix_packed is double-buffered by slot-cycle parity instead
// (program order proves safety). 0 = off. Buffer = RLE_REGBUF/32 int pairs per lane (registers).
// buffered drain only pays when the drain is long enough for early slot-release to matter;
// near-empty warp-tiles (long segs) stay classic to avoid pure reshape overhead
#ifndef RLE_REGBUF
#  define RLE_REGBUF 256 // register-buffer cap (runs/warp-tile), champion since v40
#endif
#ifndef RLE_RB_MIN
#  define RLE_RB_MIN 8
#endif

// This is important for position staging on dense cases (16 way bank conflicts).
__device__ __forceinline__ int swizzle_xor_stride32(int x)
{
  return x ^ (x >> 5);
}

// CLC = 1 => use shiny new blackwell feature (UGETNEXTWORKID)
// CLC = 0 => use atomics for work stealing. no perf difference observed on blackwell
// The CLC knob removed since I want to focus on blackwell perf first
// i.e. we fry one fish at a time :)

constexpr int kWarpTileSize = 32 * kIPT;
constexpr int kTileSize     = kNumCompWarps * kWarpTileSize;
static_assert(kTileSize <= 0xffff, "per-tile run_count/open_len must fit the 16-bit state-word fields");
constexpr int kNumWarps          = 1 /*load*/ + kNumCompWarps + 1 /*poll*/ + kNumStoreWarps + 1 /*bookkeeper*/;
constexpr int kNumThreads        = kNumWarps * 32;
constexpr unsigned kFullMask     = 0xffffffffu;
constexpr int poll_warp_id       = 1 + kNumCompWarps;
constexpr int store_warp_id      = poll_warp_id + 1;
constexpr int bookkeeper_warp_id = store_warp_id + kNumStoreWarps;

// for each input tile, we need to store the keys and in-tile positions
// for in tile position we can just do unsigned int16 since tile size is never bigger than 2^16
// each key slot carries kSlotPad extra leading elements
// we overcopy one 16B chunk to the left, this does two things at once:
// 1. no need for aligned address
// 2. we get the last tiles boundary element
constexpr int kSlotPad    = 16 / sizeof(KeyT); // elements; 16 bytes = cp_async_bulk quantum
constexpr int kSlotStride = kTileSize + kSlotPad;
constexpr size_t kDynSmem =
  (size_t) kStages * kSlotStride * sizeof(KeyT) + (size_t) kPosStages * kTileSize * sizeof(short);

// tile_partial_states: one word per tile
// Layout: u64 [launch_gen:32][open_len:16][run_count:16]
// launch_gen is needed to reuse allocations per launch
// (this is needed to eliminate overhead of allocating the buffer. CRITICAL for perf!)
// an aligned 64-bit access is already non-tearing, but atomic_ref doesn't hurt and has clear semantics
// TODO(templates pass): make TilePartialStateT a struct wrapping the u64 with launch_gen() /
// run_count() / open_len() accessors -- call sites become packed_words[i].launch_gen() (the
// "from what" rides on the receiver, so the extract_* function names shrink away); fold in when
// the RLE_*_T macros become template parameters.
using TilePartialStateT = u64;

__device__ __forceinline__ unsigned extract_launch_gen_from_tile_partial_state(TilePartialStateT w)
{
  return (unsigned) (w >> 32);
}

__device__ __forceinline__ int extract_run_count_from_tile_partial_state(TilePartialStateT w)
{
  return (int) (w & 0xffffu);
}

__device__ __forceinline__ int extract_open_len_from_tile_partial_state(TilePartialStateT w)
{
  return (int) ((w >> 16) & 0xffffu);
}

__device__ __forceinline__ void
publish_state(TilePartialStateT* tile_state_arr, int tile_idx, unsigned launch_gen, int run_count, int open_len)
{
  TilePartialStateT w = ((u64) launch_gen << 32) | ((u64) (unsigned) open_len << 16) | (u64) (unsigned) run_count;
  cuda::atomic_ref<TilePartialStateT, cuda::thread_scope_device> a(tile_state_arr[tile_idx]);
  a.store(w, cuda::memory_order_relaxed);
}

// return the state (even if not yet publish for this launch, caller checks it)
// we do not want to spin here
__device__ __forceinline__ TilePartialStateT load_state(TilePartialStateT* tile_state_arr, int tile_idx)
{
  cuda::atomic_ref<TilePartialStateT, cuda::thread_scope_device> a(tile_state_arr[tile_idx]);
  return a.load(cuda::memory_order_relaxed);
}

// what is going to be the type of the prefix (run_count, open_len)?
// TODO(templates pass): replace ulonglong2 with a named alignas(16) struct { run_count, open_len }
// -- keeps the single STS.128/LDS.128 access, kills the meaningless .x/.y members, and the
// pack/unpack helpers collapse into the struct (same treatment as TilePartialStateT).
using PrefixT = cuda::std::conditional_t<(sizeof(OffT) > 4), ulonglong2, u64>;

// how do we pack them? if P is 32 bit, we compact them into 1 word. Otherwise, 2 words!
template <class P = PrefixT>
__device__ __forceinline__ P pack_prefix(OffT run_count, OffT open_len)
{
  if constexpr (sizeof(OffT) > 4)
  {
    return P{(u64) run_count, (u64) open_len};
  }
  else
  {
    return ((u64) (unsigned) open_len << 32) | (unsigned) run_count;
  }
}

template <class P>
__device__ __forceinline__ OffT prefix_run_count(P p)
{
  if constexpr (sizeof(OffT) > 4)
  {
    return (OffT) p.x;
  }
  else
  {
    return (OffT) (unsigned) (p & 0xffffffffull);
  }
}
template <class P>
__device__ __forceinline__ OffT prefix_open_len(P p)
{
  if constexpr (sizeof(OffT) > 4)
  {
    return (OffT) p.y;
  }
  else
  {
    return (OffT) (unsigned) (p >> 32);
  }
}

// position of the n-th set bit of flag_mask
// requires popc(flag_mask) > rank.
// __fns(flag_mask, 0, rank+1) computes the same thing but has NO hardware op on sm_100a and is slower
__device__ __forceinline__ int nth_set_bit(unsigned flag_mask, int rank)
{
  // each step: if the wanted bit is not among the low half's set bits, skip that half entirely
  // this is manually unrolled to reduce the count of generated SASS instructions
  int bit_position         = 0;
  int set_bits_in_low_half = __popc(flag_mask & 0xffffu);
  if (rank >= set_bits_in_low_half)
  {
    rank -= set_bits_in_low_half;
    bit_position += 16;
    flag_mask >>= 16;
  }
  set_bits_in_low_half = __popc(flag_mask & 0xffu);
  if (rank >= set_bits_in_low_half)
  {
    rank -= set_bits_in_low_half;
    bit_position += 8;
    flag_mask >>= 8;
  }
  set_bits_in_low_half = __popc(flag_mask & 0xfu);
  if (rank >= set_bits_in_low_half)
  {
    rank -= set_bits_in_low_half;
    bit_position += 4;
    flag_mask >>= 4;
  }
  set_bits_in_low_half = __popc(flag_mask & 0x3u);
  if (rank >= set_bits_in_low_half)
  {
    rank -= set_bits_in_low_half;
    bit_position += 2;
    flag_mask >>= 2;
  }
  if (rank >= (int) (flag_mask & 1u))
  {
    bit_position += 1;
  }
  return bit_position;
}

__device__ __forceinline__ void poll_and_fold(
  TilePartialStateT* tile_partial_states,
  unsigned launch_gen,
  int tile_id,
  int& last_seen_tile_id,
  OffT& last_seen_prefix_run_count,
  OffT& last_seen_prefix_open_length,
  int lane_id,
  OffT& curr_prefix_run_count,
  OffT& curr_prefix_open_length)
{
  while (last_seen_tile_id < tile_id)
  {
    const int remain = tile_id - last_seen_tile_id;
    // # of tiles to fold this iteration
    const int chunk = remain < 32 * kPollMlp ? remain : 32 * kPollMlp;
    // lane l owns the contiguous tiles
    // [last_seen_tile_id + l*kPollMlp, last_seen_tile_id + l*kPollMlp + kPollMlp)
    // clamped to `chunk`
    const int lane_base = last_seen_tile_id + lane_id * kPollMlp;
    int lane_tile_count = chunk - lane_id * kPollMlp;
    lane_tile_count     = lane_tile_count < 0 ? 0 : (lane_tile_count > kPollMlp ? kPollMlp : lane_tile_count);
    // issue all kPollMlp loads up front, then spin until this lane's owned tiles are all published (MLP)
    TilePartialStateT packed_words[kPollMlp] = {}; // must zero initialize
    bool ready;
    do
    {
      ready = true;
#pragma unroll
      for (int i = 0; i < kPollMlp; ++i)
      {
        if (i < lane_tile_count && extract_launch_gen_from_tile_partial_state(packed_words[i]) != launch_gen)
        {
          packed_words[i] = load_state(tile_partial_states, lane_base + i);
          if (extract_launch_gen_from_tile_partial_state(packed_words[i]) != launch_gen)
          {
            ready = false;
          }
        }
      }
    } while (__ballot_sync(kFullMask, !ready) != 0u);
    // ordered reduce this lane's own tiles (increasing left -> right)
    int lane_run_count = 0, lane_open_length = 0;
#pragma unroll
    for (int i = 0; i < kPollMlp; ++i)
    {
      if (i < lane_tile_count)
      {
        const int tile_run_count   = extract_run_count_from_tile_partial_state(packed_words[i]);
        const int tile_open_length = extract_open_len_from_tile_partial_state(packed_words[i]);
        lane_run_count             = lane_run_count + tile_run_count;
        lane_open_length           = (tile_run_count > 0) ? tile_open_length : (lane_open_length + tile_open_length);
      }
    }
    // cross lane fold over 32 lane aggregates
    const int chunk_run_count      = __reduce_add_sync(kFullMask, lane_run_count);
    const unsigned lanes_with_runs = __ballot_sync(kFullMask, lane_run_count > 0);
    const int last_run_lane        = lanes_with_runs ? (31 - __clz(lanes_with_runs)) : 0;
    const int chunk_open_length    = __reduce_add_sync(kFullMask, (lane_id >= last_run_lane) ? lane_open_length : 0);
    // combine last_seen_prefix with the chunk aggregate
    const OffT new_run_count = last_seen_prefix_run_count + chunk_run_count;
    const OffT new_open_length =
      (chunk_run_count > 0) ? (OffT) chunk_open_length : (last_seen_prefix_open_length + chunk_open_length);
    last_seen_prefix_run_count   = new_run_count;
    last_seen_prefix_open_length = new_open_length;
    last_seen_tile_id += chunk;
  }
  curr_prefix_run_count   = last_seen_prefix_run_count;
  curr_prefix_open_length = last_seen_prefix_open_length;
}

// we aim for 1 block/SM since it is easier to manage resources: do not need to worry about occupancy anymore
__launch_bounds__(kNumThreads, 1) __global__ void persistent_rle(
  const KeyT* __restrict__ d_keys,
  KeyT* __restrict__ d_unique,
  LenT* __restrict__ d_counts,
  NumRunsT* __restrict__ d_num_runs,
  TilePartialStateT* __restrict__ tile_partial_states,
  const unsigned* __restrict__ d_launch_gen,
  OffT num_items,
  int num_tiles)
{
  // [kStages][kTileSize] int32 ring (input keys)
  // [kStages][kTileSize] int16 staged head positions
  extern __shared__ char smem_raw[]; // 16B-aligned; KeyT alignment <= 16 for all supported types
  KeyT* const tile_buf    = (KeyT*) smem_raw;
  short* const staged_pos = (short*) (tile_buf + (size_t) kStages * kSlotStride);
  __shared__ int tile_id_ring[kStages]; // which global tile each ring slot holds (LOAD gets it with try_cancel)
  __shared__ int warp_run_counts[kStages][kNumCompWarps]; // per compute warp run counts
  __shared__ unsigned flag_ring[kStages][kNumCompWarps * 32]; // staged head-flag words (flag-word path)
  __shared__ int warp_first_heads[kStages][kNumCompWarps]; // per compute warp first head idx (-1 if none)
  __shared__ int warp_last_heads[kStages][kNumCompWarps]; // per compute warp last head idx (-1 if none)
  // POLL -> STORE handoff: [open_len_prefix:32][run_count_prefix:32] packed, one access per side
  // under REGBUF the prefix is double-buffered by slot-cycle parity: POLL(g+kStages) writes the
  // OTHER half from the one STORE(g)/BOOKKEEPER(g) read, and program order proves safety (a
  // store warp reads pipeline_gen g's half before its pipeline_gen g+kStages empty-arrive, which gates the load
  // that gates poll's next write to that half). No consumed-barrier -- v38's pfx_free put the
  // store wake-up latency inside the POLL's serial loop and cost -2..-5pt at seg128+.
  __shared__ PrefixT prefix_packed[kStages][2];
  // STORE --pos_free--> COMPUTE staging (only when kPosStages < kStages): all store warps arrive
  // after their drain finished READING a pos slot; compute waits before re-staging into it
  __shared__ u64 pos_free[kPosStages];
  // barriers (per ring slot):
  // LOAD --full--> COMPUTE & POLL
  // COMPUTE(all warps) --computed--> COMPUTE warp0
  // warp0 calculates & publishes this tile's aggregate to the global
  // POLL --prefixed--> STORE
  // STORE --empty--> LOAD & POLL
  // (an `assigned` barrier that woke POLL at TMA-issue time was measured a small pure loss once the
  // poll spin stopped re-loading valid states -- see the bench notes; POLL waits `full` like COMPUTE)
  __shared__ u64 full[kStages];
  __shared__ u64 computed[kStages], prefixed[kStages], empty[kStages];
  // COMPUTE warp w --staged_wt[w]--> STORE: per-warp-tile handoff, so store warps drain a warp-tile
  // as soon as ITS positions are staged instead of waiting for all 8 compute warps (warp 0 is always
  // last -- it does the publish fan-in first). The shared metadata store also needs (run counts,
  // first/last heads, tile_id_ring) is covered by `computed`.
  __shared__ u64 staged_wt[kStages][kNumCompWarps];

  // try_cancel writes a 16-byte response into clc_resp + completes clc_bar's tx.
  __shared__ __align__(16) uint4 clc_resp;
  __shared__ u64 clc_bar;

  const int thr_id          = threadIdx.x;
  const int warp_id         = thr_id >> 5;
  const int lane_id         = thr_id & 31;
  const int blk_id          = blockIdx.x;
  const unsigned launch_gen = __ldg(d_launch_gen); // written by the init kernel, stream-ordered

  if (thr_id == 0)
  {
    for (int slot_id = 0; slot_id < kStages; ++slot_id)
    {
      ptx::mbarrier_init(&full[slot_id], 1);
      ptx::mbarrier_init(&computed[slot_id], kNumCompWarps); // every compute warp arrives
      ptx::mbarrier_init(&prefixed[slot_id], 1);
      ptx::mbarrier_init(&empty[slot_id], kNumStoreWarps + 1); // store warps + the bookkeeper
      for (int cw = 0; cw < kNumCompWarps; ++cw)
      {
        ptx::mbarrier_init(&staged_wt[slot_id][cw], 1); // that compute warp's lane0, after its scatter
      }
    }
    for (int p = 0; p < kPosStages; ++p)
    {
      ptx::mbarrier_init(&pos_free[p], kNumStoreWarps);
    }

    ptx::mbarrier_init(&clc_bar, 1); // 1 arrival
  }
  // normal smem writes (e.g. mbarrier_init) go through the generic proxy
  // the TMA operations access shared memory through the async proxy. these are separate visibility domains,
  // so the init writes are not automatically visible to TMA.
  ptx::fence_proxy_async(ptx::space_shared);
  __syncthreads();

  // if you are load
  if (warp_id == 0)
  {
    // CLC tile assignment: gen0 tile = this CTA's launch id (blockIdx.x)
    int tile_id = blk_id;
    if (lane_id == 0)
    {
      // 16 is the try_cancel byte tx
      ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &clc_bar, 16);
      ptx::clusterlaunchcontrol_try_cancel(&clc_resp, &clc_bar);
    }
    for (int pipeline_gen = 0;; ++pipeline_gen)
    {
      const int slot_id  = pipeline_gen % kStages; // which slot is this?
      const int slot_gen = pipeline_gen / kStages; // how many times is this slot used?
      if (pipeline_gen >= kStages)
      {
        // need to wait for slot to be free
        while (!ptx::mbarrier_try_wait_parity(&empty[slot_id], (unsigned) ((slot_gen - 1) & 1)))
        {
        }
      }
      if (lane_id == 0)
      {
        tile_id_ring[slot_id] = tile_id;
      }
      if (tile_id >= num_tiles)
      {
        if (lane_id == 0)
        {
          ptx::mbarrier_arrive(&full[slot_id]);
        }
        __syncwarp();
        break;
      }
      // over-fetch one 16B chunk to the left: slot[0..kSlotPad-1] = previous tile's last keys,
      // so COMPUTE reads element 0's predecessor from smem instead of a blocking LDG here.
      // tile 0 has no predecessor and skips the over-fetch; its slot[0..kSlotPad-1] stay
      // unread because is_global_first forces element 0's head flag.
      const bool first_tile = (tile_id == 0);
      const int tile_len    = (int) min((OffT) kTileSize, num_items - (OffT) tile_id * kTileSize);
      // partial LAST tile: cp_async_bulk moves 16B multiples only, and reading past num_items
      // would touch memory the user never allocated -- round the TMA down to the quantum and fetch
      // the ragged tail (< 16B) with plain per-lane loads. The epilogue stores happen-before the
      // release arrive below, so full[]'s acquire waiters see them along with the TMA bytes.
      const int tma_elems = (tile_len == kTileSize) ? kTileSize : (tile_len & ~(kSlotPad - 1));
      if (tile_len != kTileSize)
      {
        KeyT* const slot_keys = tile_buf + (size_t) slot_id * kSlotStride + kSlotPad;
        for (int e = tma_elems + lane_id; e < tile_len; e += 32)
        {
          slot_keys[e] = d_keys[(size_t) tile_id * kTileSize + e];
        }
        __syncwarp(); // all epilogue stores done before lane 0's release arrive
      }
      if (lane_id == 0)
      {
        const unsigned nbytes = (unsigned) (((size_t) tma_elems + (first_tile ? 0 : kSlotPad)) * sizeof(KeyT));
        ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &full[slot_id], nbytes);
        if (nbytes != 0)
        {
          ptx::cp_async_bulk(
            ptx::space_shared,
            ptx::space_global,
            tile_buf + (size_t) slot_id * kSlotStride + (first_tile ? kSlotPad : 0),
            d_keys + (size_t) tile_id * kTileSize - (first_tile ? 0 : kSlotPad),
            nbytes,
            &full[slot_id]);
        }
      }
      __syncwarp();
      // consume the prefetched cancel
      // this is ok since it should be fast to get next cancelled id
      if (lane_id == 0)
      {
        while (!ptx::mbarrier_try_wait_parity(&clc_bar, (unsigned) (pipeline_gen & 1)))
        {
        }
        // try_cancel wrote clc_resp via the async proxy
        ptx::fence_proxy_async(ptx::space_shared);
        const bool canceled = ptx::clusterlaunchcontrol_query_cancel_is_canceled(clc_resp);
        int nxt             = num_tiles; // if no more work was cancellable
        if (canceled)
        {
          nxt = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x<int>(clc_resp);
          ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &clc_bar, 16);
          ptx::clusterlaunchcontrol_try_cancel(&clc_resp, &clc_bar);
        }
        tile_id = nxt;
      }
      tile_id = __shfl_sync(kFullMask, tile_id, 0);
    }
  }
  // if you are compute
  else if (warp_id <= kNumCompWarps)
  {
    const int compute_warp_id = warp_id - 1;
    const int warp_tile_base  = compute_warp_id * kWarpTileSize;
    for (int pipeline_gen = 0;; ++pipeline_gen)
    {
      const int slot_id  = pipeline_gen % kStages;
      const int slot_gen = pipeline_gen / kStages;
      while (!ptx::mbarrier_try_wait_parity(&full[slot_id], (unsigned) (slot_gen & 1)))
      {
      }
      const int tile_id = tile_id_ring[slot_id];
      if (tile_id >= num_tiles)
      {
        if (lane_id == 0)
        {
          // drain: STORE waits computed + its warp-tile's staged_wt, so arrive both
          ptx::mbarrier_arrive(&computed[slot_id]);
          ptx::mbarrier_arrive(&staged_wt[slot_id][compute_warp_id]);
        }
        break;
      }
      // slot is ready! (under split, half-1 warps index through +2*kSlotPad so pad1 sits at loc
      // kHalfTile-1's predecessor position; all their accesses incl. loc-1 stay in half 1 + pad1)
      const KeyT* key_buf = tile_buf + (size_t) slot_id * kSlotStride + kSlotPad;
      const int tile_len  = (int) min((OffT) kTileSize, num_items - (OffT) tile_id * kTileSize);
      int local_run_count = 0, warp_first_head = -1, warp_last_head = -1;
      static_assert(kIPT <= 32, "one lane per iter requires kIPT<=32");
      // start calculating head_flags:
      // each iter is 32 consecutive elements (lane L owns loc = warp_tile_base + iter*32 + L)
      // head = (key != predecessor)
      // __ballot makes a 32-bit head mask per iter
      // the lane whose lane_id == iter stashes it, so after the loop lane L holds chunk L's mask
      // NOTE: this loop is a measured local optimum -- FOUR restructures have lost to it (shfl preds,
      // two int4 forms, and stripping the bounds/global-first predicates for full tiles at +6-21%).
      // The predicates ride free in idle issue slots; changing the loop breaks its pipelining.
      short* const pos_dst = staged_pos + (size_t) (pipeline_gen % kPosStages) * kTileSize;
      unsigned my_flags    = 0;
#pragma unroll
      for (int iter = 0; iter < kIPT; ++iter)
      {
        const int loc             = warp_tile_base + iter * 32 + lane_id;
        const KeyT key            = (loc < tile_len) ? key_buf[loc] : KeyT{};
        const KeyT pred           = key_buf[loc - 1]; // loc==0 reads the over-fetched slot[kSlotPad-1]
        const int is_global_first = (tile_id == 0 && loc == 0);
        const int head            = (loc < tile_len) ? (is_global_first ? 1 : (key != pred)) : 0;
        const unsigned flags      = __ballot_sync(kFullMask, head);
        if (lane_id == iter)
        {
          my_flags = flags;
        }
      }
      local_run_count = __reduce_add_sync(kFullMask, __popc(my_flags));
      // each lane in a warp now has a mask that tells which chunk is non empty
      const unsigned nonempty_chunk_mask = __ballot_sync(kFullMask, my_flags != 0u);
      // if warptile is non empty (has heads), we get the location of warps first head and last head
      if (nonempty_chunk_mask)
      {
        const int first_chunk           = __ffs(nonempty_chunk_mask) - 1;
        const int last_chunk            = 31 - __clz(nonempty_chunk_mask);
        const unsigned first_chunk_mask = __shfl_sync(kFullMask, my_flags, first_chunk);
        const unsigned last_chunk_mask  = __shfl_sync(kFullMask, my_flags, last_chunk);
        warp_first_head                 = warp_tile_base + first_chunk * 32 + (__ffs(first_chunk_mask) - 1);
        warp_last_head                  = warp_tile_base + last_chunk * 32 + 31 - __clz(last_chunk_mask);
      }
      // now, we calculate warptile aggregates
      if (lane_id == 0)
      {
        warp_run_counts[slot_id][compute_warp_id]  = local_run_count;
        warp_first_heads[slot_id][compute_warp_id] = warp_first_head;
        warp_last_heads[slot_id][compute_warp_id]  = warp_last_head;
        ptx::mbarrier_arrive(&computed[slot_id]); // each compute warp arrives
      }
      // warp 0 waits all compute warp arrivals so that every warp's results are visible
      // then collect results from all warptiles and publish the tile run count and tile open len
      if (compute_warp_id == 0)
      {
        while (!ptx::mbarrier_try_wait_parity(&computed[slot_id], (unsigned) (slot_gen & 1)))
        {
        }
        {
          // kNumCompWarps<=32 so one lane/warp fits
          // (in practice we will never have anything close to 32)
          static_assert(kNumCompWarps <= 32, "insane...");
          const bool active        = lane_id < kNumCompWarps;
          const int warp_run_count = active ? warp_run_counts[slot_id][lane_id] : 0;
          const int run_count      = __reduce_add_sync(kFullMask, warp_run_count);
          // last head = the highest-index warp that has any run (its last_head is the tile's last head)
          const unsigned warps_with_runs = __ballot_sync(kFullMask, active && warp_run_count > 0);
          int last_head_idx              = -1;
          // if we have any heads, get last head index
          if (warps_with_runs)
          {
            const int last_warp_with_runs = 31 - __clz(warps_with_runs);
            last_head_idx =
              __shfl_sync(kFullMask, active ? warp_last_heads[slot_id][lane_id] : -1, last_warp_with_runs);
          }
          if (lane_id == 0)
          {
            const int open_len = (run_count > 0) ? (tile_len - last_head_idx) : tile_len;
            // CRITICAL: publish as soon as possible, this is why we calculate head_flags first
            publish_state(tile_partial_states, tile_id, launch_gen, run_count, open_len);
          }
        }
      }
      // now we start to calculate head positions
      const bool stage_flags = (local_run_count < RLE_FW_THRESH);
      if (stage_flags)
      {
        flag_ring[slot_id][compute_warp_id * 32 + lane_id] = my_flags;
      }
      else
      {
        if constexpr (kPosStages < kStages)
        {
          // the pos slot is shared by pipeline_gens g, g+kPosStages, ...: wait for the drains of pipeline_gen
          // g-kPosStages to have finished reading it (satisfied by pipeline offset in steady state)
          if (pipeline_gen >= kPosStages)
          {
            while (!ptx::mbarrier_try_wait_parity(
              &pos_free[pipeline_gen % kPosStages], (unsigned) ((pipeline_gen / kPosStages - 1) & 1)))
            {
            }
          }
        }
        {
          // we store run R at warp_tile_base + (R ^ (R>>5)) to avoid bank conflicts for dense cases
          // (CRITICAL for MaxSeg=1,2,4)
          int head_scan = __popc(my_flags); // start: this word's head count
#pragma unroll
          for (int offset = 1; offset < 32; offset <<= 1)
          {
            const int pred_head_scan = __shfl_up_sync(kFullMask, head_scan, offset);
            if (lane_id >= offset)
            {
              head_scan += pred_head_scan;
            }
          }
          // head_scan is a running sum of run_count, so each lane know each chunk's base
          const int word_run_base = head_scan - __popc(my_flags);
          if (lane_id < kIPT)
          {
            // NOTE: this peel is warp-parallel ALREADY (32 independent per-lane chains); a 4-way
            // byte-split "ILP" variant measured 5-8pt SLOWER -- per-lane chain depth is not the
            // binding constraint here, instruction count is. Don't touch.
            const int word_pos     = warp_tile_base + lane_id * 32; // element position of bit 0 of this word
            unsigned pending_heads = my_flags; // this word's head mask; we need to "peel" it headbit by headbit
            int run_index          = word_run_base; // run-order slot for this word's next head
            while (pending_heads)
            {
              const int head_offset = __ffs(pending_heads) - 1; // offset (0..31) of the next head within the word
              pos_dst[warp_tile_base + swizzle_xor_stride32(run_index)] = (short) (word_pos + head_offset);
              ++run_index;
              pending_heads &= (pending_heads - 1); // clear the lowest set bit
            }
          }
        }
      } // adaptive-staging else scope
      if (lane_id == 0)
      {
        ptx::mbarrier_arrive(&staged_wt[slot_id][compute_warp_id]); // this warp-tile's positions ready
      }
    }
  }
  // if you are poll
  else if (warp_id == poll_warp_id)
  {
    int last_seen_tile_id             = 0;
    OffT last_seen_prefix_run_count   = 0;
    OffT last_seen_prefix_open_length = 0;
    for (int pipeline_gen = 0;; ++pipeline_gen)
    {
      const int slot_id  = pipeline_gen % kStages;
      const int slot_gen = pipeline_gen / kStages;
      while (!ptx::mbarrier_try_wait_parity(&full[slot_id], (unsigned) (slot_gen & 1)))
      {
      }
      const int tile_id = tile_id_ring[slot_id];
      if (tile_id >= num_tiles)
      {
        if (lane_id == 0)
        {
          ptx::mbarrier_arrive(&prefixed[slot_id]); // drain
        }
        break;
      }
      OffT curr_prefix_run_count, curr_prefix_open_length;
      poll_and_fold(
        tile_partial_states,
        launch_gen,
        tile_id,
        last_seen_tile_id,
        last_seen_prefix_run_count,
        last_seen_prefix_open_length,
        lane_id,
        curr_prefix_run_count,
        curr_prefix_open_length);
      // no wait needed before overwriting the prefix slot: STORE(pipeline_gen-kStages) arrives `empty` as its
      // LAST act, LOAD cannot arm `full` for this pipeline_gen until it passed that same `empty` phase, and we
      // already passed `full` above -- so a drain-wait here could provably never spin.
      if (lane_id == 0)
      {
        prefix_packed[slot_id][slot_gen & 1] = pack_prefix(curr_prefix_run_count, curr_prefix_open_length);
        ptx::mbarrier_arrive(&prefixed[slot_id]); // prefix ready, store may proceed! (2/2)
      }
    }
  }
  // if you are store
  else if (warp_id < bookkeeper_warp_id)
  {
    const int store_warp_idx = warp_id - store_warp_id;
    for (int pipeline_gen = 0;; ++pipeline_gen)
    {
      const int slot_id = pipeline_gen % kStages;
      // wait for computed (1/3): all per-warp-tile metadata (run counts, first/last heads) is
      // written before each compute warp's `computed` arrive -- positions are NOT needed yet
      while (!ptx::mbarrier_try_wait_parity(&computed[slot_id], (unsigned) ((pipeline_gen / kStages) & 1)))
      {
      }
      const int tile_id = tile_id_ring[slot_id];
      if (tile_id >= num_tiles)
      {
        if (lane_id == 0)
        {
          ptx::mbarrier_arrive(&empty[slot_id]);
        }
        break;
      }
      // store warps and compute warps are decoupled
      // storeW>=cw -> multiple store warps split one compute warp's runs;
      // storeW<cw -> each store warp drains cw/storeW whole regions. One must divide the other.
      static_assert(kNumStoreWarps % kNumCompWarps == 0 || kNumCompWarps % kNumStoreWarps == 0,
                    "They must divide (either direction)");
      // per-warp-tile run bases: warp-parallel exclusive scan over the counts, held in REGISTERS
      // (lane i owns warp-tile i's count/base) and done BEFORE the prefixed wait so it overlaps it.
      // replaces an 8-deep serial smem load+add chain that used to run after the wait.
      const int wt_count_ln = (lane_id < kNumCompWarps) ? warp_run_counts[slot_id][lane_id] : 0;
      int wt_incl_ln        = wt_count_ln;
#pragma unroll
      for (int offset = 1; offset < kNumCompWarps; offset <<= 1)
      {
        const int p = __shfl_up_sync(kFullMask, wt_incl_ln, offset);
        if (lane_id >= offset)
        {
          wt_incl_ln += p;
        }
      }
      const int wt_base_ln  = wt_incl_ln - wt_count_ln; // lane i: warp-tile i's exclusive run base
      const KeyT* tile_keys = tile_buf + (size_t) slot_id * kSlotStride + kSlotPad;
      // staged positions
      const short* run_positions = staged_pos + (size_t) (pipeline_gen % kPosStages) * kTileSize;
      // wait for prefixed (2/3); drains only need the run-count prefix (addresses) -- the
      // open-length half is bookkeeper-only
      auto wait_prefixed_and_read = [&]() {
        while (!ptx::mbarrier_try_wait_parity(&prefixed[slot_id], (unsigned) ((pipeline_gen / kStages) & 1)))
        {
        }
        return prefix_run_count(prefix_packed[slot_id][(pipeline_gen / kStages) & 1]);
      };
      // Drain runs [run_begin, run_end) of warp-tile `warp_tile_id`'s staged output into the global arrays.
      // Per run: gather its key from the run's head position -> d_unique, and write its length -> d_counts
      // (= next run's head pos - this run's head pos).
      // The warp tile's last run spans into the next warp-tile, so its length is fixed up separately.
      auto drain = [&](OffT curr_prefix_run_count,
                       int warp_tile_id,
                       int warp_tile_run_base,
                       int warp_tile_run_count,
                       int run_begin,
                       int run_end) {
        // global run index of this warp-tile's run 0 = tile's exclusive prefix + this warp-tile's base within the
        // tile
        const OffT global_run_base = curr_prefix_run_count + warp_tile_run_base;
        const int warp_tile_offset = warp_tile_id * kWarpTileSize; // this warp-tile's base in the staged arrays
        if (warp_tile_run_count < RLE_FW_THRESH)
        {
          // rank-select decode from staged flag words. All shuffles run warp-uniformly (uniform
          // trip counts, no shfl inside predicated paths) -- the cnt-shfl lesson.
          const unsigned my_word = flag_ring[slot_id][warp_tile_id * 32 + lane_id];
          const int my_pc        = __popc(my_word);
          int incl               = my_pc;
#pragma unroll
          for (int o = 1; o < 32; o <<= 1)
          {
            const int p = __shfl_up_sync(kFullMask, incl, o);
            if (lane_id >= o)
            {
              incl += p;
            }
          }
          const int word_excl = incl - my_pc; // lane w: # of runs before word w
          // suffix-min of per-word first-head positions: lane w -> first head in words >= w
          int nxt_min = my_pc ? (lane_id * 32 + __ffs(my_word) - 1) : 0x7fffffff;
#pragma unroll
          for (int o = 1; o < 32; o <<= 1)
          {
            const int c = __shfl_down_sync(kFullMask, nxt_min, o);
            nxt_min     = min(nxt_min, (lane_id + o < 32) ? c : 0x7fffffff);
          }
          const int niter = (run_end - run_begin + 31) >> 5;
          for (int it = 0; it < niter; ++it)
          {
            const int run_idx = run_begin + it * 32 + lane_id;
            // largest w with word_excl(w) <= run_idx (word_excl non-decreasing, excl(0)=0)
            int w = 0;
#pragma unroll
            for (int step = 16; step; step >>= 1)
            {
              const int cand = w + step;
              const int e    = __shfl_sync(kFullMask, word_excl, cand & 31);
              if (cand < 32 && e <= run_idx)
              {
                w = cand;
              }
            }
            const int j           = run_idx - __shfl_sync(kFullMask, word_excl, w);
            const unsigned mw     = __shfl_sync(kFullMask, my_word, w);
            const int nxt_after_w = __shfl_sync(kFullMask, nxt_min, (w + 1) & 31);
            const int pcw         = __popc(mw);
            const int local_bit   = nth_set_bit(mw, (j < pcw) ? j : 0);
            const int local_pos   = w * 32 + local_bit;
            // rank j+1's bit = next set bit above rank j's bit -- no second rank query (measured
            // B200 2026-07-06: +1.3 BWUtil pts seg64, +0.9/+0.7 seg128/256, flat elsewhere).
            // ~1u << local_bit masks bits <= local_bit (well-defined for local_bit 0..31);
            // j+1 < pcw guarantees a higher bit exists whenever the value is consumed.
            const int in_word_nxt = w * 32 + __ffs(mw & (~1u << local_bit)) - 1;
            const int next_local  = (j + 1 < pcw) ? in_word_nxt : nxt_after_w;
            if (run_idx < run_end)
            {
              const int head_pos        = warp_tile_offset + local_pos;
              const OffT global_run_idx = global_run_base + run_idx;
              d_unique[global_run_idx]  = tile_keys[head_pos];
              if (run_idx + 1 < warp_tile_run_count)
              {
                d_counts[global_run_idx] = next_local - local_pos;
              }
            }
          }
          return;
        }
#pragma unroll 2
        for (int run_idx = run_begin + lane_id; run_idx < run_end; run_idx += 32)
        {
          const OffT global_run_idx = global_run_base + run_idx;
          const int head_pos        = (int) run_positions[warp_tile_offset + swizzle_xor_stride32(run_idx)];
          // PROBE: WARP-LOCAL stride-1 gather address of identical cost (WRONG results); pos loads
          // unchanged. Warp-local matters: v32's tile-local fake collapsed all store warps onto the
          // same words and cost -4pt dense by itself. At seg1 this fake == the real pattern exactly
          // (every element is a head), so the dense delta doubles as the probe's sanity check.

          d_unique[global_run_idx] = tile_keys[head_pos]; // gather the run's key at its head position
          if (run_idx + 1 < warp_tile_run_count)
          {
            // within-warp delta (next head - this head); the last run is fixed separately
            const int run_length = (int) run_positions[warp_tile_offset + swizzle_xor_stride32(run_idx + 1)] - head_pos;
            d_counts[global_run_idx] = run_length;
          }
        }
      };
      if constexpr (kNumStoreWarps >= kNumCompWarps)
      {
        // if we have more store warps, each warptile is split between store warps
        constexpr int kStoreWarpsPerWarpTile = kNumStoreWarps / kNumCompWarps;
        const int warp_tile_id               = store_warp_idx / kStoreWarpsPerWarpTile;
        const int sub                        = store_warp_idx % kStoreWarpsPerWarpTile;
        const int warp_tile_run_count        = __shfl_sync(kFullMask, wt_count_ln, warp_tile_id);
        const int warp_tile_run_base         = __shfl_sync(kFullMask, wt_base_ln, warp_tile_id);
        if (warp_tile_run_count >= RLE_RB_MIN
            && warp_tile_run_count <= ((sizeof(KeyT) <= 4) ? RLE_REGBUF : (sizeof(KeyT) == 8 ? 128 : 64)))
        {
          const int run_begin = (int) ((long) warp_tile_run_count * sub / kStoreWarpsPerWarpTile);
          const int run_end   = (int) ((long) warp_tile_run_count * (sub + 1) / kStoreWarpsPerWarpTile);
          // wait for staged_wt (3/3) FIRST -- decode runs before the prefix exists
          while (!ptx::mbarrier_try_wait_parity(
            &staged_wt[slot_id][warp_tile_id], (unsigned) ((pipeline_gen / kStages) & 1)))
          {
          }
          // register-budget scale: big keys hold fewer buffered runs/lane (16B keys: 2 -> 8 regs)
          constexpr int kRegBufCap  = (sizeof(KeyT) <= 4) ? RLE_REGBUF : (sizeof(KeyT) == 8 ? 128 : 64);
          constexpr int kBufPerLane = (kRegBufCap + 31) / 32;
          KeyT buf_key[kBufPerLane];
          int buf_cnt[kBufPerLane];
          const int warp_tile_offset = warp_tile_id * kWarpTileSize;
          const int niter            = (run_end - run_begin + 31) >> 5;
          if (warp_tile_run_count < RLE_FW_THRESH)
          {
            // rank-select decode (same as the classic flag-word path, but into registers)
            const unsigned my_word = flag_ring[slot_id][warp_tile_id * 32 + lane_id];
            const int my_pc        = __popc(my_word);
            int incl               = my_pc;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1)
            {
              const int p = __shfl_up_sync(kFullMask, incl, o);
              if (lane_id >= o)
              {
                incl += p;
              }
            }
            const int word_excl = incl - my_pc;
            int nxt_min         = my_pc ? (lane_id * 32 + __ffs(my_word) - 1) : 0x7fffffff;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1)
            {
              const int c = __shfl_down_sync(kFullMask, nxt_min, o);
              nxt_min     = min(nxt_min, (lane_id + o < 32) ? c : 0x7fffffff);
            }
#pragma unroll
            for (int it = 0; it < kBufPerLane; ++it)
            {
              if (it >= niter)
              {
                break;
              }
              const int run_idx = run_begin + it * 32 + lane_id;
              int w             = 0;
#pragma unroll
              for (int step = 16; step; step >>= 1)
              {
                const int cand = w + step;
                const int e    = __shfl_sync(kFullMask, word_excl, cand & 31);
                if (cand < 32 && e <= run_idx)
                {
                  w = cand;
                }
              }
              const int j           = run_idx - __shfl_sync(kFullMask, word_excl, w);
              const unsigned mw     = __shfl_sync(kFullMask, my_word, w);
              const int nxt_after_w = __shfl_sync(kFullMask, nxt_min, (w + 1) & 31);
              const int pcw         = __popc(mw);
              const int local_bit   = nth_set_bit(mw, (j < pcw) ? j : 0);
              const int local_pos   = w * 32 + local_bit;
              const int in_word_nxt = w * 32 + __ffs(mw & (~1u << local_bit)) - 1;
              // inactive lanes (run_idx >= run_end) decode garbage positions up to wt_offset+1023,
              // which is out of the key window whenever the warp-tile is under 1024 elements (any
              // key type over 4B, or debug kIPT<32) -- predicate the read, keep the shuffles above
              // unconditional
              buf_key[it] = (run_idx < run_end) ? tile_keys[warp_tile_offset + local_pos] : KeyT{};
              buf_cnt[it] = ((j + 1 < pcw) ? in_word_nxt : nxt_after_w) - local_pos;
            }
          }
          else
          {
#pragma unroll
            for (int it = 0; it < kBufPerLane; ++it)
            {
              if (it >= niter)
              {
                break;
              }
              const int run_idx = run_begin + it * 32 + lane_id;
              const bool act    = run_idx < run_end;
              // stale ring shorts can be OOB gather addresses -- clamp inactive lanes to 0
              const int head_pos = act ? (int) run_positions[warp_tile_offset + swizzle_xor_stride32(run_idx)] : 0;
              buf_key[it]        = tile_keys[head_pos];
              buf_cnt[it]        = (act && run_idx + 1 < warp_tile_run_count)
                                   ? (int) run_positions[warp_tile_offset + swizzle_xor_stride32(run_idx + 1)] - head_pos
                                   : 0;
            }
          }
          if (lane_id == 0)
          {
            if constexpr (kPosStages < kStages)
            {
              ptx::mbarrier_arrive(&pos_free[pipeline_gen % kPosStages]); // pos reads done -- BEFORE the prefix wait
            }
            ptx::mbarrier_arrive(&empty[slot_id]); // key reads done too: outputs live in registers
          }
          const OffT global_run_base = wait_prefixed_and_read() + warp_tile_run_base;
#pragma unroll
          for (int it = 0; it < kBufPerLane; ++it)
          {
            if (it >= niter)
            {
              break;
            }
            const int run_idx = run_begin + it * 32 + lane_id;
            if (run_idx < run_end)
            {
              const OffT global_run_idx = global_run_base + run_idx;
              d_unique[global_run_idx]  = buf_key[it];
              if (run_idx + 1 < warp_tile_run_count)
              {
                d_counts[global_run_idx] = buf_cnt[it];
              }
            }
          }
          continue; // pos_free/empty already arrived for this pipeline_gen
        }
        // classic path -- champion order: prefixed wait, then staged_wt, then drain
        const OffT curr_prefix_run_count = wait_prefixed_and_read();
        // wait for staged_wt (3/3): only THIS warp-tile's positions -- not the other 7 compute warps
        while (
          !ptx::mbarrier_try_wait_parity(&staged_wt[slot_id][warp_tile_id], (unsigned) ((pipeline_gen / kStages) & 1)))
        {
        }
        drain(curr_prefix_run_count,
              warp_tile_id,
              warp_tile_run_base,
              warp_tile_run_count,
              (int) ((long) warp_tile_run_count * sub / kStoreWarpsPerWarpTile),
              (int) ((long) warp_tile_run_count * (sub + 1) / kStoreWarpsPerWarpTile));
      }
      else
      {
        const OffT curr_prefix_run_count = wait_prefixed_and_read();
        // fewer store warps than compute regions: each store warp walks whole warptiles
        for (int warp_tile_id = store_warp_idx; warp_tile_id < kNumCompWarps; warp_tile_id += kNumStoreWarps)
        {
          const int warp_tile_run_count = __shfl_sync(kFullMask, wt_count_ln, warp_tile_id);
          const int warp_tile_run_base  = __shfl_sync(kFullMask, wt_base_ln, warp_tile_id);
          while (!ptx::mbarrier_try_wait_parity(
            &staged_wt[slot_id][warp_tile_id], (unsigned) ((pipeline_gen / kStages) & 1)))
          {
          }
          drain(curr_prefix_run_count, warp_tile_id, warp_tile_run_base, warp_tile_run_count, 0, warp_tile_run_count);
        }
      }
      if (lane_id == 0)
      {
        if constexpr (kPosStages < kStages)
        {
          ptx::mbarrier_arrive(&pos_free[pipeline_gen % kPosStages]); // this warp's drain no longer reads the pos slot
        }
        // store done, load may proceed!
        ptx::mbarrier_arrive(&empty[slot_id]);
      }
    }
  }
  // if you are the bookkeeper: ALL prefix-dependent boundary bookkeeping, off the store warps'
  // drain path. IKET showed this block (350-390ns) riding the ONE store warp that gates `empty`
  // in every regime where store binds (seg1-16); as its own warp it runs parallel to the drains
  // and is absorbed by slack where poll binds.
  else
  {
    for (int pipeline_gen = 0;; ++pipeline_gen)
    {
      const int slot_id = pipeline_gen % kStages;
      while (!ptx::mbarrier_try_wait_parity(&computed[slot_id], (unsigned) ((pipeline_gen / kStages) & 1)))
      {
      }
      const int tile_id = tile_id_ring[slot_id];
      if (tile_id >= num_tiles)
      {
        if (lane_id == 0)
        {
          ptx::mbarrier_arrive(&empty[slot_id]);
        }
        break;
      }
      const int tile_len = (int) min((OffT) kTileSize, num_items - (OffT) tile_id * kTileSize);
      const bool is_last = (tile_id == num_tiles - 1);
      // per-warp-tile counts/bases in registers, same scan as the store warps (lane i = warp-tile i)
      const int wt_count_ln = (lane_id < kNumCompWarps) ? warp_run_counts[slot_id][lane_id] : 0;
      int wt_incl_ln        = wt_count_ln;
#pragma unroll
      for (int offset = 1; offset < kNumCompWarps; offset <<= 1)
      {
        const int p = __shfl_up_sync(kFullMask, wt_incl_ln, offset);
        if (lane_id >= offset)
        {
          wt_incl_ln += p;
        }
      }
      const int wt_base_ln        = wt_incl_ln - wt_count_ln;
      const int tile_total_runs   = __shfl_sync(kFullMask, wt_incl_ln, kNumCompWarps - 1);
      const unsigned wt_runs_mask = __ballot_sync(kFullMask, wt_count_ln > 0);
      while (!ptx::mbarrier_try_wait_parity(&prefixed[slot_id], (unsigned) ((pipeline_gen / kStages) & 1)))
      {
      }
      const PrefixT packed_prefix        = prefix_packed[slot_id][(pipeline_gen / kStages) & 1];
      const OffT curr_prefix_run_count   = prefix_run_count(packed_prefix);
      const OffT curr_prefix_open_length = prefix_open_len(packed_prefix);
      // per-warp-tile boundary: a warp-tile's last run is closed by the next nonempty warp-tile's
      // first head. lane L handles warp-tile L.
      if (lane_id < kNumCompWarps && wt_count_ln > 0)
      {
        const unsigned later_wts       = wt_runs_mask >> (lane_id + 1); // nonempty warp-tiles after L
        const OffT last_run_global_idx = curr_prefix_run_count + wt_base_ln + wt_count_ln - 1;
        if (later_wts)
        {
          const int next_wt             = lane_id + 1 + __ffs(later_wts) - 1;
          d_counts[last_run_global_idx] = warp_first_heads[slot_id][next_wt] - warp_last_heads[slot_id][lane_id];
        }
        else if (is_last)
        {
          // if we are the last warptile of the whole input, we end here
          d_counts[last_run_global_idx] = tile_len - warp_last_heads[slot_id][lane_id];
        }
        // else: this run is open in this tile, now this became a job for the next tile (see below)
      }
      // now we need to finish last tile's open run
      if (lane_id == 0)
      {
        const bool any_head  = (wt_runs_mask != 0);
        const int first_head = any_head ? warp_first_heads[slot_id][__ffs(wt_runs_mask) - 1] : -1;
        // if our tile has a head, i.e. it stops here
        if (any_head && curr_prefix_run_count > 0)
        {
          d_counts[curr_prefix_run_count - 1] = curr_prefix_open_length + first_head;
        }
        // if we are last tile with no head: we have to close it here
        if (is_last && !any_head && curr_prefix_run_count > 0)
        {
          d_counts[curr_prefix_run_count - 1] = curr_prefix_open_length + tile_len;
        }
        // otherwise, next tile's problem
        if (is_last)
        {
          *d_num_runs = (NumRunsT) (curr_prefix_run_count + tile_total_runs);
        }
        ptx::mbarrier_arrive(&empty[slot_id]); // bookkeeping done, slot may recycle
      }
    }
  }
}

inline void persistent_rle_launch(
  const KeyT* d_keys,
  KeyT* d_unique,
  LenT* d_counts,
  NumRunsT* d_num_runs,
  TilePartialStateT* tile_state,
  const unsigned* d_launch_gen,
  OffT num_items,
  int num_tiles,
  cudaStream_t stream)
{
  // raise the dynamic-smem cap once (idempotent; kept off the per-launch path)
  static const bool smem_cap_set = [] {
    cudaFuncSetAttribute(persistent_rle, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) kDynSmem);
    return true;
  }();
  (void) smem_cap_set;

  // no per-launch clear of tile_state: states are generation-tagged (see publish_state) and the
  // generation lives in DEVICE memory (see the init kernel in rle_dispatch.cuh) -- no host-side
  // state, so calls are CUDA-graph-capturable and thread-safe per temp-storage allocation.
  const int blocks = num_tiles;

  persistent_rle<<<blocks, kNumThreads, kDynSmem, stream>>>(
    d_keys, d_unique, d_counts, d_num_runs, tile_state, d_launch_gen, num_items, num_tiles);
}
} // namespace rle_impl
