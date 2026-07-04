// Standalone persistent-block RLE-encode kernel (int keys) for Blackwell (B200).
// Impl only -- see persistent_rle_bench.cu for the official-nvbench comparison vs cub::DeviceRunLengthEncode.
// Types are hard-coded to int for now (keys/counts/offsets all int32).
#pragma once

#include <cuda/atomic>
#include <cuda/ptx>
#include <cuda/std/cstdint>

#include <algorithm> // std::min (host launcher)

#include <cuda_runtime_api.h>

namespace ptx = cuda::ptx;
using u64     = cuda::std::uint64_t;

#ifndef K_IPT
#  define K_IPT 32
#endif
constexpr int kIPT = K_IPT; // FP32s/thread; tile = kNumCompWarps*kIPT*32 = 8192 (small K_IPT: local-GPU debug)
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
// positions ring depth: positions are written at staging and consumed by the drain ~2 gens later,
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
#ifndef RLE_DRAIN_UNROLL4
#  define RLE_DRAIN_UNROLL4 0
#endif
#ifndef RLE_DRAIN_UNROLL2
#  define RLE_DRAIN_UNROLL2 (!RLE_DRAIN_UNROLL4)
#endif

#ifndef RLE_POLL_ORACLE
#  define RLE_POLL_ORACLE 0 // 1 = WRONG RESULTS: free-prefix ceiling probe (skips the poll fold)
#endif

// RLE_BK_EARLY_ARRIVE: bookkeeper reads all slot metadata into registers BEFORE the prefix wait and
//   arrives `empty` right after reading the prefix -- its global writes come off the recycle chain.
// RLE_STCS_OUT: streaming (evict-first) stores for d_unique/d_counts -- keeps ~2GB of output from
//   churning the L2 lines the prefix states and staged data live in.
#ifndef RLE_BK_EARLY_ARRIVE
#  define RLE_BK_EARLY_ARRIVE 0
#endif
#ifndef RLE_STCS_OUT
#  define RLE_STCS_OUT 0
#endif
// RLE_GLOBAL_GATHER: drain gathers run keys from GLOBAL d_keys (L2-hot: the tile just streamed
// through) instead of smem tile_keys. The smem gather is bank-conflicted at seg4-16 (heads sit
// 4-16 elements apart -> few banks); global loads have no bank structure and dense stays coalesced.
#ifndef RLE_GLOBAL_GATHER
#  define RLE_GLOBAL_GATHER 0
#endif
// RLE_POLL_BACKOFF: ns to __nanosleep when a full reload pass still has missing frontier states.
// Probe+fix for the L2 state-line hammering theory of the mid-regime oracle residual: 148 polls
// spin-loading the same hot lines with strong loads. 0 = off (spin hard, today's behavior).
#ifndef RLE_POLL_BACKOFF
#  define RLE_POLL_BACKOFF 0
#endif
// RLE_CNT_SHFL: derive run counts from the NEIGHBOR lane's position (shfl_down) instead of a second
// swizzled LDS.16 at run_idx+1 -- one pos load per run instead of two (lane-31 boundary comes from
// software-pipelining the next iteration's load). Trades a sometimes-conflicted LDS for a
// never-conflicted shuffle on the same MIO pipe.
#ifndef RLE_CNT_SHFL
#  define RLE_CNT_SHFL 0
#endif
// RLE_PROBE_GATHER_FREE / RLE_PROBE_PEEL_FREE: WRONG-RESULTS perf probes (like RLE_POLL_ORACLE).
// Replace the two measured bank-conflict hot spots (v31: gather line ~951K, peel STS ~915K
// excessive wavefronts at seg8) with conflict-free addresses of identical instruction cost.
// The A/B delta vs base IS the harvestable conflict prize -- measure before restructuring.
#ifndef RLE_PROBE_GATHER_FREE
#  define RLE_PROBE_GATHER_FREE 0
#endif
#ifndef RLE_PROBE_PEEL_FREE
#  define RLE_PROBE_PEEL_FREE 0
#endif
// RLE_FLAGWORD: adaptive staging. Warp-tiles with run count < RLE_FW_THRESH stage their 32 raw
// head-flag words (one conflict-free STS.32 per lane; no scan, no peel, no pos-ring use) and the
// drain rank-selects head positions from them (5-shfl binary search + branchless nth-set-bit, pure
// registers, no LDS in the run loop). Dense warp-tiles keep the champion peel+positions path:
// decode cost scales with runs, the staging saving doesn't. Both sides derive the SAME predicate
// from the staged warp run count, so no mode flag is exchanged. Collects the measured peel-write
// conflict prize (v33: +0.4..+1.3 at seg4-16) and deletes staging work + pos_free coupling at mid.
#ifndef RLE_FLAGWORD
#  define RLE_FLAGWORD 1 // champion since v35: strictly >= s5p3 at all 11 regimes
#endif
#ifndef RLE_FW_THRESH
#  define RLE_FW_THRESH 48 // decode crossover ~32-64 runs/warp-tile; 64/96 regress seg32
#endif
// RLE_REGBUF: prefix-decoupled drain. Warp-tiles with run count <= RLE_REGBUF decode+gather into
// REGISTERS before the prefixed wait (only the output ADDRESS needs the prefix), then release the
// pos slot AND the key slot -- `empty` stops waiting on the prefix chain entirely for buffered
// warp-tiles; the final store burst trails the pipeline by the prefix latency and nothing
// downstream orders on it. Early `empty` breaks the old "poll never overwrites a live
// prefix_packed" ordering proof; prefix_packed is double-buffered by slot-cycle parity instead
// (program order proves safety). 0 = off. Buffer = RLE_REGBUF/32 int pairs per lane (registers).
#ifndef RLE_REGBUF
#  define RLE_REGBUF 256 // champion since v40: +0.8..+2.0 across seg8-256 for -0.3 dense/1M
#endif
// buffered drain only pays when the drain is long enough for early slot-release to matter;
// near-empty warp-tiles (long segs) stay classic to avoid pure reshape overhead
#ifndef RLE_RB_MIN
#  define RLE_RB_MIN 8
#endif
// RLE_SEQ_EARLY: give POLL its own tile-id barrier armed at TMA-ISSUE time instead of waiting for
// TMA COMPLETION. The fold reads only tile_seq[slot] (written before the TMA is issued), never the
// keys -- waiting on full[] wastes a whole TMA flight (~600ns) of frontier latency per gen, and the
// prefixed publish shifts earlier by that much device-wide. Recycling safety is unchanged: the
// barrier is armed only after the empty[] wait, exactly like full[], so the "poll never overwrites
// a live prefix_packed slot" proof still holds.
#ifndef RLE_SEQ_EARLY
#  define RLE_SEQ_EARLY 0
#endif
// RLE_TMA_SPLIT: deliver each tile as TWO self-contained half-TMAs (each with its own 16B left-pad,
// so warp kNumCompWarps/2's predecessor element arrives in half-2's pad). Compute warps of half 1
// start flagging ~a half-TMA earlier -- attacks the TMA->all-compute serialization in the gen floor.
#ifndef RLE_TMA_SPLIT
#  define RLE_TMA_SPLIT 0
#endif
#ifndef RLE_L2_POLICY
// persisting accessPolicyWindow on tile_partial_states: +0.6pt at 2^20, +0.45 at 4096, floor +0.1,
// but costs -0.2..-0.5 at seg2-16 (carveout size irrelevant; the trade is inherent to the window).
// seg2-16 are priority regimes => default OFF; enable for long-run-leaning workloads.
#  define RLE_L2_POLICY 0
#endif

// This is important for position staging on dense cases (16 way bank conflicts).
// RLE_SWZ selects the mix: 0 = x^(x>>5) (ncu-tuned for DENSE head patterns; measured 4.6x MORE STS
// conflicts at seg2 whose run indices stride differently), 1 = x^(x>>4), 2 = x^(x>>4)^(x>>5).
// All variants are bijections, so correctness is layout-independent (compute writes / store reads
// through this same function).
#ifndef RLE_SWZ
#  define RLE_SWZ 0
#endif
__device__ __forceinline__ int swizzle_xor_stride32(int x)
{
#if RLE_SWZ == 1
  return x ^ (x >> 4);
#elif RLE_SWZ == 2
  return x ^ (x >> 4) ^ (x >> 5);
#else
  return x ^ (x >> 5);
#endif
}

// CLC = 1 => use shiny new blackwell feature (UGETNEXTWORKID)
// CLC = 0 => use atomics for work stealing
// no perf difference observed on blackwell
#ifndef USE_CLC
#  define USE_CLC 1
#endif

constexpr int kWarpTileSize = 32 * kIPT;
constexpr int kTileSize     = kNumCompWarps * kWarpTileSize;
static_assert(kTileSize <= 0xffff, "per-tile run_count/open_len must fit the 16-bit state-word fields");
constexpr int kNumWarps          = 1 /*load*/ + kNumCompWarps + 1 /*poll*/ + kNumStoreWarps + 1 /*bookkeeper*/;
constexpr int kNumThreads        = kNumWarps * 32;
constexpr unsigned kFullMask     = 0xffffffffu;
constexpr int poll_warp_id       = 1 + kNumCompWarps;
constexpr int store_warp_id      = poll_warp_id + 1;
constexpr int bookkeeper_warp_id = store_warp_id + kNumStoreWarps;
// for each input tile, we need to store the keys (I32) and in tile position
// for in tile position we can just do U16 since tile size is never bigger than 2^16.
// each key slot carries kSlotPad extra leading ints: the TMA over-fetches one 16B chunk to the
// left, so slot[0..kSlotPad-1] hold the previous tile's last keys and element 0's predecessor
// is slot[kSlotPad-1] -- no separate (blocking) LDG of the previous tile's last key needed.
constexpr int kSlotPad = 4; // ints; 16 bytes = cp_async_bulk alignment quantum
#if RLE_TMA_SPLIT
constexpr int kHalfTile   = kTileSize / 2;
constexpr int kSlotStride = kTileSize + 2 * kSlotPad; // [pad0|half0|pad1|half1]
static_assert(kNumCompWarps % 2 == 0 && kHalfTile % 4 == 0, "split needs even halves");
#else
constexpr int kSlotStride = kTileSize + kSlotPad;
#endif
constexpr size_t kDynSmem =
  (size_t) kStages * kSlotStride * sizeof(int) + (size_t) kPosStages * kTileSize * sizeof(short);

// tile_partial_states is one 64-bit word [launch_gen:32][open_len:16][run_count:16].
// run_count and open_len are per-tile so both fit 16 bits; the high half holds the launch
// generation, so a word is "published" iff its gen matches this launch -- stale words from prior
// launches read as unpublished and the per-launch cudaMemset of the state array is not needed
// (the array is zeroed ONCE at allocation; gen starts at 1 so zero never matches).
// an aligned 64-bit access is already non-tearing, but atomic_ref doesn't hurt and has clear semantics
__device__ __forceinline__ void
publish_state(u64* tile_state_arr, int tile_idx, unsigned launch_gen, int run_count, int open_len)
{
  u64 w = ((u64) launch_gen << 32) | ((u64) (unsigned) open_len << 16) | (u64) (unsigned) run_count;
  cuda::atomic_ref<u64, cuda::thread_scope_device> a(tile_state_arr[tile_idx]);
  a.store(w, cuda::memory_order_relaxed);
}

// non-blocking single load of the raw packed word (valid bit may be 0)
__device__ __forceinline__ u64 load_state(u64* tile_state_arr, int tile_idx)
{
  cuda::atomic_ref<u64, cuda::thread_scope_device> a(tile_state_arr[tile_idx]);
  return a.load(cuda::memory_order_relaxed);
}

// Computes the exclusive prefix of the tile_id, i.e. the aggregate over tiles [0, tile_id)
// we do this by keeping the prefixes of the last generation and poll the partial states in
// [last_seen_tile_id, tile_id) and fold it with the aggregate we held
// position of the n-th (0-indexed) set bit of m; branchless popc bisect. Requires popc(m) > n.
__device__ __forceinline__ int nth_set_bit(unsigned m, int n)
{
  int pos = 0;
  int c   = __popc(m & 0xffffu);
  if (n >= c)
  {
    n -= c;
    pos += 16;
    m >>= 16;
  }
  c = __popc(m & 0xffu);
  if (n >= c)
  {
    n -= c;
    pos += 8;
    m >>= 8;
  }
  c = __popc(m & 0xfu);
  if (n >= c)
  {
    n -= c;
    pos += 4;
    m >>= 4;
  }
  c = __popc(m & 0x3u);
  if (n >= c)
  {
    n -= c;
    pos += 2;
    m >>= 2;
  }
  if (n >= (int) (m & 1u))
  {
    pos += 1;
  }
  return pos;
}

__device__ __forceinline__ void poll_and_fold(
  u64* tile_partial_states,
  unsigned launch_gen,
  int tile_id,
  int& last_seen_tile_id,
  int& last_seen_prefix_run_count,
  int& last_seen_prefix_open_length,
  int lane_id,
  int& curr_prefix_run_count,
  int& curr_prefix_open_length)
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
    // MLP: issue all kPollMlp loads up front, then spin until this lane's owned tiles are all published.
    // published words are immutable within a launch, so RE-load only the still-missing ones: while the
    // warp spins on a frontier straggler, the (up to 127) already-valid states must NOT be re-fetched
    // from L2 every spin iteration.
    u64 packed_words[kPollMlp] = {}; // gen 0 never matches (launch_gen >= 1) => starts "missing"
    bool ready;
    do
    {
      ready = true;
#pragma unroll
      for (int i = 0; i < kPollMlp; ++i)
      {
        if (i < lane_tile_count && (unsigned) (packed_words[i] >> 32) != launch_gen)
        {
          packed_words[i] = load_state(tile_partial_states, lane_base + i);
          if ((unsigned) (packed_words[i] >> 32) != launch_gen)
          {
            ready = false;
          }
        }
      }
#if RLE_POLL_BACKOFF
      if (__ballot_sync(kFullMask, !ready) == 0u)
      {
        break;
      }
      __nanosleep(RLE_POLL_BACKOFF);
    } while (true);
#else
    } while (__ballot_sync(kFullMask, !ready) != 0u);
#endif
    // ordered reduce this lane's own tiles (increasing left -> right)
    int lane_run_count = 0, lane_open_length = 0;
#pragma unroll
    for (int i = 0; i < kPollMlp; ++i)
    {
      if (i < lane_tile_count)
      {
        const int tile_run_count   = (int) (packed_words[i] & 0xffffu);
        const int tile_open_length = (int) ((packed_words[i] >> 16) & 0xffffu);
        lane_run_count             = lane_run_count + tile_run_count;
        lane_open_length           = (tile_run_count > 0) ? tile_open_length : (lane_open_length + tile_open_length);
      }
    }
    // cross lane fold over 32 lane aggregates -- only the chunk TOTAL is needed, so no scan:
    // total run count is a plain sum; total open length follows the fold's absorbing rule: any run
    // resets the open count, so it's the sum of lane opens from the LAST run-bearing lane onward
    // (that lane's own open already sits after its last run), or the sum of all opens if no lane
    // has a run. lanes beyond lane_tile_count hold (0,0) and contribute nothing.
    // (a 15-shfl segmented inclusive scan whose lane-31 value was the only consumed result used to
    // sit here; ballot+redux measured 1.6-6.9% faster end-to-end at seg>=64)
    const int chunk_run_count      = __reduce_add_sync(kFullMask, lane_run_count);
    const unsigned lanes_with_runs = __ballot_sync(kFullMask, lane_run_count > 0);
    const int last_run_lane        = lanes_with_runs ? (31 - __clz(lanes_with_runs)) : 0;
    const int chunk_open_length    = __reduce_add_sync(kFullMask, (lane_id >= last_run_lane) ? lane_open_length : 0);
    // combine last_seen_prefix with the chunk aggregate
    const int new_run_count = last_seen_prefix_run_count + chunk_run_count;
    const int new_open_length =
      (chunk_run_count > 0) ? chunk_open_length : (last_seen_prefix_open_length + chunk_open_length);
    last_seen_prefix_run_count   = new_run_count;
    last_seen_prefix_open_length = new_open_length;
    last_seen_tile_id += chunk;
  }
  curr_prefix_run_count   = last_seen_prefix_run_count;
  curr_prefix_open_length = last_seen_prefix_open_length;
}

// we aim for 1 block/SM since it is easier to manage resources: do not need to worry about occupancy anymore
__launch_bounds__(kNumThreads, 1) __global__ void persistent_rle(
  const int* __restrict__ d_keys,
  int* __restrict__ d_unique,
  int* __restrict__ d_counts,
  int* __restrict__ d_num_runs,
  u64* __restrict__ tile_partial_states,
#if !USE_CLC
  int* __restrict__ global_tile_counter, // global work steal counter when there is no CLC
#endif
  unsigned launch_gen, // >=1; stamps this launch's tile states (see publish_state)
  int num_items,
  int num_tiles)
{
  // [kStages][kTileSize] int32 ring (input keys)
  // [kStages][kTileSize] int16 staged head positions
  extern __shared__ int tile_buf[];
  short* const staged_pos = (short*) (tile_buf + (size_t) kStages * kSlotStride);
  __shared__ int tile_seq[kStages]; // which global tile each ring slot holds (LOAD gets it with try_cancel)
  __shared__ int warp_run_counts[kStages][kNumCompWarps]; // per compute warp run counts
#if RLE_FLAGWORD
  __shared__ unsigned flag_ring[kStages][kNumCompWarps * 32]; // staged head-flag words (fw path)
#endif
  __shared__ int warp_first_heads[kStages][kNumCompWarps]; // per compute warp first head idx (-1 if none)
  __shared__ int warp_last_heads[kStages][kNumCompWarps]; // per compute warp last head idx (-1 if none)
  // POLL -> STORE handoff: [open_len_prefix:32][run_count_prefix:32] packed, one access per side
  // under REGBUF the prefix is double-buffered by slot-cycle parity: POLL(g+kStages) writes the
  // OTHER half from the one STORE(g)/BOOKKEEPER(g) read, and program order proves safety (a
  // store warp reads gen g's half before its gen g+kStages empty-arrive, which gates the load
  // that gates poll's next write to that half). No consumed-barrier -- v38's pfx_free put the
  // store wake-up latency inside the POLL's serial loop and cost -2..-5pt at seg128+.
  __shared__ u64 prefix_packed[kStages][RLE_REGBUF ? 2 : 1];
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
#if RLE_TMA_SPLIT
  __shared__ u64 full[kStages][2]; // one per half-TMA; compute warp w waits its half, POLL waits half 0
#else
  __shared__ u64 full[kStages][1];
#endif
  __shared__ u64 computed[kStages], prefixed[kStages], empty[kStages];
#if RLE_SEQ_EARLY
  __shared__ u64 seq_ready[kStages]; // tile_seq[slot] valid -- armed at TMA issue, POLL's trigger
#endif
  // COMPUTE warp w --staged_wt[w]--> STORE: per-warp-tile handoff, so store warps drain a warp-tile
  // as soon as ITS positions are staged instead of waiting for all 8 compute warps (warp 0 is always
  // last -- it does the publish fan-in first). The shared metadata store also needs (run counts,
  // first/last heads, tile_seq) is covered by `computed`.
  __shared__ u64 staged_wt[kStages][kNumCompWarps];

#if USE_CLC
  // try_cancel writes a 16-byte response into clc_resp + completes clc_bar's tx.
  __shared__ __align__(16) uint4 clc_resp;
  __shared__ u64 clc_bar;
#endif

  const int thr_id  = threadIdx.x;
  const int warp_id = thr_id >> 5;
  const int lane_id = thr_id & 31;
  const int blk_id  = blockIdx.x;

  if (thr_id == 0)
  {
    for (int slot_id = 0; slot_id < kStages; ++slot_id)
    {
      for (int h = 0; h < (RLE_TMA_SPLIT ? 2 : 1); ++h)
      {
        ptx::mbarrier_init(&full[slot_id][h], 1);
      }
      ptx::mbarrier_init(&computed[slot_id], kNumCompWarps); // every compute warp arrives
#if RLE_SEQ_EARLY
      ptx::mbarrier_init(&seq_ready[slot_id], 1);
#endif
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

#if USE_CLC
    ptx::mbarrier_init(&clc_bar, 1); // 1 arrival
#endif
  }
  // normal smem writes (e.g. mbarrier_init) go through the generic proxy
  // the TMA operations access shared memory through the async proxy. these are separate visibility domains,
  // so the init writes are not automatically visible to TMA.
  ptx::fence_proxy_async(ptx::space_shared);
  __syncthreads();

  // if you are load
  if (warp_id == 0)
  {
#if USE_CLC
    // CLC tile assignment: gen0 tile = this CTA's launch id (blockIdx.x)
    int tile_id = blk_id;
    if (lane_id == 0)
    {
      // 16 is the try_cancel byte tx
      ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &clc_bar, 16);
      ptx::clusterlaunchcontrol_try_cancel(&clc_resp, &clc_bar);
    }
#endif
    for (int gen = 0;; ++gen)
    {
      const int slot_id  = gen % kStages; // which slot is this?
      const int slot_gen = gen / kStages; // how many times is this slot used?
      if (gen >= kStages)
      {
        // need to wait for slot to be free
        while (!ptx::mbarrier_try_wait_parity(&empty[slot_id], (unsigned) ((slot_gen - 1) & 1)))
        {
        }
      }
#if !USE_CLC
      // work-steal: grab the next global tile via the atomic counter (one tile per atomic)
      int tile_id = 0;
      if (lane_id == 0)
      {
        tile_id = atomicAdd(global_tile_counter, 1);
      }
      tile_id = __shfl_sync(kFullMask, tile_id, 0);
#endif
      if (lane_id == 0)
      {
        tile_seq[slot_id] = tile_id;
#if RLE_SEQ_EARLY
        ptx::mbarrier_arrive(&seq_ready[slot_id]); // covers the sentinel path too (write precedes it)
#endif
      }
      if (tile_id >= num_tiles)
      {
        if (lane_id == 0)
        {
          for (int h = 0; h < (RLE_TMA_SPLIT ? 2 : 1); ++h)
          {
            ptx::mbarrier_arrive(&full[slot_id][h]);
          }
        }
        __syncwarp();
        break;
      }
      if (lane_id == 0)
      {
        // over-fetch one 16B chunk to the left: slot[0..kSlotPad-1] = previous tile's last keys,
        // so COMPUTE reads element 0's predecessor from smem instead of a blocking LDG here.
        // tile 0 has no predecessor and skips the over-fetch; its slot[0..kSlotPad-1] stay
        // unread because is_global_first forces element 0's head flag.
        const bool first_tile = (tile_id == 0);
#if RLE_TMA_SPLIT
        int* const slot_base = tile_buf + (size_t) slot_id * kSlotStride;
        // half 0: [pad0|half0]
        const unsigned nb0 = (unsigned) ((kHalfTile + (first_tile ? 0 : kSlotPad)) * sizeof(int));
        ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &full[slot_id][0], nb0);
        ptx::cp_async_bulk(
          ptx::space_shared,
          ptx::space_global,
          slot_base + (first_tile ? kSlotPad : 0),
          d_keys + (size_t) tile_id * kTileSize - (first_tile ? 0 : kSlotPad),
          nb0,
          &full[slot_id][0]);
        // half 1: [pad1|half1], pad1 = elements kHalfTile-4..kHalfTile-1 (always in-bounds, even tile 0)
        const unsigned nb1 = (unsigned) ((kHalfTile + kSlotPad) * sizeof(int));
        ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &full[slot_id][1], nb1);
        ptx::cp_async_bulk(
          ptx::space_shared,
          ptx::space_global,
          slot_base + kSlotPad + kHalfTile,
          d_keys + (size_t) tile_id * kTileSize + kHalfTile - kSlotPad,
          nb1,
          &full[slot_id][1]);
#else
        const unsigned nbytes = (unsigned) ((kTileSize + (first_tile ? 0 : kSlotPad)) * sizeof(int));
        ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &full[slot_id][0], nbytes);
        ptx::cp_async_bulk(
          ptx::space_shared,
          ptx::space_global,
          tile_buf + (size_t) slot_id * kSlotStride + (first_tile ? kSlotPad : 0),
          d_keys + (size_t) tile_id * kTileSize - (first_tile ? 0 : kSlotPad),
          nbytes,
          &full[slot_id][0]);
#endif
      }
      __syncwarp();
#if USE_CLC
      // consume the prefetched cancel
      // this is ok since it should be fast to get next cancelled id
      if (lane_id == 0)
      {
        while (!ptx::mbarrier_try_wait_parity(&clc_bar, (unsigned) (gen & 1)))
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
#endif
    }
  }
  // if you are compute
  else if (warp_id <= kNumCompWarps)
  {
    const int compute_warp_id = warp_id - 1;
    const int warp_tile_base  = compute_warp_id * kWarpTileSize;
    for (int gen = 0;; ++gen)
    {
      const int slot_id  = gen % kStages;
      const int slot_gen = gen / kStages;
#if RLE_TMA_SPLIT
      const int my_half = (compute_warp_id >= kNumCompWarps / 2) ? 1 : 0;
#else
      const int my_half = 0;
#endif
      while (!ptx::mbarrier_try_wait_parity(&full[slot_id][my_half], (unsigned) (slot_gen & 1)))
      {
      }
      const int tile_id = tile_seq[slot_id];
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
      const int* key_buf  = tile_buf + (size_t) slot_id * kSlotStride + kSlotPad + (my_half ? kSlotPad : 0);
      const int tile_len  = min(kTileSize, num_items - tile_id * kTileSize);
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
      short* const pos_dst = staged_pos + (size_t) (gen % kPosStages) * kTileSize;
      unsigned my_flags    = 0;
#pragma unroll
      for (int iter = 0; iter < kIPT; ++iter)
      {
        const int loc             = warp_tile_base + iter * 32 + lane_id;
        const int key             = (loc < tile_len) ? key_buf[loc] : 0;
        const int pred            = key_buf[loc - 1]; // loc==0 reads the over-fetched slot[kSlotPad-1]
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
#if RLE_FLAGWORD
      const bool stage_flags = (local_run_count < RLE_FW_THRESH);
      if (stage_flags)
      {
        flag_ring[slot_id][compute_warp_id * 32 + lane_id] = my_flags;
      }
      else
#endif
      {
        if constexpr (kPosStages < kStages)
        {
          // the pos slot is shared by gens g, g+kPosStages, ...: wait for the drains of gen
          // g-kPosStages to have finished reading it (satisfied by pipeline offset in steady state)
          if (gen >= kPosStages)
          {
            while (!ptx::mbarrier_try_wait_parity(&pos_free[gen % kPosStages], (unsigned) ((gen / kPosStages - 1) & 1)))
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
#if RLE_PROBE_PEEL_FREE
              // PROBE: lane-major slot (lane_id + 32*step) -- stride-1 shorts across lanes at each
              // peel step = conflict-free; drain then reads garbage positions (WRONG results)
              pos_dst[warp_tile_base + lane_id + ((run_index - word_run_base) << 5)] = (short) (word_pos + head_offset);
#else
              pos_dst[warp_tile_base + swizzle_xor_stride32(run_index)] = (short) (word_pos + head_offset);
#endif
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
    int last_seen_tile_id = 0, last_seen_prefix_run_count = 0, last_seen_prefix_open_length = 0;
    for (int gen = 0;; ++gen)
    {
      const int slot_id  = gen % kStages;
      const int slot_gen = gen / kStages;
#if RLE_SEQ_EARLY
      while (!ptx::mbarrier_try_wait_parity(&seq_ready[slot_id], (unsigned) (slot_gen & 1)))
#else
      while (!ptx::mbarrier_try_wait_parity(&full[slot_id][0], (unsigned) (slot_gen & 1)))
#endif
      {
      }
      const int tile_id = tile_seq[slot_id];
      if (tile_id >= num_tiles)
      {
        if (lane_id == 0)
        {
          ptx::mbarrier_arrive(&prefixed[slot_id]); // drain
        }
        break;
      }
      int curr_prefix_run_count, curr_prefix_open_length;
#if RLE_POLL_ORACLE
      // CEILING PROBE ONLY -- WRONG RESULTS. Skips the cross-tile fold to measure the free-prefix
      // ceiling. v2: fake prefixes SPREAD like real ones (tile_id x tile-0's run count, read once)
      // so output writes distribute realistically -- the prefix=0 version contaminated run-heavy
      // regimes with cross-SM same-address write contention.
      if (last_seen_prefix_run_count == 0) // reused as the cached per-tile estimate (probe-only)
      {
        u64 s0 = 0;
        do
        {
          s0 = load_state(tile_partial_states, 0);
        } while ((unsigned) (s0 >> 32) != launch_gen);
        last_seen_prefix_run_count = (int) (s0 & 0xffffu) | 1; // >=1 so we don't re-poll
      }
      const long long est_pref = (long long) tile_id * last_seen_prefix_run_count;
      curr_prefix_run_count    = (est_pref < 0x40000000ll) ? (int) est_pref : 0x40000000;
      curr_prefix_open_length  = 0;
      last_seen_tile_id        = tile_id;
      (void) last_seen_prefix_open_length;
#else
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
#endif
      // no wait needed before overwriting the prefix slot: STORE(gen-kStages) arrives `empty` as its
      // LAST act, LOAD cannot arm `full` for this gen until it passed that same `empty` phase, and we
      // already passed `full` above -- so a drain-wait here could provably never spin.
      if (lane_id == 0)
      {
        prefix_packed[slot_id][RLE_REGBUF ? (slot_gen & 1) : 0] =
          ((u64) (unsigned) curr_prefix_open_length << 32) | (unsigned) curr_prefix_run_count;
        ptx::mbarrier_arrive(&prefixed[slot_id]); // prefix ready, store may proceed! (2/2)
      }
    }
  }
  // if you are store
  else if (warp_id < bookkeeper_warp_id)
  {
    const int store_warp_idx = warp_id - store_warp_id;
    for (int gen = 0;; ++gen)
    {
      const int slot_id = gen % kStages;
      // wait for computed (1/3): all per-warp-tile metadata (run counts, first/last heads) is
      // written before each compute warp's `computed` arrive -- positions are NOT needed yet
      while (!ptx::mbarrier_try_wait_parity(&computed[slot_id], (unsigned) ((gen / kStages) & 1)))
      {
      }
      const int tile_id = tile_seq[slot_id];
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
      const int wt_base_ln = wt_incl_ln - wt_count_ln; // lane i: warp-tile i's exclusive run base
#if RLE_GLOBAL_GATHER
      const int* tile_keys = d_keys + (size_t) tile_id * kTileSize; // L2-hot; no smem bank structure
#else
      const int* tile_keys = tile_buf + (size_t) slot_id * kSlotStride + kSlotPad;
#endif
#if RLE_TMA_SPLIT
#  define RLE_KEY_AT(p) tile_keys[(p) + (((p) >= kHalfTile) ? kSlotPad : 0)]
#else
#  define RLE_KEY_AT(p) tile_keys[(p)]
#endif
      // staged positions
      const short* run_positions = staged_pos + (size_t) (gen % kPosStages) * kTileSize;
      // wait for prefixed (2/3); drains only need the run-count prefix (addresses) -- the
      // open-length half is bookkeeper-only
      auto wait_prefixed_and_read = [&]() {
        while (!ptx::mbarrier_try_wait_parity(&prefixed[slot_id], (unsigned) ((gen / kStages) & 1)))
        {
        }
        return (int) (prefix_packed[slot_id][RLE_REGBUF ? ((gen / kStages) & 1) : 0] & 0xffffffffull);
      };
      // Drain runs [run_begin, run_end) of warp-tile `warp_tile_id`'s staged output into the global arrays.
      // Per run: gather its key from the run's head position -> d_unique, and write its length -> d_counts
      // (= next run's head pos - this run's head pos).
      // The warp tile's last run spans into the next warp-tile, so its length is fixed up separately.
      auto drain =
        [&](int curr_prefix_run_count,
            int warp_tile_id,
            int warp_tile_run_base,
            int warp_tile_run_count,
            int run_begin,
            int run_end) {
          // global run index of this warp-tile's run 0 = tile's exclusive prefix + this warp-tile's base within the
          // tile
          const int global_run_base  = curr_prefix_run_count + warp_tile_run_base;
          const int warp_tile_offset = warp_tile_id * kWarpTileSize; // this warp-tile's base in the staged arrays
#if RLE_FLAGWORD
          if (warp_tile_run_count < RLE_FW_THRESH)
          {
            // rank-select decode from staged flag words. All shuffles run warp-uniformly (uniform
            // trip counts, no shfl inside predicated paths) -- the cnt-shfl lesson.
            const unsigned my_word = flag_ring[slot_id][warp_tile_id * 32 + lane_id];
            const int my_pc        = __popc(my_word);
            int incl               = my_pc;
#  pragma unroll
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
#  pragma unroll
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
#  pragma unroll
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
              const int local_pos   = w * 32 + nth_set_bit(mw, (j < pcw) ? j : 0);
              const int in_word_nxt = w * 32 + nth_set_bit(mw, (j + 1 < pcw) ? (j + 1) : 0);
              const int next_local  = (j + 1 < pcw) ? in_word_nxt : nxt_after_w;
              if (run_idx < run_end)
              {
                const int head_pos       = warp_tile_offset + local_pos;
                const int global_run_idx = global_run_base + run_idx;
#  if RLE_STCS_OUT
                __stcs(d_unique + global_run_idx, RLE_KEY_AT(head_pos));
#  else
                d_unique[global_run_idx] = RLE_KEY_AT(head_pos);
#  endif
                if (run_idx + 1 < warp_tile_run_count)
                {
#  if RLE_STCS_OUT
                  __stcs(d_counts + global_run_idx, next_local - local_pos);
#  else
                  d_counts[global_run_idx] = next_local - local_pos;
#  endif
                }
              }
            }
            return;
          }
#endif
#if RLE_CNT_SHFL
          // one pos load per run: run r+1's head sits in lane l+1's register (shfl_down); the lane-31
          // boundary is fed by software-pipelining the NEXT iteration's (guarded) load. The loop is
          // warp-uniform (niter) so the shuffles never see exited lanes.
          const int niter = (run_end - run_begin + 31) / 32;
          int head_pos    = (run_begin + lane_id < run_end)
                            ? (int) run_positions[warp_tile_offset + swizzle_xor_stride32(run_begin + lane_id)]
                            : 0;
#  if RLE_DRAIN_UNROLL4
#    pragma unroll 4
#  elif RLE_DRAIN_UNROLL2
#    pragma unroll 2
#  endif
          for (int it = 0; it < niter; ++it)
          {
            const int run_idx     = run_begin + it * 32 + lane_id;
            const int preload_idx = run_idx + 32;
            const int preload_pos =
              (preload_idx < run_end) ? (int) run_positions[warp_tile_offset + swizzle_xor_stride32(preload_idx)] : 0;
            const int nbr_pos      = __shfl_down_sync(kFullMask, head_pos, 1); // lane l+1's run = run_idx+1
            const int next0_pos    = __shfl_sync(kFullMask, preload_pos, 0); // next iteration, lane 0
            const int next_run_pos = (lane_id == 31) ? next0_pos : nbr_pos;
            if (run_idx < run_end)
            {
              const int global_run_idx = global_run_base + run_idx;
#  if RLE_STCS_OUT
              __stcs(d_unique + global_run_idx, RLE_KEY_AT(head_pos));
#  else
              d_unique[global_run_idx] = RLE_KEY_AT(head_pos);
#  endif
              if (run_idx + 1 < warp_tile_run_count)
              {
                // slice-boundary fallback (run_idx+1 == run_end < warp_tile_run_count) only exists
                // when a warp-tile is split across store warps; whole-warp-tile configs never take it
                const int cnt =
                  ((run_idx + 1 < run_end) ? next_run_pos
                                           : (int) run_positions[warp_tile_offset + swizzle_xor_stride32(run_idx + 1)])
                  - head_pos;
#  if RLE_STCS_OUT
                __stcs(d_counts + global_run_idx, cnt);
#  else
                d_counts[global_run_idx] = cnt;
#  endif
              }
            }
            head_pos = preload_pos;
          }
#else
#  if RLE_DRAIN_UNROLL4
#    pragma unroll 4
#  elif RLE_DRAIN_UNROLL2
#    pragma unroll 2
#  endif
          for (int run_idx = run_begin + lane_id; run_idx < run_end; run_idx += 32)
          {
            const int global_run_idx = global_run_base + run_idx;
            const int head_pos       = (int) run_positions[warp_tile_offset + swizzle_xor_stride32(run_idx)];
            // PROBE: WARP-LOCAL stride-1 gather address of identical cost (WRONG results); pos loads
            // unchanged. Warp-local matters: v32's tile-local fake collapsed all store warps onto the
            // same words and cost -4pt dense by itself. At seg1 this fake == the real pattern exactly
            // (every element is a head), so the dense delta doubles as the probe's sanity check.
            const int gather_pos = RLE_PROBE_GATHER_FREE ? (warp_tile_offset + run_idx) : head_pos;
#  if RLE_STCS_OUT
            __stcs(d_unique + global_run_idx, RLE_KEY_AT(gather_pos)); // streaming: outputs are never re-read
#  else
            d_unique[global_run_idx] = RLE_KEY_AT(gather_pos); // gather the run's key at its head position
#  endif
            if (run_idx + 1 < warp_tile_run_count)
            {
              // within-warp delta (next head - this head); the last run is fixed separately
              const int cnt = (int) run_positions[warp_tile_offset + swizzle_xor_stride32(run_idx + 1)] - head_pos;
#  if RLE_STCS_OUT
              __stcs(d_counts + global_run_idx, cnt);
#  else
              d_counts[global_run_idx] = cnt;
#  endif
            }
          }
#endif
        };
      if constexpr (kNumStoreWarps >= kNumCompWarps)
      {
        // if we have more store warps, each warptile is split between store warps
        constexpr int kStoreWarpsPerWarpTile = kNumStoreWarps / kNumCompWarps;
        const int warp_tile_id               = store_warp_idx / kStoreWarpsPerWarpTile;
        const int sub                        = store_warp_idx % kStoreWarpsPerWarpTile;
        const int warp_tile_run_count        = __shfl_sync(kFullMask, wt_count_ln, warp_tile_id);
        const int warp_tile_run_base         = __shfl_sync(kFullMask, wt_base_ln, warp_tile_id);
#if RLE_REGBUF
        if (warp_tile_run_count >= RLE_RB_MIN && warp_tile_run_count <= RLE_REGBUF)
        {
          const int run_begin = (int) ((long) warp_tile_run_count * sub / kStoreWarpsPerWarpTile);
          const int run_end   = (int) ((long) warp_tile_run_count * (sub + 1) / kStoreWarpsPerWarpTile);
          // wait for staged_wt (3/3) FIRST -- decode runs before the prefix exists
          while (!ptx::mbarrier_try_wait_parity(&staged_wt[slot_id][warp_tile_id], (unsigned) ((gen / kStages) & 1)))
          {
          }
          constexpr int kBufPerLane = (RLE_REGBUF + 31) / 32;
          int buf_key[kBufPerLane];
          int buf_cnt[kBufPerLane];
          const int warp_tile_offset = warp_tile_id * kWarpTileSize;
          const int niter            = (run_end - run_begin + 31) >> 5;
#  if RLE_FLAGWORD
          if (warp_tile_run_count < RLE_FW_THRESH)
          {
            // rank-select decode (same as the classic fw path, but into registers)
            const unsigned my_word = flag_ring[slot_id][warp_tile_id * 32 + lane_id];
            const int my_pc        = __popc(my_word);
            int incl               = my_pc;
#    pragma unroll
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
#    pragma unroll
            for (int o = 1; o < 32; o <<= 1)
            {
              const int c = __shfl_down_sync(kFullMask, nxt_min, o);
              nxt_min     = min(nxt_min, (lane_id + o < 32) ? c : 0x7fffffff);
            }
#    pragma unroll
            for (int it = 0; it < kBufPerLane; ++it)
            {
              if (it >= niter)
              {
                break;
              }
              const int run_idx = run_begin + it * 32 + lane_id;
              int w             = 0;
#    pragma unroll
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
              const int local_pos   = w * 32 + nth_set_bit(mw, (j < pcw) ? j : 0);
              const int in_word_nxt = w * 32 + nth_set_bit(mw, (j + 1 < pcw) ? (j + 1) : 0);
              buf_key[it]           = RLE_KEY_AT(warp_tile_offset + local_pos);
              buf_cnt[it]           = ((j + 1 < pcw) ? in_word_nxt : nxt_after_w) - local_pos;
            }
          }
          else
#  endif
          {
#  pragma unroll
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
              buf_key[it]        = RLE_KEY_AT(head_pos);
              buf_cnt[it]        = (act && run_idx + 1 < warp_tile_run_count)
                                   ? (int) run_positions[warp_tile_offset + swizzle_xor_stride32(run_idx + 1)] - head_pos
                                   : 0;
            }
          }
          if (lane_id == 0)
          {
            if constexpr (kPosStages < kStages)
            {
              ptx::mbarrier_arrive(&pos_free[gen % kPosStages]); // pos reads done -- BEFORE the prefix wait
            }
            ptx::mbarrier_arrive(&empty[slot_id]); // key reads done too: outputs live in registers
          }
          const int global_run_base = wait_prefixed_and_read() + warp_tile_run_base;
#  pragma unroll
          for (int it = 0; it < kBufPerLane; ++it)
          {
            if (it >= niter)
            {
              break;
            }
            const int run_idx = run_begin + it * 32 + lane_id;
            if (run_idx < run_end)
            {
              const int global_run_idx = global_run_base + run_idx;
              d_unique[global_run_idx] = buf_key[it];
              if (run_idx + 1 < warp_tile_run_count)
              {
                d_counts[global_run_idx] = buf_cnt[it];
              }
            }
          }
          continue; // pos_free/empty already arrived for this gen
        }
#endif
        // classic path -- champion order: prefixed wait, then staged_wt, then drain
        const int curr_prefix_run_count = wait_prefixed_and_read();
        // wait for staged_wt (3/3): only THIS warp-tile's positions -- not the other 7 compute warps
        while (!ptx::mbarrier_try_wait_parity(&staged_wt[slot_id][warp_tile_id], (unsigned) ((gen / kStages) & 1)))
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
        const int curr_prefix_run_count = wait_prefixed_and_read();
        // fewer store warps than compute regions: each store warp walks whole warptiles
        for (int warp_tile_id = store_warp_idx; warp_tile_id < kNumCompWarps; warp_tile_id += kNumStoreWarps)
        {
          const int warp_tile_run_count = __shfl_sync(kFullMask, wt_count_ln, warp_tile_id);
          const int warp_tile_run_base  = __shfl_sync(kFullMask, wt_base_ln, warp_tile_id);
          while (!ptx::mbarrier_try_wait_parity(&staged_wt[slot_id][warp_tile_id], (unsigned) ((gen / kStages) & 1)))
          {
          }
          drain(curr_prefix_run_count, warp_tile_id, warp_tile_run_base, warp_tile_run_count, 0, warp_tile_run_count);
        }
      }
      if (lane_id == 0)
      {
        if constexpr (kPosStages < kStages)
        {
          ptx::mbarrier_arrive(&pos_free[gen % kPosStages]); // this warp's drain no longer reads the pos slot
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
    for (int gen = 0;; ++gen)
    {
      const int slot_id = gen % kStages;
      while (!ptx::mbarrier_try_wait_parity(&computed[slot_id], (unsigned) ((gen / kStages) & 1)))
      {
      }
      const int tile_id = tile_seq[slot_id];
      if (tile_id >= num_tiles)
      {
        if (lane_id == 0)
        {
          ptx::mbarrier_arrive(&empty[slot_id]);
        }
        break;
      }
      const int tile_len = min(kTileSize, num_items - tile_id * kTileSize);
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
      while (!ptx::mbarrier_try_wait_parity(&prefixed[slot_id], (unsigned) ((gen / kStages) & 1)))
      {
      }
#if RLE_BK_EARLY_ARRIVE
      // read ALL slot metadata into registers BEFORE the prefix wait (it is computed-gated), so the
      // moment the prefix lands we can release the slot -- the global writes never ride the recycle
      // chain. (the reads here duplicate the pre-wait scan's sources; all lane-local.)
      const unsigned later_wts = wt_runs_mask >> (lane_id + 1);
      int reg_next_first = 0, reg_my_last = 0, reg_first_head = -1;
      if (lane_id < kNumCompWarps && wt_count_ln > 0)
      {
        if (later_wts)
        {
          reg_next_first = warp_first_heads[slot_id][lane_id + 1 + __ffs(later_wts) - 1];
        }
        reg_my_last = warp_last_heads[slot_id][lane_id];
      }
      if (lane_id == 0 && wt_runs_mask != 0)
      {
        reg_first_head = warp_first_heads[slot_id][__ffs(wt_runs_mask) - 1];
      }
#endif
      const u64 packed_prefix           = prefix_packed[slot_id][RLE_REGBUF ? ((gen / kStages) & 1) : 0];
      const int curr_prefix_run_count   = (int) (packed_prefix & 0xffffffffull);
      const int curr_prefix_open_length = (int) (packed_prefix >> 32);
#if RLE_BK_EARLY_ARRIVE
      if (lane_id == 0)
      {
        ptx::mbarrier_arrive(&empty[slot_id]); // slot may recycle NOW; writes below use registers only
      }
      if (lane_id < kNumCompWarps && wt_count_ln > 0)
      {
        const int last_run_global_idx = curr_prefix_run_count + wt_base_ln + wt_count_ln - 1;
        if (later_wts)
        {
          d_counts[last_run_global_idx] = reg_next_first - reg_my_last;
        }
        else if (is_last)
        {
          d_counts[last_run_global_idx] = tile_len - reg_my_last;
        }
      }
      if (lane_id == 0)
      {
        const bool any_head = (wt_runs_mask != 0);
        if (any_head && curr_prefix_run_count > 0)
        {
          d_counts[curr_prefix_run_count - 1] = curr_prefix_open_length + reg_first_head;
        }
        if (is_last && !any_head && curr_prefix_run_count > 0)
        {
          d_counts[curr_prefix_run_count - 1] = curr_prefix_open_length + tile_len;
        }
        if (is_last)
        {
          *d_num_runs = curr_prefix_run_count + tile_total_runs;
        }
      }
#else
      // per-warp-tile boundary: a warp-tile's last run is closed by the next nonempty warp-tile's
      // first head. lane L handles warp-tile L.
      if (lane_id < kNumCompWarps && wt_count_ln > 0)
      {
        const unsigned later_wts      = wt_runs_mask >> (lane_id + 1); // nonempty warp-tiles after L
        const int last_run_global_idx = curr_prefix_run_count + wt_base_ln + wt_count_ln - 1;
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
          *d_num_runs = curr_prefix_run_count + tile_total_runs;
        }
        ptx::mbarrier_arrive(&empty[slot_id]); // bookkeeping done, slot may recycle
      }
#endif
    }
  }
}

inline void persistent_rle_launch(
  const int* d_keys,
  int* d_unique,
  int* d_counts,
  int* d_num_runs,
  u64* tile_state,
  [[maybe_unused]] int* global_tile_counter,
  int num_items,
  int num_tiles,
  cudaStream_t stream)
{
  // raise the dynamic-smem cap once (idempotent; kept off the per-launch path)
  static const bool smem_cap_set = [] {
    cudaFuncSetAttribute(persistent_rle, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) kDynSmem);
    return true;
  }();
  (void) smem_cap_set;

  // no per-launch clear of tile_state: states are generation-tagged (see publish_state). The array
  // only needs to be zeroed ONCE at allocation; each launch bumps the gen and stale words never match.
#if RLE_L2_POLICY
  // pin the tile-state array as L2-persisting: it is the hottest latency-critical data in the
  // kernel and otherwise fights ~3GB/launch of streaming key/output traffic for residency
  static bool l2_policy_set = [] {
#  ifndef RLE_L2_CARVEOUT
#    define RLE_L2_CARVEOUT (1 << 20) // 256KB..4MB all measured identical; 1MB default
#  endif
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, RLE_L2_CARVEOUT);
    return true;
  }();
  (void) l2_policy_set;
  cudaStreamAttrValue l2attr          = {};
  l2attr.accessPolicyWindow.base_ptr  = tile_state;
  l2attr.accessPolicyWindow.num_bytes = sizeof(u64) * (size_t) num_tiles;
  l2attr.accessPolicyWindow.hitRatio  = 1.0f;
  l2attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
  l2attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
  cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &l2attr);
#endif
  static unsigned launch_gen = 0;
  ++launch_gen;
  if (launch_gen == 0) // paranoia: on u32 wrap, skip gen 0 (matches the zeroed-at-alloc words)
  {
    ++launch_gen;
  }
#if !USE_CLC
  cudaMemsetAsync(global_tile_counter, 0, sizeof(int), stream); // reset the work-steal counter
  int numSM = 0;
  cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, 0);
  const int blocks = std::min(2 * numSM, num_tiles); // persistent work-steal grid
#else
  const int blocks = num_tiles;
#endif

  persistent_rle<<<blocks, kNumThreads, kDynSmem, stream>>>(
    d_keys,
    d_unique,
    d_counts,
    d_num_runs,
    tile_state,
#if !USE_CLC
    global_tile_counter,
#endif
    launch_gen,
    num_items,
    num_tiles);
}
