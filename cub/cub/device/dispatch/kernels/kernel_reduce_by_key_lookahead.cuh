// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/warpspeed/squad/squad.cuh>
#include <cub/device/dispatch/tuning/tuning_rle_encode.cuh>
#include <cub/util_arch.cuh>
#include <cub/util_macro.cuh>
#include <cub/warp/warp_scan.cuh>

#include <cuda/atomic>
#include <cuda/ptx>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#if defined(IKET_BUILD_ENABLED)
#  include <iket/iket_device_apis.cuh>
#  define RBK_IK_RANGE(n)  CREATE_IKET_START_END_RANGE_EX(n, uint32_t, iket::IketEventPairMode::kNameOnly)
#  define RBK_IK_BEG(n, v) IKET_RANGE_START_EX(n, (uint32_t) (v))
#  define RBK_IK_END(n, v) IKET_RANGE_END_EX(n, (uint32_t) (v))
#else
#  define RBK_IK_RANGE(n)
#  define RBK_IK_BEG(n, v)
#  define RBK_IK_END(n, v)
#endif

RBK_IK_RANGE(LoadWaitEmpty);
RBK_IK_RANGE(LoadIssue);
RBK_IK_RANGE(CompWaitFull);
RBK_IK_RANGE(CompFlag);
RBK_IK_RANGE(CompPublish);
RBK_IK_RANGE(KWaitComputed);
RBK_IK_RANGE(KWaitPrefixed);
RBK_IK_RANGE(KDrain);
RBK_IK_RANGE(VSetup);
RBK_IK_RANGE(VEmit);
RBK_IK_RANGE(VBoundary);

CUB_NAMESPACE_BEGIN

namespace detail::reduce_by_key::lookahead
{
// the kernel (and everything it needs) only exists from PTX ISA 9.2 (CUDA 13.2): the load warp requires the
// cp.async.bulk .ignore_oob qualifier. Below that, the dispatch layer compiles the lookahead path out entirely.
#if __cccl_ptx_isa >= 920
namespace ptx = ::cuda::ptx;

_CCCL_HOST_DEVICE_API constexpr int num_total_threads(const RleLookaheadPolicy& policy)
{
  // keys-only pass: load + compute + poll + key stores (values run in their own kernel)
  const int num_total_warps = 1 /*load*/ + policy.compute_warps + 1 /*poll*/ + policy.compute_warps /*key store*/;
  return num_total_warps * 32;
}

// This is important for position staging on dense cases (16 way bank conflicts).
_CCCL_DEVICE_API _CCCL_FORCEINLINE int swizzle_xor_stride32(int x)
{
  return x ^ (x >> 5);
}

constexpr unsigned full_mask = 0xffffffffu;

_CCCL_DEVICE_API _CCCL_FORCEINLINE void wait_parity(::cuda::std::uint64_t* bar, unsigned parity)
{
  while (!ptx::mbarrier_try_wait_parity(bar, parity))
  {
  }
}

// stages is a runtime value now (we pick the ring depths at launch). now a runtime divide is super expensive
// and costs ~3-6% BWUtil, so now we have to maintain a "cursor" that is far more cheaper.
struct RingCursorT
{
  int slot        = 0;
  unsigned parity = 0;

  _CCCL_DEVICE_API _CCCL_FORCEINLINE void advance(int stages)
  {
    if (++slot == stages)
    {
      slot = 0;
      parity ^= 1u;
    }
  }
};

// DISAGGREGATED TILE STATES (the reduce-by-key design): the COUNT state is published by the
// compute warps at flag-fold ("publish as soon as possible" -- no value dependency), and paces
// the poll/prefix/store chain. The VALUE state is published by the value warps whenever their
// reduction lands, and is read LAZILY by the boundary closer only on tiles that actually close
// an entering run. Nothing ever folds the value chain eagerly.
constexpr unsigned tile_published = 1u;

struct CountStateT
{
  ::cuda::std::uint64_t dword;

  _CCCL_DEVICE_API _CCCL_FORCEINLINE unsigned published_tag() const
  {
    return (unsigned) (dword >> 32);
  }

  _CCCL_DEVICE_API _CCCL_FORCEINLINE int run_count() const
  {
    return (int) (dword & 0xffffffffu);
  }

  static _CCCL_DEVICE_API _CCCL_FORCEINLINE CountStateT pack(int run_count)
  {
    return {((::cuda::std::uint64_t) tile_published << 32) | (::cuda::std::uint64_t) (unsigned) run_count};
  }
};

// per-tile VALUE RECORD, written with plain stores by the value warps and read ONLY by the
// cleanup kernel after the main kernel completes (the kernel boundary is the synchronization --
// no tags, no atomics, no cross-tile value reads during the run)
template <class ValueT, class OffT>
struct TileValueRecordT
{
  ValueT open_agg; // sum after the tile's last head (whole-tile sum when head-free)
  ValueT lead_agg; // sum before the tile's first head (whole-tile sum when head-free)
  OffT boundary_dst; // output index of the entering-run close this tile owes, or -1
  int boundary_from; // the last headed tile before this one (the close's window start)
};

template <class StateT>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void publish_state(StateT* state_arr, int tile_idx, StateT st)
{
  ::cuda::atomic_ref<::cuda::std::uint64_t, ::cuda::thread_scope_device> a(state_arr[tile_idx].dword);
  a.store(st.dword, ::cuda::memory_order_relaxed);
}

// return the state (even if not yet published for this launch, caller checks it); no spinning here
template <class StateT>
_CCCL_DEVICE_API _CCCL_FORCEINLINE StateT load_state(StateT* state_arr, int tile_idx)
{
  ::cuda::atomic_ref<::cuda::std::uint64_t, ::cuda::thread_scope_device> a(state_arr[tile_idx].dword);
  return {a.load(::cuda::memory_order_relaxed)};
}

// CRITICAL: from choose_signed_offset, it is guaranteed that OffT covers the whole index space.
// The second field carries the id of the last tile (before this one) known to contain a run
// head: the boundary closer uses it to read the lazy value-state window without any poll help.
template <class OffT, bool = (sizeof(OffT) > 4)>
struct PrefixT;

template <class OffT>
struct PrefixT<OffT, false>
{
  ::cuda::std::uint64_t dword;

  static _CCCL_DEVICE_API _CCCL_FORCEINLINE PrefixT pack(OffT run_count, int last_tile_with_runs)
  {
    return {((::cuda::std::uint64_t) (unsigned) last_tile_with_runs << 32) | (unsigned) run_count};
  }

  _CCCL_DEVICE_API _CCCL_FORCEINLINE OffT run_count() const
  {
    return (OffT) (unsigned) (dword & 0xffffffffull);
  }

  _CCCL_DEVICE_API _CCCL_FORCEINLINE int last_tile_with_runs() const
  {
    return (int) (unsigned) (dword >> 32);
  }
};

template <class OffT>
struct alignas(16) PrefixT<OffT, true>
{
  ::cuda::std::uint64_t packed_run_count;
  ::cuda::std::uint64_t packed_last_tile_with_runs;

  static _CCCL_DEVICE_API _CCCL_FORCEINLINE PrefixT pack(OffT run_count, int last_tile_with_runs)
  {
    return {(::cuda::std::uint64_t) run_count, (::cuda::std::uint64_t) (unsigned) last_tile_with_runs};
  }

  _CCCL_DEVICE_API _CCCL_FORCEINLINE OffT run_count() const
  {
    return (OffT) packed_run_count;
  }

  _CCCL_DEVICE_API _CCCL_FORCEINLINE int last_tile_with_runs() const
  {
    return (int) (unsigned) packed_last_tile_with_runs;
  }
};

// position of the n-th set bit of flag_mask, requires popc(flag_mask) > rank. Implementation is binary search.
// __fns(flag_mask, 0, rank+1) computes the same thing but has NO hardware op on sm_100a and is slower
// TODO (Nan): as per discussion with Federico, this could be in libcudacxx
_CCCL_DEVICE_API _CCCL_FORCEINLINE int nth_set_bit(unsigned flag_mask, int rank)
{
  // each step: if the wanted bit is not among the low half's set bits, skip that half entirely
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

struct WarpTileRunScanT
{
  int lane_run_count;
  int lane_runs_before;
};

// we need this because STORE and BOOKKEEPER both recalculate from slot_warp_run_counts
template <int compute_warps>
_CCCL_DEVICE_API _CCCL_FORCEINLINE WarpTileRunScanT
scan_warp_tile_run_counts(const int* slot_warp_run_counts, int lane_id)
{
  const int lane_run_count = (lane_id < compute_warps) ? slot_warp_run_counts[lane_id] : 0;
  typename WarpScan<int>::TempStorage warp_scan_storage;
  int lane_scan;
  WarpScan<int>(warp_scan_storage).InclusiveSum(lane_run_count, lane_scan);
  return {lane_run_count, lane_scan - lane_run_count};
}

template <int tile_size, int slot_pad, class KeyT>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void load_tile_keys(
  KeyT* slot,
  const KeyT* d_keys,
  int tile_id,
  int tile_len,
  bool first_tile,
  bool last_tile,
  unsigned base_skip,
  ::cuda::std::uint64_t* full_bar,
  int lane_id,
  bool keys_staged)
{
  if (lane_id == 0)
  {
    if (!keys_staged)
    {
      // vvv regressed case: no TMA; full now only mrans  "tile_id_buf[slot] is valid" vvv
      ptx::mbarrier_arrive(full_bar);
      // ^^^ regressed case ^^^
    }
    else
    {
      // if it is not first tile, we overcopy 16B to the left to get last key from last tile
      const unsigned nbytes     = (unsigned) (((size_t) tile_len + (first_tile ? 0 : slot_pad)) * sizeof(KeyT));
      const unsigned span_bytes = (nbytes + base_skip + 15u) & ~15u;
      ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, full_bar, span_bytes);
      ptx::cp_async_bulk_ignore_oob(
        ptx::space_shared,
        ptx::space_global,
        slot + (first_tile ? slot_pad : 0),
        (const KeyT*) ((const char*) (d_keys + (size_t) tile_id * tile_size - (first_tile ? 0 : slot_pad)) - base_skip),
        span_bytes,
        first_tile ? base_skip : 0u,
        last_tile ? (span_bytes - base_skip - nbytes) : 0u,
        full_bar);
    }
  }
  __syncwarp();
}

_CCCL_DEVICE_API _CCCL_FORCEINLINE int
clc_next_tile_id(uint4& clc_resp, ::cuda::std::uint64_t& clc_bar, int pipeline_gen, int num_tiles, int lane_id)
{
  int nxt = num_tiles; // if no more work was cancellable
  if (lane_id == 0)
  {
    wait_parity(&clc_bar, (unsigned) (pipeline_gen & 1));
    // try_cancel wrote clc_resp via the async proxy
    ptx::fence_proxy_async(ptx::space_shared);
    const uint4 resp_snapshot = clc_resp;
    ptx::fence_proxy_async(ptx::space_shared);
    const bool canceled = ptx::clusterlaunchcontrol_query_cancel_is_canceled(resp_snapshot);
    if (canceled)
    {
      nxt = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x<int>(resp_snapshot);
      ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &clc_bar, 16);
      ptx::clusterlaunchcontrol_try_cancel(&clc_resp, &clc_bar);
    }
  }
  return __shfl_sync(full_mask, nxt, 0);
}

// calculate head_flags: each iter is 32 consecutive elements (lane L owns loc = warp_tile_offset + iter*32 + L)
// head = (key != predecessor)
template <int items_per_thread, bool clamp_tail, class KeyT>
_CCCL_DEVICE_API _CCCL_FORCEINLINE unsigned
compute_head_flags(const KeyT* key_buf, int warp_tile_offset, int tile_len, int tile_id, int lane_id, int skip_elems)
{
  static_assert(items_per_thread <= 32, "one lane per iter requires items_per_thread<=32");
  unsigned my_flags = 0;
  // ONE read per element: the predecessor comes from a lane shuffle (row-internal) or the
  // previous row's last lane; only the first row's lane 0 reads its predecessor from memory
  // (receipted +10% at the long regimes in the standalone campaign)
  int first_pred_idx = warp_tile_offset + skip_elems - 1; // slot_pad makes this legal when staged
  if constexpr (clamp_tail)
  {
    first_pred_idx =
      (tile_id == 0) ? max(min(warp_tile_offset, tile_len - 1) - 1, 0) : min(warp_tile_offset, tile_len - 1) - 1;
    first_pred_idx = max(first_pred_idx, 0);
  }
  KeyT row_carry = key_buf[first_pred_idx];
#  pragma unroll
  for (int iter = 0; iter < items_per_thread; ++iter)
  {
    const int loc = warp_tile_offset + iter * 32 + lane_id;
    int key_idx   = loc + skip_elems;
    if constexpr (clamp_tail)
    {
      // vvv regressed case: plain global loads have no ignore_oob, so clamp the tail reads into the input.
      // the clamped values are garbage, but (loc < tile_len) below already zeroes those heads vvv
      key_idx = min(key_idx, tile_len - 1);
      // ^^^ regressed case ^^^
    }
    const KeyT key = key_buf[key_idx];
    KeyT pred      = __shfl_up_sync(full_mask, key, 1);
    if (lane_id == 0)
    {
      pred = row_carry;
    }
    const int is_global_first = (tile_id == 0 && loc == 0);
    const int head            = (loc < tile_len) ? (is_global_first ? 1 : !(key == pred)) : 0;
    const unsigned flags      = __ballot_sync(full_mask, head);
    if (lane_id == iter)
    {
      my_flags = flags;
    }
    row_carry = __shfl_sync(full_mask, key, 31);
  }
  return my_flags;
}

template <int compute_warps>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void
reduce_and_publish_count(CountStateT* count_states, int tile_id, const int* slot_warp_run_counts, int lane_id)
{
  // compute_warps<=32 so one lane/warp fits (in practice we will never have anything close to 32)
  static_assert(compute_warps <= 32, "compute_warps must be less than 32!");
  const bool active        = lane_id < compute_warps;
  const int warp_run_count = active ? slot_warp_run_counts[lane_id] : 0;
  const int run_count      = __reduce_add_sync(full_mask, warp_run_count);
  if (lane_id == 0)
  {
    // CRITICAL: publish as soon as possible, this is why we calculate head_flags first -- the
    // count chain never waits on a single value read (the disaggregation design)
    publish_state(count_states, tile_id, CountStateT::pack(run_count));
  }
}

template <int items_per_thread>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void
stage_head_positions(unsigned my_flags, short* pos_dst, int warp_tile_offset, int lane_id)
{
  // we store run R at warp_tile_offset + (R ^ (R>>5)) to avoid bank conflicts for dense cases
  // (CRITICAL for MaxSeg=1,2,4)
  int head_scan = __popc(my_flags); // start: this word's head count
  typename WarpScan<int>::TempStorage warp_scan_storage;
  WarpScan<int>(warp_scan_storage).InclusiveSum(head_scan, head_scan);
  // head_scan is a running sum of run_count, so each lane know each chunk's base
  const int runs_before_word = head_scan - __popc(my_flags);
  if (lane_id < items_per_thread)
  {
    const int word_pos     = warp_tile_offset + lane_id * 32; // element position of bit 0 of this word
    unsigned pending_heads = my_flags; // this word's head mask; we need to "peel" it headbit by headbit
    int run_index          = runs_before_word; // run-order slot for this word's next head
    while (pending_heads)
    {
      const int head_offset = __ffs(pending_heads) - 1; // offset (0..31) of the next head within the word
      pos_dst[warp_tile_offset + swizzle_xor_stride32(run_index)] = (short) (word_pos + head_offset);
      ++run_index;
      pending_heads &= (pending_heads - 1); // clear the lowest set bit
    }
  }
}

// pre-seeded plain span walk: nvcc pipelines the loads across the FADD chain on its own
// (the blocked-prefetch variant measured SLOWER: round-6 receipt)
template <class ValueT>
_CCCL_DEVICE_API _CCCL_FORCEINLINE ValueT span_sum_prefetched(const ValueT* tile_vals, int pos, const int end)
{
  ValueT agg = tile_vals[pos];
  for (++pos; pos < end; ++pos)
  {
    agg += tile_vals[pos];
  }
  return agg;
}

// values-only streaming reduce over one warp tile, boundaries from REGISTER flag words (the
// value warps snapshot the flags and run fully detached from the rings). Emits every
// within-warp-tile-closed aggregate; the trailing open sum comes back as the carry. The scan
// loop is the FROZEN form -- lead/tail ride separate passes, never inside it.
template <int items_per_thread, class ValueT, class OffT>
_CCCL_DEVICE_API __noinline__ void stream_values_from_flags(
  ValueT* d_aggregates,
  const ValueT* tile_vals,
  unsigned my_word,
  OffT global_runs_before_warp_tile,
  int warp_tile_offset,
  int tile_len,
  int lane_id,
  ValueT& tail_out)
{
  const int my_popc = __popc(my_word);
  typename WarpScan<int>::TempStorage warp_scan_storage;
  int word_scan;
  WarpScan<int>(warp_scan_storage).InclusiveSum(my_popc, word_scan);
  const int my_runs_before_word = word_scan - my_popc;
  const unsigned upto_l         = (lane_id == 31) ? 0xffffffffu : ((2u << lane_id) - 1); // bits [0, lane]
  // CHUNKED preload: 8 rows in flight instead of the whole warp tile -- the full-tile buffer put
  // the kernel at 32+ live registers/lane and SPILLED TO LOCAL under high occupancy (B200
  // receipt: VEmit 16us/warp-tile of local-memory thrash). The scan carry crosses chunks freely.
  constexpr int chunk_rows = (items_per_thread < 8) ? items_per_thread : 8;
  ValueT carry{};
  for (int chunk = 0; chunk < items_per_thread; chunk += chunk_rows)
  {
    ValueT row_vals[chunk_rows];
#  pragma unroll
    for (int cr = 0; cr < chunk_rows; ++cr)
    {
      const int iter = chunk + cr;
      const int loc  = warp_tile_offset + iter * 32 + lane_id;
      row_vals[cr]   = (iter < items_per_thread && loc < tile_len) ? tile_vals[loc] : ValueT{};
    }
#  pragma unroll
    for (int cr = 0; cr < chunk_rows; ++cr)
    {
      const int iter = chunk + cr;
      if (iter >= items_per_thread)
      {
        break;
      }
      const unsigned w = __shfl_sync(full_mask, my_word, iter);
      ValueT incl      = row_vals[cr];
#  pragma unroll
      for (int off = 1; off < 32; off <<= 1)
      {
        const unsigned upto_prev = (lane_id >= off) ? ((2u << (lane_id - off)) - 1) : 0u;
        const ValueT from_left   = __shfl_up_sync(full_mask, incl, off);
        if (lane_id >= off && ((w & upto_l & ~upto_prev) == 0u))
        {
          incl += from_left;
        }
      }
      if ((w & upto_l) == 0u)
      {
        incl += carry;
      }
      const unsigned next_word = __shfl_sync(full_mask, my_word, (iter + 1) & 31);
      const bool is_end        = (lane_id < 31) ? (((w >> (lane_id + 1)) & 1u) != 0u)
                                                : ((iter + 1 < items_per_thread) && ((next_word & 1u) != 0u));
      const int run_idx        = __shfl_sync(full_mask, my_runs_before_word, iter) + __popc(w & upto_l) - 1;
      if (is_end && run_idx >= 0)
      {
        d_aggregates[global_runs_before_warp_tile + run_idx] = incl;
      }
      carry = __shfl_sync(full_mask, incl, 31);
    }
  }
  tail_out = carry; // sum since the warp tile's last head (whole tile when head-free)
}

// warp-parallel strided sum over [begin, end) of the tile's values (lead/tail segments and the
// long-span cooperative walks; every lane participates)
template <class ValueT>
_CCCL_DEVICE_API _CCCL_FORCEINLINE ValueT warp_span_sum(const ValueT* tile_vals, int begin, int end, int lane_id)
{
  ValueT acc{};
  for (int pos = begin + lane_id; pos < end; pos += 32)
  {
    acc += tile_vals[pos];
  }
#  pragma unroll
  for (int offset = 16; offset; offset >>= 1)
  {
    acc += __shfl_xor_sync(full_mask, acc, offset);
  }
  return acc;
}

struct RunSpanT
{
  int head_pos_in_warp_tile;
  int next_head_pos;
};

// the compute warp may deem this warp tile too sparse to be worth the position-staging, and in that case it will write
// only the 32 head-flag words. Then, it is up to the store warps to "decode" the positions from the headflags.
// one warp tile is 32 chunks x 32 elements, so lane i owns word i.
// This buys 2.5% BWUtil in the MaxSegSize{2^4, 2^6, 2^8}
struct HeadFlagDecodeT
{
  unsigned lane_head_flag_word;
  int lane_runs_before_word;
  int lane_first_head_from_word;

  _CCCL_DEVICE_API _CCCL_FORCEINLINE HeadFlagDecodeT(const unsigned* slot_head_flags, int warp_tile_id, int lane_id)
      : HeadFlagDecodeT(slot_head_flags[warp_tile_id * 32 + lane_id], lane_id)
  {}

  // register-word form: the value warps snapshot flag words and decode with no smem dependency
  _CCCL_DEVICE_API _CCCL_FORCEINLINE HeadFlagDecodeT(unsigned lane_word, int lane_id)
  {
    lane_head_flag_word           = lane_word;
    const int lane_word_run_count = __popc(lane_head_flag_word);
    typename WarpScan<int>::TempStorage warp_scan_storage;
    int lane_word_run_count_scan;
    WarpScan<int>(warp_scan_storage).InclusiveSum(lane_word_run_count, lane_word_run_count_scan);
    // lane i: # of runs starting in head_flag words [0, i), i.e. in elements [0, i*32)
    lane_runs_before_word = lane_word_run_count_scan - lane_word_run_count;
    // lane i -> first head position in head flag words [i, 32)
    // if our own run_count is >0, the head is here!
    // empty should be +infinity, since we use min
    lane_first_head_from_word = lane_word_run_count ? (lane_id * 32 + __ffs(lane_head_flag_word) - 1) : 0x7fffffff;
    // if not, we loop to find the next head in flag word [i, 32). this is just a fold with min
#  pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1)
    {
      const int shuffled_first_head = __shfl_down_sync(full_mask, lane_first_head_from_word, offset);
      lane_first_head_from_word =
        min(lane_first_head_from_word, (lane_id + offset < 32) ? shuffled_first_head : 0x7fffffff);
    }
    // now, lane i holds the next head in [i, 32). we precalculate this in parallel
  }

  _CCCL_DEVICE_API _CCCL_FORCEINLINE RunSpanT decode_run(int run_idx) const
  {
    // first question: which head_flag word contains my run's (run_idx) head?
    // lane_runs_before_word's row i = number of heads in words [0, i)
    // the word containing run_dex is then the largest i with runs_before(i) that is <= j
    // we do binary search over the distributed lane_runs_before_word table held across the warp
    int flag_word_idx = 0;
#  pragma unroll
    for (int step = 16; step; step >>= 1)
    {
      // propose candidate
      const int candidate_word_idx = flag_word_idx + step;
      // read the i'th row
      const int candidate_runs_before = __shfl_sync(full_mask, lane_runs_before_word, candidate_word_idx & 31);
      if (candidate_word_idx < 32 && candidate_runs_before <= run_idx)
      {
        flag_word_idx = candidate_word_idx;
      }
    }
    // the lane now knows the index of the word containing its head
    // we need to convert it to the element position
    // where is my head in the word?
    const int run_rank_in_word = run_idx - __shfl_sync(full_mask, lane_runs_before_word, flag_word_idx);
    // get the actual word
    const unsigned flag_word = __shfl_sync(full_mask, lane_head_flag_word, flag_word_idx);
    // where's the first head in any word after mine?
    const int first_head_after_word = __shfl_sync(full_mask, lane_first_head_from_word, (flag_word_idx + 1) & 31);
    // how many heads my word has?
    const int flag_word_run_count = __popc(flag_word);
    // position of my head inside the word
    const int head_bit_in_word =
      nth_set_bit(flag_word, (run_rank_in_word < flag_word_run_count) ? run_rank_in_word : 0);
    const int head_pos_in_warp_tile = flag_word_idx * 32 + head_bit_in_word;
    // where does my run end? try find the position of next head in word
    const int next_head_in_word = flag_word_idx * 32 + __ffs(flag_word & (~1u << head_bit_in_word)) - 1;
    // does my word contain a head after mine? if not, next_head_in_word is garbage, and we use first_head_after_word
    const int next_head_pos = (run_rank_in_word + 1 < flag_word_run_count) ? next_head_in_word : first_head_after_word;
    // NOTE: for the last run head in warp tile, next_head_pos is garbage
    return {head_pos_in_warp_tile, next_head_pos};
  }
};

template <int window_size_cap, class PolicySelector, class OffT>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void poll_fold_windows(
  CountStateT* count_states,
  int tile_id,
  int& last_seen_tile_id,
  OffT& last_seen_prefix_run_count,
  int& last_tile_with_runs,
  int lane_id,
  int& dense_mode)
{
  constexpr int poll_loads_per_lane = current_policy<PolicySelector>().lookahead.poll_loads_per_lane;
  static_assert(window_size_cap >= 1 && window_size_cap <= 32 * poll_loads_per_lane,
                "the fold window must be covered by the lanes");
  while (last_seen_tile_id < tile_id)
  {
    const int remain = tile_id - last_seen_tile_id;
    // # of tiles to fold this iteration
    const int window_size                         = remain < window_size_cap ? remain : window_size_cap;
    const int lane_first_tile_id                  = last_seen_tile_id + lane_id;
    const int lane_tile_count                     = (window_size - lane_id + 31) >> 5;
    CountStateT packed_words[poll_loads_per_lane] = {}; // must zero initialize
    bool ready;
    // first, all tile COUNT states in the window must be ready (values are never polled)
    do
    {
      ready = true;
#  pragma unroll
      for (int i = 0; i < poll_loads_per_lane; ++i)
      {
        // we only try if that state is not published
        if (i < lane_tile_count && packed_words[i].published_tag() != tile_published)
        {
          packed_words[i] = load_state(count_states, lane_first_tile_id + i * 32);
          if (packed_words[i].published_tag() != tile_published)
          {
            ready = false;
          }
        }
      }
    } while (__ballot_sync(full_mask, !ready) != 0u);
    int lane_run_count = 0, lane_last_tile_with_runs_in_window = -1;
    // now, we fold the window
#  pragma unroll
    for (int i = 0; i < poll_loads_per_lane; ++i)
    {
      if (i < lane_tile_count)
      {
        // aggregate run_count per lane, this is fine since run_count is commutative
        lane_run_count += packed_words[i].run_count();
        // nominate the highest tile id with runs (the closer's lazy value-window bound)
        lane_last_tile_with_runs_in_window =
          (packed_words[i].run_count() > 0) ? (i * 32 + lane_id) : lane_last_tile_with_runs_in_window;
      }
    }
    // vote for the highest tile id with runs
    const int last_tile_with_runs_in_window = __reduce_max_sync(full_mask, lane_last_tile_with_runs_in_window);
    if (last_tile_with_runs_in_window >= 0)
    {
      last_tile_with_runs = last_seen_tile_id + last_tile_with_runs_in_window;
    }
    const int window_run_count = __reduce_add_sync(full_mask, lane_run_count);
    // dense_mode is true if window_run_count > 128
    dense_mode                 = window_run_count > (window_size << 7);
    last_seen_prefix_run_count = last_seen_prefix_run_count + window_run_count;
    last_seen_tile_id += window_size;
  }
}

template <class PolicySelector, class OffT>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void poll_and_fold(
  CountStateT* count_states,
  int tile_id,
  int& last_seen_tile_id,
  OffT& last_seen_prefix_run_count,
  int& last_tile_with_runs,
  int lane_id,
  int& dense_mode,
  OffT& curr_prefix_run_count)
{
  // adaptive poll: we decide the window size based on the density of the runs. this buys ~5% BWUtil
  // the 2 window sizes: 96 and 160 = 32 * 5 are decided by the # of SM on blackwell
  if (dense_mode)
  // when it is dense, compute has a slower rate of publishing tile states. so we wait for a smaller window first and
  // fold it. as we fold the small window, more tiles in the next window are becoming ready, so we get some overlapping
  {
    poll_fold_windows<96, PolicySelector>(
      count_states, tile_id, last_seen_tile_id, last_seen_prefix_run_count, last_tile_with_runs, lane_id, dense_mode);
  }
  else
  // when it is sparse, compute has a high rate of publishing tile states. so we just poll the big window at once
  {
    poll_fold_windows<32 * current_policy<PolicySelector>().lookahead.poll_loads_per_lane, PolicySelector>(
      count_states, tile_id, last_seen_tile_id, last_seen_prefix_run_count, last_tile_with_runs, lane_id, dense_mode);
  }
  curr_prefix_run_count = last_seen_prefix_run_count;
}

// we aim for 1 block/SM since it is easier to manage resources: we do not need to worry about occupancy anymore
template <typename PolicySelector, class KeyT, class ValueT, class NumRunsT, class OffT>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void device_reduce_by_key_lookahead_body(
  const KeyT* __restrict__ d_keys,
  const ValueT* __restrict__ d_values,
  KeyT* __restrict__ d_unique,
  ValueT* __restrict__ d_aggregates,
  NumRunsT* __restrict__ d_num_runs,
  CountStateT* __restrict__ count_states,
  unsigned* __restrict__ d_flag_words, // [num_tiles][compute_warps*32]: the persisted run structure
  PrefixT<OffT>* __restrict__ d_tile_prefix, // [num_tiles]: packed (run prefix, last headed tile)
  OffT num_items,
  int num_tiles,
  int key_ring_stages,
  int pos_ring_stages,
  bool keys_staged)
{
  static constexpr RleLookaheadPolicy policy = current_policy<PolicySelector>().lookahead;
  static_assert(16 % sizeof(KeyT) == 0, "KeyT size must be a power of two <= 16");
  static_assert(alignof(KeyT) <= 16, "Alignment <= 16");
  static_assert(policy.items_per_thread >= 1 && policy.items_per_thread <= 32, "items_per_thread must be in [1, 32]");
  static_assert(policy.compute_warps >= 1 && policy.compute_warps <= 31, "compute_warps must be in [1, 31]");
  static_assert(policy.key_ring_stages >= 1, "at least one pipeline stage");
  static_assert(policy.pos_ring_stages >= 1 && 2 * policy.pos_ring_stages >= policy.key_ring_stages,
                "pos ring parity wait aliases unless 2*pos_ring_stages >= key_ring_stages");
  static_assert(policy.floor_pos_ring_stages() <= policy.pos_ring_stages
                  && 2 * policy.floor_pos_ring_stages() >= policy.floor_key_ring_stages(),
                "the unstaged floor configuration must satisfy the pos ring parity bound");
  static_assert(policy.floor_dyn_smem_bytes() + RleLookaheadPolicy::static_smem_budget
                  <= RleLookaheadPolicy::default_smem_per_block,
                "the unstaged floor configuration must launch within the default shared memory limit on every device");
  static_assert(policy.tile_size() <= 0xffff && policy.tile_size() <= 32768,
                "tile_size must fit the 16-bit state words and signed 16-bit staged positions");
  static_assert(num_total_threads(policy) <= 1024, "a CTA is capped at 1024 threads");
  static_assert(policy.buf_per_lane() * ((int) sizeof(KeyT) + 4) <= 64,
                "reg-buf rounds must fit the 64B/lane register budget");
  static_assert(::cuda::std::is_integral_v<OffT> && policy.tile_size() <= ::cuda::std::numeric_limits<OffT>::max(),
                "OffT must be an integer type wide enough for one tile");
  constexpr int items_per_thread = policy.items_per_thread;
  constexpr int compute_warps    = policy.compute_warps;
  // TWO-PASS DESIGN: this kernel is KEYS-ONLY (flags, counts, prefixes, unique keys); the value
  // pass is a separate pipeline-free kernel fed by the persisted flag words and tile prefixes
  constexpr int key_store_warps        = compute_warps;
  constexpr int max_key_ring_stages    = policy.key_ring_stages;
  constexpr int max_pos_ring_stages    = policy.pos_ring_stages;
  constexpr int flag_staging_threshold = policy.flag_staging_threshold;
  // in the regressed case: always stage positions, so the store warps never run the flag-decode drain vvv
  const int staging_threshold  = keys_staged ? flag_staging_threshold : 0;
  constexpr int warp_tile_size = policy.warp_tile_size();
  constexpr int tile_size      = policy.tile_size();
  constexpr int slot_pad       = policy.slot_pad((int) sizeof(KeyT));
  constexpr int slot_stride    = policy.slot_stride((int) sizeof(KeyT), (int) alignof(KeyT));
  using PrefixT                = reduce_by_key::lookahead::PrefixT<OffT>;
  // the dense band's fused-stream crossover (runs per warp tile); below it, per-run span walks
  // are output-proportional and cheaper
#  ifdef RBK_STREAM_DIV
  constexpr int stream_threshold = policy.warp_tile_size() / RBK_STREAM_DIV;
#  else
  // B200 receipt: /4 pulls the seg4 band into the stream and costs nothing anywhere else
  constexpr int stream_threshold = policy.warp_tile_size() / 4;
#  endif
  // [key_ring_stages][tile_size] input keys
  // [key_ring_stages][tile_size] int16 staged head positions
  extern __shared__ char smem_raw[];
  KeyT* const tile_buf = (KeyT*) smem_raw;
  __shared__ int tile_id_buf[max_key_ring_stages]; // which global tile each ring slot holds (LOAD gets it with
                                                   // try_cancel)
  __shared__ int warp_run_counts[max_key_ring_stages][compute_warps]; // per compute warp run counts
  __shared__ unsigned head_flag_buf[max_key_ring_stages][compute_warps * 32]; // staged head-flag words

  // for POLL to pass STORE the packed (run_count prefix, open aggregate) pair
  __shared__ PrefixT prefix_packed[max_key_ring_stages];

  // LOAD --full--> COMPUTE & POLL
  // COMPUTE(all warps) --computed--> COMPUTE w0, then cw0 calculates & publishes this tile's aggregate to the global
  // POLL --prefixed--> STORE
  // STORE --empty--> LOAD & POLL
  __shared__ ::cuda::std::uint64_t full[max_key_ring_stages];
  __shared__ ::cuda::std::uint64_t computed[max_key_ring_stages], prefixed[max_key_ring_stages],
    empty[max_key_ring_stages];

  // try_cancel writes a 16-byte response into clc_resp + completes clc_bar's tx.
  __shared__ __align__(16) uint4 clc_resp;
  __shared__ ::cuda::std::uint64_t clc_bar;
  static_assert(
    sizeof(tile_id_buf) + sizeof(warp_run_counts) + sizeof(head_flag_buf) + sizeof(prefix_packed) + sizeof(full)
        + sizeof(computed) + sizeof(prefixed) + sizeof(empty) + sizeof(clc_resp) + sizeof(clc_bar)
      <= RleLookaheadPolicy::static_smem_budget,
    "static shared memory exceeds the budget assumed by the floor launch guarantee");

  const int thr_id         = threadIdx.x;
  const int lane_id        = thr_id & 31;
  const int blk_id         = blockIdx.x;
  const unsigned base_skip = (alignof(KeyT) < 16) ? ((unsigned) (size_t) d_keys & 15u) : 0u;
  const int skip_elems     = (int) (base_skip / sizeof(KeyT));
  if (thr_id == 0)
  {
    for (int slot_id = 0; slot_id < max_key_ring_stages; ++slot_id)
    {
      ptx::mbarrier_init(&full[slot_id], 1);
      ptx::mbarrier_init(&computed[slot_id], compute_warps); // every compute warp arrives
      ptx::mbarrier_init(&prefixed[slot_id], 1);
      ptx::mbarrier_init(&empty[slot_id], key_store_warps);
    }

    ptx::mbarrier_init(&clc_bar, 1); // 1 arrival
  }
  // normal smem writes (e.g. mbarrier_init) go through the generic proxy
  // the TMA operations access shared memory through the async proxy. these are separate visibility domains,
  // so the init writes are not automatically visible to TMA.
  ptx::fence_proxy_async(ptx::space_shared);
  __syncthreads();

  constexpr warpspeed::SquadDesc squadLoad{0, 1};
  constexpr warpspeed::SquadDesc squadCompute{1, compute_warps + key_store_warps};
  constexpr warpspeed::SquadDesc squadPoll{2, 1};
  constexpr warpspeed::SquadDesc squads[] = {squadLoad, squadCompute, squadPoll};

  warpspeed::squadDispatch(
    warpspeed::getSpecialRegisters(), squads, [&](warpspeed::Squad squad) _CCCL_FORCEINLINE_LAMBDA {
      // if you are load
      if (squad == squadLoad)
      {
        // CLC tile assignment: gen0 tile = this CTA's launch id (blockIdx.x)
        int tile_id = blk_id;
        if (lane_id == 0)
        {
          // 16 is the try_cancel byte tx
          ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &clc_bar, 16);
          ptx::clusterlaunchcontrol_try_cancel(&clc_resp, &clc_bar);
        }
        RingCursorT key_ring;
        int sentinels_sent = 0;
        for (int pipeline_gen = 0;; ++pipeline_gen, key_ring.advance(key_ring_stages))
        {
          const int slot_id = key_ring.slot; // which slot is this?
          const bool ik_rec = (blk_id < 8) && pipeline_gen >= 5 && pipeline_gen < 15;
          if (pipeline_gen >= key_ring_stages)
          {
            if (ik_rec)
            {
              RBK_IK_BEG(LoadWaitEmpty, pipeline_gen);
            }
            // need to wait for slot to be free
            wait_parity(&empty[slot_id], key_ring.parity ^ 1u);
            if (ik_rec)
            {
              RBK_IK_END(LoadWaitEmpty, pipeline_gen);
            }
          }
          if (lane_id == 0)
          {
            tile_id_buf[slot_id] = tile_id;
          }
          if (tile_id >= num_tiles)
          {
            if (lane_id == 0)
            {
              ptx::mbarrier_arrive(&full[slot_id]);
            }
            __syncwarp();
            // the fused squad's gen-parity groups each terminate on their OWN sentinel gen: one
            // sentinel would leave the other group waiting full forever
            if (++sentinels_sent == 2)
            {
              break;
            }
            continue;
          }
          // over-fetch one 16B chunk to the left, so that we get last tiles last key
          // tile 0 has no predecessor and skips the over-fetch
          const bool first_tile = (tile_id == 0);
          const int tile_len    = (int) min((OffT) tile_size, num_items - (OffT) tile_id * tile_size);
          load_tile_keys<tile_size, slot_pad>(
            tile_buf + (size_t) slot_id * slot_stride,
            d_keys,
            tile_id,
            tile_len,
            first_tile,
            tile_id == num_tiles - 1,
            base_skip,
            &full[slot_id],
            lane_id,
            keys_staged);
          // consume the prefetched cancel, this is ok since it should be fast to get next cancelled id
          tile_id = clc_next_tile_id(clc_resp, clc_bar, pipeline_gen, num_tiles, lane_id);
        }
      }
      // if you are compute
      else if (squad == squadCompute)
      {
        // FUSED compute+store: no store squad. Gen-parity groups A/B alternate pipeline gens, so
        // one group's prefix wait + key drain overlaps the other group's flag pass; each warp
        // publishes FIRST, then drains its own warp tile (the cross-block count chain never waits
        // on a single store)
        const int fused_rank       = squad.warpRank();
        const int group            = fused_rank / compute_warps;
        const int compute_warp_id  = fused_rank % compute_warps;
        const int warp_tile_offset = compute_warp_id * warp_tile_size;
        RingCursorT key_ring;
        if (group != 0)
        {
          key_ring.advance(key_ring_stages);
        }
        for (int pipeline_gen = group;;
             pipeline_gen += 2, key_ring.advance(key_ring_stages), key_ring.advance(key_ring_stages))
        {
          const int slot_id = key_ring.slot;
          const bool ik_rec = (blk_id < 8) && pipeline_gen >= 5 && pipeline_gen < 15;
          if (ik_rec)
          {
            RBK_IK_BEG(CompWaitFull, pipeline_gen);
          }
          wait_parity(&full[slot_id], key_ring.parity);
          if (ik_rec)
          {
            RBK_IK_END(CompWaitFull, pipeline_gen);
          }
          const int tile_id = tile_id_buf[slot_id];
          if (tile_id >= num_tiles)
          {
            // every squad breaks on the tile check BEFORE waiting computed/prefixed, so a
            // sentinel slot needs no arrivals
            break;
          }
          // slot is ready! compute is FLAGS-ONLY (the disaggregation design): the count chain
          // publishes at flag-fold and never waits on a single value read
          const int tile_len  = (int) min((OffT) tile_size, num_items - (OffT) tile_id * tile_size);
          int local_run_count = 0;
          if (ik_rec)
          {
            RBK_IK_BEG(CompFlag, pipeline_gen);
          }
          unsigned my_flags;
          if (keys_staged)
          {
            const KeyT* key_buf = tile_buf + (size_t) slot_id * slot_stride + slot_pad;
            my_flags            = compute_head_flags<items_per_thread, false>(
              key_buf, warp_tile_offset, tile_len, tile_id, lane_id, skip_elems);
          }
          else
          {
            // vvv regressed case: we load compute flags straight from global vvv
            const KeyT* key_buf = d_keys + (size_t) tile_id * tile_size;
            my_flags =
              compute_head_flags<items_per_thread, true>(key_buf, warp_tile_offset, tile_len, tile_id, lane_id, 0);
            // ^^^ regressed case ^^^
          }
          local_run_count = __reduce_add_sync(full_mask, __popc(my_flags));
          if (lane_id == 0)
          {
            warp_run_counts[slot_id][compute_warp_id] = local_run_count;
            ptx::mbarrier_arrive(&computed[slot_id]); // each compute warp arrives
          }
          // warp 0 waits all compute warp arrivals so that every warp's results are visible,
          // then folds and publishes the tile's COUNT state
          if (ik_rec)
          {
            RBK_IK_END(CompFlag, pipeline_gen);
          }
          if (compute_warp_id == 0)
          {
            if (ik_rec)
            {
              RBK_IK_BEG(CompPublish, pipeline_gen);
            }
            wait_parity(&computed[slot_id], key_ring.parity);
            reduce_and_publish_count<compute_warps>(count_states, tile_id, warp_run_counts[slot_id], lane_id);
            if (ik_rec)
            {
              RBK_IK_END(CompPublish, pipeline_gen);
            }
          }
          // no position staging: consumers decode run boundaries from the flag words. The words
          // also PERSIST to global -- they are the whole run structure the value pass needs, so
          // values never share this kernel at all (the two-pass design)
          head_flag_buf[slot_id][compute_warp_id * 32 + lane_id]                                 = my_flags;
          d_flag_words[(size_t) tile_id * (compute_warps * 32) + compute_warp_id * 32 + lane_id] = my_flags;
          __syncwarp();
          // ---- the store half: all counts, then the prefix, then drain OWN warp tile ----
          if (ik_rec)
          {
            RBK_IK_BEG(KWaitComputed, pipeline_gen);
          }
          wait_parity(&computed[slot_id], key_ring.parity); // no-op for the publisher, it already waited
          if (ik_rec)
          {
            RBK_IK_END(KWaitComputed, pipeline_gen);
          }
          const auto [lane_warp_tile_run_count, lane_runs_before_warp_tile] =
            scan_warp_tile_run_counts<compute_warps>(warp_run_counts[slot_id], lane_id);
          if (ik_rec)
          {
            RBK_IK_BEG(KWaitPrefixed, pipeline_gen);
          }
          wait_parity(&prefixed[slot_id], key_ring.parity);
          if (ik_rec)
          {
            RBK_IK_END(KWaitPrefixed, pipeline_gen);
          }
          if (ik_rec)
          {
            RBK_IK_BEG(KDrain, pipeline_gen);
          }
          const OffT curr_prefix_run_count = prefix_packed[slot_id].run_count();
          const int tile_total_runs =
            __shfl_sync(full_mask, lane_runs_before_warp_tile + lane_warp_tile_run_count, compute_warps - 1);
          if (compute_warp_id == 0 && lane_id == 0)
          {
            d_tile_prefix[tile_id] = prefix_packed[slot_id];
            if (tile_id == num_tiles - 1)
            {
              *d_num_runs = (NumRunsT) (curr_prefix_run_count + tile_total_runs);
            }
          }
          const int warp_tile_run_count   = __shfl_sync(full_mask, lane_warp_tile_run_count, compute_warp_id);
          const int runs_before_warp_tile = __shfl_sync(full_mask, lane_runs_before_warp_tile, compute_warp_id);
          if (warp_tile_run_count != 0)
          {
            const KeyT* tile_keys                   = keys_staged ? tile_buf + (size_t) slot_id * slot_stride + slot_pad
                                                                  : d_keys + (size_t) tile_id * tile_size;
            const int key_skip                      = keys_staged ? skip_elems : 0;
            const OffT global_runs_before_warp_tile = curr_prefix_run_count + runs_before_warp_tile;
#  ifdef K_DRAIN_THRESH
            constexpr int word_serial_threshold = K_DRAIN_THRESH;
#  else
            constexpr int word_serial_threshold = 256;
#  endif
            if (warp_tile_run_count >= staging_threshold && warp_tile_run_count <= word_serial_threshold)
            {
              // word-serial band (IKET receipt: KDrain flat 1.38us/gen at seg4-16 = the broadcast
              // band's fixed 2-shuffles-per-word loop on the MIO pipe): each lane owns its own
              // flag word and extracts its runs serially -- no shuffles, no sync collectives, so
              // the divergence is safe; one smem read + one store per RUN
              const unsigned my_word = head_flag_buf[slot_id][compute_warp_id * 32 + lane_id];
              const int my_popc      = __popc(my_word);
              typename WarpScan<int>::TempStorage warp_scan_storage;
              int word_scan;
              WarpScan<int>(warp_scan_storage).InclusiveSum(my_popc, word_scan);
              int rank     = word_scan - my_popc;
              unsigned rem = my_word;
              while (rem != 0)
              {
                const int bit = __ffs(rem) - 1;
                rem &= rem - 1;
                d_unique[global_runs_before_warp_tile + rank] =
                  tile_keys[warp_tile_offset + lane_id * 32 + bit + key_skip];
                ++rank;
              }
            }
            else if (warp_tile_run_count >= staging_threshold)
            {
              // row-stream band for ANY non-sparse count: the rows are SMEM reads, so the
              // amplification at lower run counts costs LDS bandwidth, not DRAM (the flag-decode
              // per-run gather costs ~15 shuffle ops each -- the seg16 KEYS anomaly)
              // dense band: every run's key from coalesced key rows, emitted at head lanes --
              // no position reads (kept OUT of any scan loop; this loop has no cross-row deps)
              const unsigned my_word = head_flag_buf[slot_id][compute_warp_id * 32 + lane_id];
              const int my_popc      = __popc(my_word);
              typename WarpScan<int>::TempStorage warp_scan_storage;
              int word_scan;
              WarpScan<int>(warp_scan_storage).InclusiveSum(my_popc, word_scan);
              const int my_runs_before_word = word_scan - my_popc;
              const unsigned upto_l         = (lane_id == 31) ? 0xffffffffu : ((2u << lane_id) - 1);
#  pragma unroll
              for (int iter = 0; iter < items_per_thread; ++iter)
              {
                const unsigned w    = __shfl_sync(full_mask, my_word, iter);
                const int word_base = __shfl_sync(full_mask, my_runs_before_word, iter);
                if ((w >> lane_id) & 1u)
                {
                  const int run_idx                                = word_base + __popc(w & upto_l) - 1;
                  const int loc                                    = warp_tile_offset + iter * 32 + lane_id;
                  d_unique[global_runs_before_warp_tile + run_idx] = tile_keys[loc + key_skip];
                }
              }
            }
            else
            {
              // sparse band: decode head positions from the flag words. decode_run shuffles with
              // the full mask, so every round must be executed by every lane (divergent per-lane
              // loops around it DEADLOCK -- receipted the hard way)
              const HeadFlagDecodeT dec(head_flag_buf[slot_id], compute_warp_id, lane_id);
              const int rounds = (warp_tile_run_count + 31) >> 5;
              for (int it = 0; it < rounds; ++it)
              {
                const int run_idx  = it * 32 + lane_id;
                const RunSpanT run = dec.decode_run(run_idx < warp_tile_run_count ? run_idx : 0);
                if (run_idx < warp_tile_run_count)
                {
                  d_unique[global_runs_before_warp_tile + run_idx] =
                    tile_keys[warp_tile_offset + run.head_pos_in_warp_tile + key_skip];
                }
              }
            }
          }
          if (ik_rec)
          {
            RBK_IK_END(KDrain, pipeline_gen);
          }
          __syncwarp();
          if (lane_id == 0)
          {
            ptx::mbarrier_arrive(&empty[slot_id]); // keys drained: the ring recycles
          }
        }
      }
      // if you are poll
      else if (squad == squadPoll)
      {
        int last_seen_tile_id           = 0;
        OffT last_seen_prefix_run_count = 0;
        int last_tile_with_runs         = -1; // most recent tile known to contain a run head
        int poll_dense_mode             = 1;
        RingCursorT key_ring;
        for (int pipeline_gen = 0;; ++pipeline_gen, key_ring.advance(key_ring_stages))
        {
          const int slot_id = key_ring.slot;
          wait_parity(&full[slot_id], key_ring.parity);
          const int tile_id = tile_id_buf[slot_id];
          if (tile_id >= num_tiles)
          {
            if (lane_id == 0)
            {
              ptx::mbarrier_arrive(&prefixed[slot_id]);
            }
            break;
          }
          OffT curr_prefix_run_count;
          poll_and_fold<PolicySelector>(
            count_states,
            tile_id,
            last_seen_tile_id,
            last_seen_prefix_run_count,
            last_tile_with_runs,
            lane_id,
            poll_dense_mode,
            curr_prefix_run_count);
          __syncwarp();
          if (lane_id == 0)
          {
            prefix_packed[slot_id] = PrefixT::pack(curr_prefix_run_count, last_tile_with_runs);
            ptx::mbarrier_arrive(&prefixed[slot_id]); // prefix ready, key stores + value warps may proceed
          }
        }
      }
    });
}

template <class PolicySelector, class StateT>
_CCCL_KERNEL_ATTRIBUTES void DeviceReduceByKeyLookaheadInitKernel(StateT* states, ::cuda::std::int64_t n_states)
{
  const ::cuda::std::int64_t i = (::cuda::std::int64_t) blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_states)
  {
    states[i] = StateT{};
  }
}

// THE VALUE PASS: pipeline-free. One 8-warp block per tile; the persisted flag words are the
// whole run structure, the persisted tile prefix is the whole cross-tile coordination. No rings,
// no barriers beyond __syncthreads, no lookahead, no work stealing -- latency hides behind plain
// occupancy. Emits every within-tile-closed aggregate + the boundary record for the cleanup pass.
template <typename PolicySelector, class ValueT, class OffT>
__launch_bounds__(current_policy<PolicySelector>().lookahead.compute_warps * 32, 6)
  _CCCL_KERNEL_ATTRIBUTES void DeviceReduceByKeyLookaheadValueKernel(
    const ValueT* __restrict__ d_values,
    ValueT* __restrict__ d_aggregates,
    const unsigned* __restrict__ d_flag_words,
    const PrefixT<OffT>* __restrict__ d_tile_prefix,
    TileValueRecordT<ValueT, OffT>* __restrict__ value_records,
    OffT num_items,
    int num_tiles)
{
  static constexpr RleLookaheadPolicy policy = current_policy<PolicySelector>().lookahead;
  constexpr int items_per_thread             = policy.items_per_thread;
  constexpr int compute_warps                = policy.compute_warps;
  constexpr int warp_tile_size               = policy.warp_tile_size();
  constexpr int tile_size                    = policy.tile_size();
#  ifdef RBK_STREAM_DIV
  constexpr int stream_threshold = warp_tile_size / RBK_STREAM_DIV;
#  else
  // B200 receipt: /4 pulls the seg4 band into the stream and costs nothing anywhere else
  constexpr int stream_threshold = warp_tile_size / 4;
#  endif
  __shared__ int wt_counts[compute_warps];
  __shared__ ValueT wt_leads[compute_warps];
  __shared__ ValueT wt_tails[compute_warps];

  const int tile_id = (int) blockIdx.x;
  if (tile_id >= num_tiles)
  {
    return;
  }
  const bool ik_rec = tile_id < 8;
  if (ik_rec)
  {
    RBK_IK_BEG(VSetup, tile_id);
  }
  const int wt                  = (int) (threadIdx.x >> 5);
  const int lane_id             = (int) (threadIdx.x & 31);
  const int tile_len            = (int) min((OffT) tile_size, num_items - (OffT) tile_id * tile_size);
  const ValueT* tile_vals       = d_values + (size_t) tile_id * tile_size;
  const unsigned my_word        = d_flag_words[(size_t) tile_id * (compute_warps * 32) + wt * 32 + lane_id];
  const int warp_tile_run_count = __reduce_add_sync(full_mask, __popc(my_word));
  if (lane_id == 0)
  {
    wt_counts[wt] = warp_tile_run_count;
  }
  __syncthreads();
  const auto [lane_warp_tile_run_count, lane_runs_before_warp_tile] =
    scan_warp_tile_run_counts<compute_warps>(wt_counts, lane_id);
  const int runs_before_warp_tile         = __shfl_sync(full_mask, lane_runs_before_warp_tile, wt);
  const OffT curr_prefix_run_count        = d_tile_prefix[tile_id].run_count();
  const OffT global_runs_before_warp_tile = curr_prefix_run_count + runs_before_warp_tile;
  const int warp_tile_offset              = wt * warp_tile_size;
  const int wt_end                        = min(warp_tile_offset + warp_tile_size, tile_len);
  if (ik_rec)
  {
    RBK_IK_END(VSetup, tile_id);
  }
  if (ik_rec)
  {
    RBK_IK_BEG(VEmit, tile_id);
  }
  // boundary sums + the banded within-warp-tile emission (the receipted forms)
  ValueT wt_lead{};
  ValueT wt_tail{};
  if (warp_tile_run_count == 0)
  {
    wt_lead = warp_span_sum(tile_vals, warp_tile_offset, wt_end, lane_id);
    wt_tail = wt_lead; // head-free: the whole warp tile leads AND trails
  }
  else if (warp_tile_run_count >= stream_threshold)
  {
    stream_values_from_flags<items_per_thread>(
      d_aggregates, tile_vals, my_word, global_runs_before_warp_tile, warp_tile_offset, tile_len, lane_id, wt_tail);
    const HeadFlagDecodeT dec(my_word, lane_id);
    const RunSpanT first_run = dec.decode_run(0);
    wt_lead = warp_span_sum(tile_vals, warp_tile_offset, warp_tile_offset + first_run.head_pos_in_warp_tile, lane_id);
  }
  else
  {
    const HeadFlagDecodeT dec(my_word, lane_id);
    if (warp_tile_run_count < 16)
    {
      // long-span band (mean span >= 64): sub-warp cooperative walks, 8 lanes per span in waves
      // of 4 runs. Cutoff receipts: 8 dropped seg256's ~8x128 shape into one-lane serial walks
      // (26% trough -> 43% at cutoff 32); 32 dragged seg64's ~31x32 shape into wave overhead
      // (-11%). Span-driven boundary.
      const RunSpanT lane_run = dec.decode_run(lane_id < warp_tile_run_count ? lane_id : 0);
      if (warp_tile_run_count < 4)
      {
        // very few, very long spans: ALL 32 lanes walk each span (8-lane waves here quadruple
        // the serial depth exactly where spans are longest -- the 1K/4K regression receipt;
        // crossover receipts: 8 re-serialized seg256's ~7-run tiles, 701 vs 650us)
        for (int run_idx = 0; run_idx + 1 < warp_tile_run_count; ++run_idx)
        {
          const int head   = __shfl_sync(full_mask, lane_run.head_pos_in_warp_tile, run_idx);
          const int next   = __shfl_sync(full_mask, lane_run.next_head_pos, run_idx);
          const ValueT agg = warp_span_sum(tile_vals, warp_tile_offset + head, warp_tile_offset + next, lane_id);
          if (lane_id == 0)
          {
            d_aggregates[global_runs_before_warp_tile + run_idx] = agg;
          }
        }
      }
      else
      {
        const int sub_lane  = lane_id & 7;
        const int sub_group = lane_id >> 3;
        for (int wave = 0; wave + 1 <= warp_tile_run_count; wave += 4)
        {
          const int run_idx    = wave + sub_group;
          const bool active    = run_idx + 1 < warp_tile_run_count; // the last run is the closes' job
          const int head       = __shfl_sync(full_mask, lane_run.head_pos_in_warp_tile, run_idx & 31);
          const int next       = __shfl_sync(full_mask, lane_run.next_head_pos, run_idx & 31);
          const int span_begin = warp_tile_offset + head;
          const int span_end   = active ? warp_tile_offset + next : span_begin;
          ValueT agg{};
          for (int pos = span_begin + sub_lane; pos < span_end; pos += 8)
          {
            agg += tile_vals[pos];
          }
#  pragma unroll
          for (int offset = 4; offset; offset >>= 1)
          {
            agg += __shfl_xor_sync(full_mask, agg, offset);
          }
          if (active && sub_lane == 0)
          {
            d_aggregates[global_runs_before_warp_tile + run_idx] = agg;
          }
        }
      }
    }
    else
    {
      // mid band: one run per lane, plain pre-seeded span walks (decode rounds are warp-uniform)
      const int rounds = (warp_tile_run_count + 31) >> 5;
      for (int it = 0; it < rounds; ++it)
      {
        const int run_idx  = it * 32 + lane_id;
        const RunSpanT run = dec.decode_run(run_idx < warp_tile_run_count ? run_idx : 0);
        if (run_idx + 1 < warp_tile_run_count)
        {
          d_aggregates[global_runs_before_warp_tile + run_idx] = span_sum_prefetched(
            tile_vals, warp_tile_offset + run.head_pos_in_warp_tile, warp_tile_offset + run.next_head_pos);
        }
      }
    }
    const RunSpanT first_run = dec.decode_run(0);
    const RunSpanT last_run  = dec.decode_run(warp_tile_run_count - 1);
    wt_lead = warp_span_sum(tile_vals, warp_tile_offset, warp_tile_offset + first_run.head_pos_in_warp_tile, lane_id);
    wt_tail = warp_span_sum(tile_vals, warp_tile_offset + last_run.head_pos_in_warp_tile, wt_end, lane_id);
  }
  if (lane_id == 0)
  {
    wt_leads[wt] = wt_lead;
    wt_tails[wt] = wt_tail;
  }
  if (ik_rec)
  {
    RBK_IK_END(VEmit, tile_id);
  }
  __syncthreads();
  if (ik_rec)
  {
    RBK_IK_BEG(VBoundary, tile_id);
  }
  // warp 0: the record + the within-tile boundary closes (aggregate chains over the wt sums)
  if (wt == 0)
  {
    const int lane_count                    = (lane_id < compute_warps) ? wt_counts[lane_id] : 0;
    const unsigned nonempty_warp_tiles_mask = __ballot_sync(full_mask, lane_count > 0);
    const bool any_head                     = (nonempty_warp_tiles_mask != 0);
    const bool is_last_tile                 = (tile_id == num_tiles - 1);
    {
      const int last_headed  = any_head ? (31 - __clz(nonempty_warp_tiles_mask)) : 0;
      const int first_headed = any_head ? (__ffs(nonempty_warp_tiles_mask) - 1) : compute_warps;
      ValueT open_agg        = (lane_id < compute_warps && lane_id >= last_headed) ? wt_tails[lane_id] : ValueT{};
      ValueT lead_agg        = (lane_id < compute_warps && lane_id < first_headed) ? wt_tails[lane_id] : ValueT{};
      if (any_head && lane_id == first_headed)
      {
        lead_agg += wt_leads[first_headed];
      }
#  pragma unroll
      for (int offset = 16; offset; offset >>= 1)
      {
        open_agg += __shfl_xor_sync(full_mask, open_agg, offset);
        lead_agg += __shfl_xor_sync(full_mask, lead_agg, offset);
      }
      if (lane_id == 0)
      {
        TileValueRecordT<ValueT, OffT> rec;
        rec.open_agg = open_agg;
        rec.lead_agg = lead_agg;
        rec.boundary_dst =
          (curr_prefix_run_count > 0 && (any_head || is_last_tile)) ? (OffT) (curr_prefix_run_count - 1) : (OffT) -1;
        rec.boundary_from      = d_tile_prefix[tile_id].last_tile_with_runs();
        value_records[tile_id] = rec;
      }
    }
    if (lane_id < compute_warps && lane_count > 0)
    {
      const unsigned later = nonempty_warp_tiles_mask >> (lane_id + 1);
      // this warp's scan already gave lane i warp-tile i's runs-before
      const OffT last_run_global_idx = curr_prefix_run_count + lane_runs_before_warp_tile + lane_count - 1;
      if (later)
      {
        const int next_headed = lane_id + 1 + __ffs(later) - 1;
        ValueT closing        = wt_tails[lane_id];
        for (int w2 = lane_id + 1; w2 < next_headed; ++w2)
        {
          closing += wt_tails[w2]; // head-free middles: whole sums
        }
        closing += wt_leads[next_headed];
        d_aggregates[last_run_global_idx] = closing;
      }
      else if (is_last_tile)
      {
        ValueT closing = wt_tails[lane_id];
        for (int w2 = lane_id + 1; w2 < compute_warps; ++w2)
        {
          closing += wt_tails[w2];
        }
        d_aggregates[last_run_global_idx] = closing;
      }
      // else: open into the next tile -- the cleanup pass closes it
    }
  }
  if (ik_rec)
  {
    RBK_IK_END(VBoundary, tile_id);
  }
}

// boundary cleanup: one warp per tile that owes an entering-run close. Runs AFTER the main
// kernel (the launch boundary synchronizes), reads plain records, writes each cross-tile run's
// aggregate = open(from) + whole-sums of the head-free tiles between + lead(this). Every window
// is disjoint, sums fold in fixed order (deterministic), total work is O(num_tiles) amortized.
template <class ValueT, class OffT>
_CCCL_KERNEL_ATTRIBUTES void DeviceReduceByKeyLookaheadCleanupKernel(
  ValueT* d_aggregates, TileValueRecordT<ValueT, OffT>* value_records, int num_tiles)
{
  const int warp_global = (int) ((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
  const int lane_id     = (int) (threadIdx.x & 31);
  if (warp_global >= num_tiles)
  {
    return;
  }
  const TileValueRecordT<ValueT, OffT> rec = value_records[warp_global];
  if (rec.boundary_dst < 0)
  {
    return;
  }
  const int from = (rec.boundary_from >= 0) ? rec.boundary_from : 0;
  ValueT carry{};
  for (int base = from; base < warp_global; base += 32)
  {
    const int t = base + lane_id;
    // open_agg of the window start is its tail chain; the head-free tiles after it contribute
    // their whole-tile sums (== their open_agg)
    ValueT part = (t < warp_global) ? value_records[t].open_agg : ValueT{};
#  pragma unroll
    for (int offset = 16; offset; offset >>= 1)
    {
      part += __shfl_xor_sync(full_mask, part, offset);
    }
    carry += part;
  }
  if (lane_id == 0)
  {
    d_aggregates[rec.boundary_dst] = carry + rec.lead_agg;
  }
}

template <typename PolicySelector>
[[nodiscard]] _CCCL_HOST_DEVICE_API _CCCL_CONSTEVAL int get_device_reduce_by_key_lookahead_launch_bounds() noexcept
{
  return num_total_threads(current_policy<PolicySelector>().lookahead);
}

// need a variable template for clang in CUDA mode to avoid:
// error: 'launch_bounds' attribute requires parameter 0 to be an integer constant
template <typename PolicySelector>
inline constexpr int device_reduce_by_key_lookahead_launch_bounds =
  get_device_reduce_by_key_lookahead_launch_bounds<PolicySelector>();

template <typename PolicySelector, class KeyT, class ValueT, class NumRunsT, class OffT>
__launch_bounds__(device_reduce_by_key_lookahead_launch_bounds<PolicySelector>, 1)
  _CCCL_KERNEL_ATTRIBUTES void DeviceReduceByKeyLookaheadKernel(
    const KeyT* __restrict__ d_keys,
    const ValueT* __restrict__ d_values,
    KeyT* __restrict__ d_unique,
    ValueT* __restrict__ d_aggregates,
    NumRunsT* __restrict__ d_num_runs,
    CountStateT* count_states,
    unsigned* d_flag_words,
    PrefixT<OffT>* d_tile_prefix,
    OffT num_items,
    int num_tiles,
    int key_ring_stages,
    int pos_ring_stages,
    bool keys_staged)
{
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (device_reduce_by_key_lookahead_body<PolicySelector>(
       d_keys,
       d_values,
       d_unique,
       d_aggregates,
       d_num_runs,
       count_states,
       d_flag_words,
       d_tile_prefix,
       num_items,
       num_tiles,
       key_ring_stages,
       pos_ring_stages,
       keys_staged);))
}
#endif // __cccl_ptx_isa >= 920
} // namespace detail::reduce_by_key::lookahead

CUB_NAMESPACE_END
