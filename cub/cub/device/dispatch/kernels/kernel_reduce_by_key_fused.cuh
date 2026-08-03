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

#include <cub/device/dispatch/kernels/kernel_reduce_by_key_lookahead.cuh>

CUB_NAMESPACE_BEGIN

namespace detail::reduce_by_key::lookahead
{
#if __cccl_ptx_isa >= 920

// fused single pass: per-tile lookback state. tag 0 = unpublished (init clears), 1 = PARTIAL
// (own run count), 2 = INCLUSIVE (prefix + own). The inclusive publish is a RELEASE store made
// AFTER the tile's PrefixT store, so a reader that acquires tag 2 may read PrefixT[t] safely.
struct FusedStateT
{
  ::cuda::std::uint64_t dword;

  _CCCL_DEVICE_API _CCCL_FORCEINLINE unsigned tag() const
  {
    return (unsigned) (dword >> 62);
  }

  _CCCL_DEVICE_API _CCCL_FORCEINLINE ::cuda::std::int64_t count() const
  {
    return (::cuda::std::int64_t) (dword & 0x3fffffffffffffffull);
  }

  static _CCCL_DEVICE_API _CCCL_FORCEINLINE FusedStateT pack(unsigned tag, ::cuda::std::int64_t count)
  {
    return {((::cuda::std::uint64_t) tag << 62) | (::cuda::std::uint64_t) count};
  }
};

// ===== FUSED SINGLE PASS: keys + values in one block-per-tile kernel; head flags never leave
// the SM (temp storage = O(tiles): FusedStateT + PrefixT + TileValueRecordT). Design doc:
// ~/.claude/plans/rbk-fused-single-pass.md =====
template <typename PolicySelector,
          class KeyT,
          class ValueT,
          class NumRunsT,
          class OffT,
          class ReductionOpT = ::cuda::std::plus<>>
__launch_bounds__(current_policy<PolicySelector>().lookahead.compute_warps * 32)
  _CCCL_KERNEL_ATTRIBUTES void DeviceReduceByKeyFusedKernel(
    const KeyT* __restrict__ d_keys,
    const ValueT* __restrict__ d_values,
    KeyT* __restrict__ d_unique,
    ValueT* __restrict__ d_aggregates,
    NumRunsT* __restrict__ d_num_runs,
    FusedStateT* fused_states,
    PrefixT<OffT>* d_tile_prefix,
    TileValueRecordT<ValueT, OffT>* __restrict__ value_records,
    OffT num_items,
    int num_tiles,
    ReductionOpT reduction_op = {})
{
  static constexpr RleLookaheadPolicy policy = current_policy<PolicySelector>().lookahead;
  constexpr int items_per_thread             = policy.items_per_thread;
  constexpr int compute_warps                = policy.compute_warps;
  constexpr int warp_tile_size               = policy.warp_tile_size();
  constexpr int tile_size                    = policy.tile_size();
#  ifdef RBK_STREAM_DIV
  constexpr int stream_threshold = warp_tile_size / RBK_STREAM_DIV;
#  else
  constexpr int stream_threshold = warp_tile_size / 4;
#  endif
  constexpr int pad_elems = 16 / (int) sizeof(KeyT); // 16B head pad: the tile-boundary predecessor
#  ifdef RBK_FUSED_KEYS_GLOBAL
  // occupancy variant: keys are read from GLOBAL (once, streaming; L1 pairs the i/i-1 reads).
  // smem holds only the values tile -> ~33KB/block -> 6 blocks/SM (48 warps vs 24 staged-keys).
  constexpr size_t keys_bytes = 0;
#  else
  constexpr size_t keys_bytes = (((size_t) (tile_size + pad_elems) * sizeof(KeyT)) + 15) & ~(size_t) 15;
#  endif
  extern __shared__ char fused_smem_raw[];
  KeyT* const staged_keys   = (KeyT*) fused_smem_raw;
  ValueT* const staged_vals = (ValueT*) (fused_smem_raw + keys_bytes);
  __shared__ ::cuda::std::uint64_t keys_bar;
  __shared__ ::cuda::std::uint64_t vals_bar;
  __shared__ int wt_counts[compute_warps];
  __shared__ ValueT wt_leads[compute_warps];
  __shared__ ValueT wt_tails[compute_warps];
  __shared__ int wt_leads_has[compute_warps];
  __shared__ int wt_tails_has[compute_warps];
  __shared__ OffT s_prefix;
  __shared__ int s_last_runs_tile;

  const int tile_id  = (int) blockIdx.x;
  const int tile_len = (int) min((OffT) tile_size, num_items - (OffT) tile_id * tile_size);
#  ifdef RBK_FUSED_KEYS_GLOBAL
  const KeyT* const keys_tile = d_keys + (size_t) tile_id * tile_size;
#  endif
  if (threadIdx.x == 0)
  {
    ptx::mbarrier_init(&keys_bar, 1);
    ptx::mbarrier_init(&vals_bar, 1);
    ptx::fence_proxy_async(ptx::space_shared);
#  ifndef RBK_FUSED_KEYS_GLOBAL
    // keys with a 16B head pad holding the tile-boundary predecessor. Tile 0 has no
    // predecessor: it stages at dst+16 from an in-bounds source (its pad is uninitialized
    // smem; is_global_first forces the head at loc 0) -- a negative global src is NOT a
    // valid ignore_oob use (the reference only aligns down WITHIN the allocation)
    if (tile_id == 0)
    {
      const unsigned kb    = (unsigned) ((size_t) tile_len * sizeof(KeyT));
      const unsigned kspan = (kb + 15u) & ~15u;
      ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &keys_bar, kspan);
      ptx::cp_async_bulk_ignore_oob(
        ptx::space_shared, ptx::space_global, staged_keys + pad_elems, d_keys, kspan, 0u, kspan - kb, &keys_bar);
    }
    else
    {
      const unsigned kb    = (unsigned) ((size_t) (tile_len + pad_elems) * sizeof(KeyT));
      const unsigned kspan = (kb + 15u) & ~15u;
      ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &keys_bar, kspan);
      ptx::cp_async_bulk_ignore_oob(
        ptx::space_shared,
        ptx::space_global,
        staged_keys,
        d_keys + (size_t) tile_id * tile_size - pad_elems,
        kspan,
        0u,
        kspan - kb,
        &keys_bar);
    }
#  endif // !RBK_FUSED_KEYS_GLOBAL
#  ifdef RBK_FUSED_NOVALS
    ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &vals_bar, 0);
#  else
    const unsigned vb    = (unsigned) ((size_t) tile_len * sizeof(ValueT));
    const unsigned vspan = (vb + 15u) & ~15u;
    ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &vals_bar, vspan);
    ptx::cp_async_bulk_ignore_oob(
      ptx::space_shared,
      ptx::space_global,
      staged_vals,
      d_values + (size_t) tile_id * tile_size,
      vspan,
      0u,
      vspan - vb,
      &vals_bar);
#  endif
  }
  __syncthreads();

  const int wt      = (int) (threadIdx.x >> 5);
  const int lane_id = (int) (threadIdx.x & 31);
#  ifdef RBK_FUSED_WAITSWAP
  wait_parity(&vals_bar, 0); // discriminator: is the staleness tied to the FIRST wait?
#  endif
#  ifndef RBK_FUSED_KEYS_GLOBAL
  wait_parity(&keys_bar, 0);
#  endif
#  ifdef RBK_FUSED_DELAY
  __nanosleep(4000); // experiment: does the first flag read go stale because the wait
                     // completes before the data (tx accounting) or because of ordering?
#  endif
#  ifdef RBK_FUSED_FENCE
  __threadfence_block(); // experiment: ordering (async-proxy visibility), not timing
#  endif
#  ifdef RBK_FUSED_STOP1
  return; // gate: after keys TMA wait
#  endif
#  ifdef RBK_FUSED_DEBUG
  if (tile_id == 1 && wt == 0)
  {
    d_aggregates[96 + lane_id] = (ValueT) staged_keys[pad_elems + lane_id]; // FIRST reads
  }
#  endif
  // buffer-predecessor flag pass (the RLE reference form): pred = key_buf[idx-1], legal via
  // the staged 16B head pad. The shuffle-carry form MISCOMPILES here (nvcc 13.5/sm_120a:
  // first inlined copy drops the iter-0 compare chain feeding the first ballot -- receipts
  // in ~/.claude/plans/rbk-fused-single-pass.md)
  unsigned my_word = 0;
#  pragma unroll
  for (int iter = 0; iter < items_per_thread; ++iter)
  {
    const int loc             = wt * warp_tile_size + iter * 32 + lane_id;
    const int is_global_first = (tile_id == 0 && loc == 0);
#  ifdef RBK_FUSED_KEYS_GLOBAL
    const int in_bounds = loc < tile_len;
    const KeyT key      = in_bounds ? keys_tile[loc] : KeyT{};
    const KeyT pred     = (in_bounds && !is_global_first) ? keys_tile[loc - 1] : KeyT{};
    const int head      = in_bounds ? (is_global_first ? 1 : !(key == pred)) : 0;
#  else
    const KeyT key  = staged_keys[pad_elems + loc];
    const KeyT pred = staged_keys[pad_elems + loc - 1];
    const int head  = (loc < tile_len) ? (is_global_first ? 1 : !(key == pred)) : 0;
#  endif
    const unsigned flags = __ballot_sync(full_mask, head);
#  ifdef RBK_FUSED_DEBUG
    if (iter == 0 && tile_id == 1 && wt == 0)
    {
      d_aggregates[160 + lane_id] = (ValueT) head; // per-lane head at iter 0
      d_aggregates[192 + lane_id] = (ValueT) flags; // ballot result at iter 0
      d_aggregates[224 + lane_id] = (ValueT) key; // the loop's OWN key read
    }
#  endif
    if (lane_id == iter)
    {
      my_word = flags;
    }
  }
#  ifdef RBK_FUSED_DEBUG
  if (tile_id == 1 && wt == 0)
  {
    d_aggregates[lane_id]   = (ValueT) my_word;
    const unsigned my_word2 = compute_head_flags<items_per_thread, false>(
      staged_keys, wt * warp_tile_size, tile_len, tile_id, lane_id, pad_elems);
    d_aggregates[32 + lane_id] = (ValueT) my_word2; // recompute: differs => race
  }
  return;
#  endif
  const int warp_tile_run_count = __reduce_add_sync(full_mask, __popc(my_word));
  if (lane_id == 0)
  {
    wt_counts[wt] = warp_tile_run_count;
  }
  __syncthreads();
  const auto [lane_warp_tile_run_count, lane_runs_before_warp_tile] =
    scan_warp_tile_run_counts<compute_warps>(wt_counts, lane_id);
  const int runs_before_warp_tile = __shfl_sync(full_mask, lane_runs_before_warp_tile, wt);
  const int tile_total_runs =
    __shfl_sync(full_mask, lane_runs_before_warp_tile + lane_warp_tile_run_count, compute_warps - 1);

  // publish PARTIAL + decoupled LOOKBACK (warp 0); everyone else proceeds to the values wait
  if (wt == 0)
  {
    if (lane_id == 0 && tile_id > 0 && tile_id != num_tiles - 1)
    {
      publish_state(fused_states, tile_id, FusedStateT::pack(1u, tile_total_runs));
    }
    OffT prefix        = 0;
    int last_runs_tile = -1;
    int t              = tile_id;
    while (t > 0)
    {
      const int base    = (t - 32 > 0) ? (t - 32) : 0;
      const int idx     = base + lane_id;
      const bool active = idx < t;
      FusedStateT st{0};
      unsigned pending;
      do
      {
        if (active && st.tag() == 0u)
        {
          ::cuda::atomic_ref<::cuda::std::uint64_t, ::cuda::thread_scope_device> a(fused_states[idx].dword);
          st = FusedStateT{a.load(::cuda::memory_order_acquire)};
        }
        pending = __ballot_sync(full_mask, active && st.tag() == 0u);
      } while (pending != 0u);
      const unsigned incl_mask = __ballot_sync(full_mask, active && st.tag() == 2u);
      const int cut            = (incl_mask != 0u) ? (31 - __clz(incl_mask)) : -1;
      const bool counted       = active && lane_id > cut;
      prefix += (OffT) __reduce_add_sync(full_mask, counted ? (int) st.count() : 0);
      const unsigned runs_mask = __ballot_sync(full_mask, counted && st.count() > 0);
      if (last_runs_tile < 0 && runs_mask != 0u)
      {
        last_runs_tile = base + (31 - __clz(runs_mask));
      }
      if (cut >= 0)
      {
        const ::cuda::std::int64_t incl_count = __shfl_sync(full_mask, (long long) st.count(), cut);
        prefix += (OffT) incl_count;
        if (last_runs_tile < 0)
        {
          // the inclusive tile's own count decides whether IT is the last headed tile
          const PrefixT<OffT> p = d_tile_prefix[base + cut]; // safe: released before its tag-2
          last_runs_tile        = ((OffT) incl_count - p.run_count() > 0) ? (base + cut) : p.last_tile_with_runs();
        }
        break;
      }
      t = base;
    }
    if (lane_id == 0)
    {
      d_tile_prefix[tile_id] = PrefixT<OffT>::pack(prefix, last_runs_tile);
      ::cuda::atomic_ref<::cuda::std::uint64_t, ::cuda::thread_scope_device> a(fused_states[tile_id].dword);
      a.store(FusedStateT::pack(2u, (::cuda::std::int64_t) (prefix + tile_total_runs)).dword,
              ::cuda::memory_order_release);
      s_prefix         = prefix;
      s_last_runs_tile = last_runs_tile;
      if (tile_id == num_tiles - 1)
      {
        *d_num_runs = (NumRunsT) (prefix + tile_total_runs);
      }
    }
  }
  __syncthreads();
#  ifdef RBK_FUSED_STOP2
  return; // gate: after lookback
#  endif
  const OffT curr_prefix_run_count        = s_prefix;
  const OffT global_runs_before_warp_tile = curr_prefix_run_count + runs_before_warp_tile;
  const int warp_tile_offset              = wt * warp_tile_size;
  const int wt_end                        = min(warp_tile_offset + warp_tile_size, tile_len);

#  ifdef RBK_FUSED_DEBUG2
  wait_parity(&vals_bar, 0); // MUST drain the in-flight TMA before returning (CTA exit with
                             // an outstanding bulk transfer = device fault)
  if (tile_id == 0 && lane_id == 0)
  {
    d_aggregates[300 + wt * 8 + 0] = (ValueT) warp_tile_run_count;
    d_aggregates[300 + wt * 8 + 1] = (ValueT) (int) global_runs_before_warp_tile;
    d_aggregates[300 + wt * 8 + 2] = (ValueT) runs_before_warp_tile;
    d_aggregates[300 + wt * 8 + 3] = (ValueT) (int) s_prefix;
    d_aggregates[300 + wt * 8 + 4] = (ValueT) wt_end;
    d_aggregates[300 + wt * 8 + 5] = (ValueT) (int) ((size_t) staged_vals & 15);
    d_aggregates[300 + wt * 8 + 6] = (ValueT) (int) my_word;
    d_aggregates[300 + wt * 8 + 7] = (ValueT) tile_len;
  }
  return;
#  endif
  // unique keys straight from staged smem, row-major (coalesced ascending slots per row).
  // LAW (the bug-3 receipt): warp collectives NEVER go inside divergent branches -- the rank
  // shuffle runs unconditionally on all lanes; only the store is predicated.
  {
    const int my_popc = __popc(my_word);
    typename WarpScan<int>::TempStorage kws;
    int word_scan;
    WarpScan<int>(kws).InclusiveSum(my_popc, word_scan);
    const int my_runs_before_word = word_scan - my_popc;
    const unsigned below          = (1u << lane_id) - 1u;
    for (int iter = 0; iter < items_per_thread; ++iter)
    {
      const unsigned w = __shfl_sync(full_mask, my_word, iter);
      const int rbw    = __shfl_sync(full_mask, my_runs_before_word, iter); // ALL lanes: converged
      if ((w >> lane_id) & 1u)
      {
        d_unique[global_runs_before_warp_tile + rbw + __popc(w & below)] =
#  ifdef RBK_FUSED_KEYS_GLOBAL
          keys_tile[warp_tile_offset + iter * 32 + lane_id];
#  else
          staged_keys[pad_elems + warp_tile_offset + iter * 32 + lane_id];
#  endif
      }
    }
  }

#  ifdef RBK_FUSED_STOP3
  return; // gate: before values wait/bands
#  endif
  ValueT wt_lead{};
  ValueT wt_tail{};
  wait_parity(&vals_bar, 0);
#  ifdef RBK_FUSED_STOP4
  return; // gate: after values wait
#  endif
  auto emit_bands = [&](const ValueT* tile_vals, auto staged_tag) _CCCL_FORCEINLINE_LAMBDA {
    constexpr bool from_smem = decltype(staged_tag)::value;
    if (warp_tile_run_count == 0)
    {
      wt_lead = warp_span_fold(tile_vals, warp_tile_offset, wt_end, lane_id, reduction_op);
      wt_tail = wt_lead; // head-free: the whole warp tile leads AND trails
    }
    else if (from_smem && warp_tile_run_count < 4)
    {
      // whole-warp span band (restored): a 2-run warp tile paid the full rotate + ~390-inst walk
      // (SASS census); two coalesced span sums cost ~60. All 32 lanes walk each span together.
      const HeadFlagDecodeT dec(my_word, lane_id);
      const RunSpanT lane_run = dec.decode_run(lane_id < warp_tile_run_count ? lane_id : 0);
      for (int run_idx = 0; run_idx + 1 < warp_tile_run_count; ++run_idx)
      {
        const int head = __shfl_sync(full_mask, lane_run.head_pos_in_warp_tile, run_idx);
        const int next = __shfl_sync(full_mask, lane_run.next_head_pos, run_idx);
        const ValueT agg =
          warp_span_fold(tile_vals, warp_tile_offset + head, warp_tile_offset + next, lane_id, reduction_op);
        if (lane_id == 0)
        {
          d_aggregates[global_runs_before_warp_tile + run_idx] = agg;
        }
      }
      const RunSpanT first_run = dec.decode_run(0);
      const RunSpanT last_run  = dec.decode_run(warp_tile_run_count - 1);
      wt_lead                  = warp_span_fold(
        tile_vals, warp_tile_offset, warp_tile_offset + first_run.head_pos_in_warp_tile, lane_id, reduction_op);
      wt_tail =
        warp_span_fold(tile_vals, warp_tile_offset + last_run.head_pos_in_warp_tile, wt_end, lane_id, reduction_op);
    }
    else if (from_smem && sizeof(ValueT) == 4 && items_per_thread == 32 && warp_tile_run_count < stream_threshold)
    {
      // (4-byte values, 32x32 FULL warp tiles only: the quad rotate/loads cast through uint4;
      // partial warp tiles take the flagged row stream)
      if constexpr (sizeof(ValueT) == 4 && items_per_thread == 32)
      {
        chunk_reduce_rotated<items_per_thread>(
          d_aggregates,
          staged_vals,
          my_word,
          global_runs_before_warp_tile,
          warp_tile_offset,
          wt_end,
          lane_id,
          reduction_op,
          wt_lead,
          wt_tail);
      }
    }
    else if (warp_tile_run_count >= stream_threshold || sizeof(ValueT) != 4 || items_per_thread != 32)
    {
      // stream density router (B200 receipts): quad wins to ~half density (seg4 -9.4%), pair
      // from there to ~3/4 (seg2, quad's interior-emission cost +14% there), the adaptive row
      // stream at near-all-heads (seg1)
      int stream_form = 0; // 0 = row
      if constexpr (from_smem && sizeof(ValueT) == 4 && items_per_thread == 32 && ::cuda::std::is_same_v<ValueT, float>
                    && ::cuda::std::is_same_v<ReductionOpT, ::cuda::std::plus<>>)
      {
        // the quad/pair forms are plus<float> codegen (FADD asm, identity algebra); other ops
        // and partial warp tiles take the row stream.
        // source counters (seg4): the run-count distribution tail spilled 7.9% of the kernel
        // into the row path; the pair form is receipted-good through this density
        stream_form = (warp_tile_run_count < warp_tile_size / 2) ? 2
                    : (warp_tile_run_count < (7 * warp_tile_size) / 8)
                      ? 1
                      : 0;
      }
      if (stream_form == 2)
      {
        if constexpr (from_smem && sizeof(ValueT) == 4 && items_per_thread == 32
                      && ::cuda::std::is_same_v<ValueT, float>
                      && ::cuda::std::is_same_v<ReductionOpT, ::cuda::std::plus<>>)
        {
          stream_values_quad<items_per_thread>(
            d_aggregates, tile_vals, my_word, global_runs_before_warp_tile, warp_tile_offset, lane_id, wt_tail);
        }
      }
      else if (stream_form == 1)
      {
        if constexpr (from_smem && sizeof(ValueT) == 4 && items_per_thread == 32
                      && ::cuda::std::is_same_v<ValueT, float>
                      && ::cuda::std::is_same_v<ReductionOpT, ::cuda::std::plus<>>)
        {
          stream_values_paired<items_per_thread>(
            d_aggregates, tile_vals, my_word, global_runs_before_warp_tile, warp_tile_offset, lane_id, wt_tail);
        }
      }
      else
      {
        stream_values_from_flags<items_per_thread, true>(
          d_aggregates,
          tile_vals,
          my_word,
          global_runs_before_warp_tile,
          warp_tile_offset,
          tile_len,
          lane_id,
          reduction_op,
          wt_tail);
      }
      const HeadFlagDecodeT dec(my_word, lane_id);
      const RunSpanT first_run = dec.decode_run(0);
      wt_lead                  = warp_span_fold(
        tile_vals, warp_tile_offset, warp_tile_offset + first_run.head_pos_in_warp_tile, lane_id, reduction_op);
    }
    // (no final else: the hot path is staged-full only; every other shape bailed to the cold
    // outline above)
  };

  if (tile_len == tile_size)
  {
    emit_bands(staged_vals, ::cuda::std::true_type{});
  }
  else
  {
    // the SHORT LAST TILE (the missing if): the band ladder assumes full tiles; route the
    // partial tile through the flagged row stream from staged smem, exactly like the value
    // kernel's tile-level gate
    if (warp_tile_run_count == 0)
    {
      wt_lead = warp_span_fold(staged_vals, warp_tile_offset, wt_end, lane_id, reduction_op);
      wt_tail = wt_lead;
    }
    else
    {
      stream_values_from_flags<items_per_thread, false>(
        d_aggregates,
        staged_vals,
        my_word,
        global_runs_before_warp_tile,
        warp_tile_offset,
        tile_len,
        lane_id,
        reduction_op,
        wt_tail);
      const HeadFlagDecodeT dec(my_word, lane_id);
      const RunSpanT first_run = dec.decode_run(0);
      wt_lead                  = warp_span_fold(
        staged_vals, warp_tile_offset, warp_tile_offset + first_run.head_pos_in_warp_tile, lane_id, reduction_op);
    }
  }
#  ifdef RBK_FUSED_STOP5
  return; // gate: after bands
#  endif
  if (lane_id == 0)
  {
    wt_leads[wt]          = wt_lead;
    wt_tails[wt]          = wt_tail;
    const int wt_nonempty = wt_end > warp_tile_offset;
    wt_tails_has[wt]      = wt_nonempty;
    wt_leads_has[wt]      = (warp_tile_run_count == 0) ? wt_nonempty : (int) !(my_word & 1u);
  }
  __syncthreads();
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
      // element-holding warp tiles are a PREFIX (trailing wts past tile_len are empty), so
      // both folds are contiguous smem ranges: lane i takes the i-th element; cub::WarpReduce
      // handles the valid prefix (order-preserving ascending tree, result on lane 0)
      const int n_wt_nonempty = min(compute_warps, (tile_len - 1) / warp_tile_size + 1);
      const int n_open        = n_wt_nonempty - last_headed; // >= 1 on a live tile
      const int n_lead        = min(first_headed, n_wt_nonempty);
      typename WarpReduce<ValueT>::TempStorage fold_storage;
      ValueT open_agg =
        WarpReduce<ValueT>(fold_storage)
          .Reduce((lane_id < n_open) ? wt_tails[last_headed + lane_id] : ValueT{}, reduction_op, n_open);
      ValueT lead_agg =
        WarpReduce<ValueT>(fold_storage).Reduce((lane_id < n_lead) ? wt_tails[lane_id] : ValueT{}, reduction_op, n_lead);
      if (lane_id == 0)
      {
        int lead_has = n_lead > 0;
        if (any_head && wt_leads_has[first_headed])
        {
          lead_agg = lead_has ? reduction_op(lead_agg, wt_leads[first_headed]) : wt_leads[first_headed];
          lead_has = 1;
        }
        TileValueRecordT<ValueT, OffT> rec;
        rec.open_agg = open_agg; // never empty on a live tile: it always contains its last head
        rec.lead_agg = lead_agg;
        rec.lead_has = lead_has;
        rec.boundary_dst =
          (curr_prefix_run_count > 0 && (any_head || is_last_tile)) ? (OffT) (curr_prefix_run_count - 1) : (OffT) -1;
        rec.boundary_from      = s_last_runs_tile;
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
        // a headed warp tile's own tail always has (it contains its last head)
        ValueT closing = wt_tails[lane_id];
        for (int w2 = lane_id + 1; w2 < next_headed; ++w2)
        {
          if (wt_tails_has[w2])
          {
            closing = reduction_op(closing, wt_tails[w2]); // head-free middles: whole folds
          }
        }
        if (wt_leads_has[next_headed])
        {
          closing = reduction_op(closing, wt_leads[next_headed]);
        }
        d_aggregates[last_run_global_idx] = closing;
      }
      else if (is_last_tile)
      {
        ValueT closing = wt_tails[lane_id];
        for (int w2 = lane_id + 1; w2 < compute_warps; ++w2)
        {
          if (wt_tails_has[w2])
          {
            closing = reduction_op(closing, wt_tails[w2]);
          }
        }
        d_aggregates[last_run_global_idx] = closing;
      }
      // else: open into the next tile -- the cleanup pass closes it
    }
  }
}

#endif // __cccl_ptx_isa >= 920
} // namespace detail::reduce_by_key::lookahead

CUB_NAMESPACE_END
