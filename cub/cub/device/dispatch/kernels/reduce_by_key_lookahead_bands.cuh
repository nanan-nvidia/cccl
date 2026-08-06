#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/device/dispatch/kernels/reduce_by_key_lookahead_helpers.cuh>

CUB_NAMESPACE_BEGIN

namespace detail::reduce_by_key
{
// when there is less than 4 runs in this warptile, we just do warp reduce one by one
template <class ValueT, class ReductionOpT>
_CCCL_DEVICE_API _CCCL_FORCEINLINE ValueT
warp_span_fold(const ValueT* tile_vals, int begin, int end, int lane_id, ReductionOpT op)
{
  const int len = end - begin;
  // this is always odd to avoid bank conflicts
  const int chunk   = (len > 0) ? (((len + 31) / 32) | 1) : 1;
  const int lo      = begin + lane_id * chunk;
  const int hi      = min(lo + chunk, end);
  const int n_valid = (len > 0) ? ((len + chunk - 1) / chunk) : 0;
  ValueT acc{};
  // reduce within each lane
  if (lo < hi)
  {
    acc = tile_vals[lo];
    for (int pos = lo + 1; pos < hi; ++pos)
    {
      acc = op(acc, tile_vals[pos]);
    }
  }
  typename WarpReduce<ValueT>::TempStorage storage;
  return WarpReduce<ValueT>(storage).Reduce(acc, op, n_valid);
}

template <int items_per_thread, int kLaneElems, class ValueT, class OffT, class ReductionOpT>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void stream_band(
  ValueT* __restrict__ d_aggregates, // global output array
  const ValueT* __restrict__ tile_vals, // staged values in smem
  unsigned my_flags, // this lanes head_flag
  OffT global_runs_before_warp_tile,
  int warp_tile_offset,
  int lane_id,
  ReductionOpT op,
  ValueT& lead_out, // partials of lead
  ValueT& tail_out)
{
  static_assert(items_per_thread == 32 && sizeof(ValueT) == 4, "the prototype pins 32x32 warp tiles and 4-byte values");
  static_assert(kLaneElems == 32 || kLaneElems == 4 || kLaneElems == 2 || kLaneElems == 1,
                "unsupported granularity would silently take the one-element arm");
  constexpr int kRounds                 = items_per_thread / kLaneElems;
  ValueT* const out                     = d_aggregates + global_runs_before_warp_tile;
  const unsigned mask_lanes_at_or_below = (lane_id == 31) ? 0xffffffffu : ((2u << lane_id) - 1);

  if constexpr (kLaneElems == 32)
  {
    // rotate row r's eight 16B quads by (r & 7) IN PLACE so the per-lane column walk below is
    // bank-conflict-free (quads because a 128-bit shared access executes in 4 phases of 8 lanes)
    ValueT* const wt_base = const_cast<ValueT*>(tile_vals) + warp_tile_offset;
#pragma unroll
    for (int rr = 0; rr < 8; ++rr)
    {
      const int row = rr * 4 + (lane_id >> 3); // 4 rows per iteration, 8 lanes each
      const int q   = lane_id & 7;
      ValueT v4[4];
      *(uint4*) v4 = *(const uint4*) (wt_base + row * 32 + q * 4);
      __syncwarp(); // the whole warp's reads retire before any lane's rotated write lands
      *(uint4*) (wt_base + row * 32 + (((q + row) & 7) * 4)) = *(const uint4*) v4;
    }
    __syncwarp();
  }

  // between rounds...
  ValueT carry{}; // fold since the last head seen so far
  bool carry_has     = false; // empty until the first round completes
  bool lead_exported = false; // have we closed the run from prev round
  int run_base       = 0; // runs starting in rounds already finished
  // when there are many rounds, we do not fully unroll to save instr cache & reg pressure
  constexpr int kUnroll =
    (kLaneElems == 32)  ? 1 //  1 round  -> whole thing
    : (kLaneElems == 4) ? 8 //  8 rounds -> full
    : (kLaneElems == 2)
      ? 4 // 16 rounds -> 4-deep
      : 8; // 32 rounds -> 8-deep
#pragma unroll(kUnroll)
  for (int round = 0; round < kRounds; ++round)
  {
    // ---- my kLaneElems head bits + the rank of my first owned run ----
    unsigned lane_heads;
    int rank_base;
    [[maybe_unused]] unsigned round_word = 0; // <1> emission: my whole round word
    int round_runs                       = 0; // runs starting in this round (advances run_base at round end)
    // at which output slot does my first run live? (rank base)
    if constexpr (kLaneElems == 32)
    {
      lane_heads        = my_flags;
      const int my_popc = __popc(my_flags);
      typename WarpScan<int>::TempStorage warp_scan_storage;
      int rank_scan;
      WarpScan<int>(warp_scan_storage).InclusiveSum(my_popc, rank_scan);
      rank_base = rank_scan - my_popc;
    }
    else if constexpr (kLaneElems == 4)
    {
      const unsigned w0            = __shfl_sync(full_mask, my_flags, 4 * round);
      const unsigned w1            = __shfl_sync(full_mask, my_flags, 4 * round + 1);
      const unsigned w2            = __shfl_sync(full_mask, my_flags, 4 * round + 2);
      const unsigned w3            = __shfl_sync(full_mask, my_flags, 4 * round + 3);
      const unsigned my_round_word = (lane_id < 8) ? w0 : (lane_id < 16) ? w1 : (lane_id < 24) ? w2 : w3;
      const int bit_position       = (lane_id & 7) * 4;
      lane_heads                   = (my_round_word >> bit_position) & 0xFu;
      rank_base                    = run_base + ((lane_id >= 8) ? __popc(w0) : 0) + ((lane_id >= 16) ? __popc(w1) : 0)
                + ((lane_id >= 24) ? __popc(w2) : 0) + __popc(my_round_word & ((1u << bit_position) - 1u));
      round_runs = __popc(w0) + __popc(w1) + __popc(w2) + __popc(w3);
    }
    else if constexpr (kLaneElems == 2)
    {
      const unsigned w0            = __shfl_sync(full_mask, my_flags, 2 * round);
      const unsigned w1            = __shfl_sync(full_mask, my_flags, 2 * round + 1);
      const unsigned my_round_word = (lane_id < 16) ? w0 : w1;
      const int bit_position       = (lane_id & 15) * 2;
      lane_heads                   = (my_round_word >> bit_position) & 0x3u;
      rank_base  = run_base + ((lane_id < 16) ? 0 : __popc(w0)) + __popc(my_round_word & ((1u << bit_position) - 1u));
      round_runs = __popc(w0) + __popc(w1);
    }
    else
    {
      round_word = __shfl_sync(full_mask, my_flags, round);
      lane_heads = (round_word >> lane_id) & 1u;
      rank_base  = run_base + __popc(round_word & ((1u << lane_id) - 1u));
      round_runs = __popc(round_word);
    }

    // folding the values
    ValueT lane_lead_agg, lane_open_agg;
    int heads_seen;
    if constexpr (kLaneElems == 32)
    {
      const ValueT* const my_row = tile_vals + warp_tile_offset + lane_id * 32;
      ValueT v[4];
      *(uint4*) v   = *(const uint4*) (my_row + ((lane_id & 7) << 2));
      lane_lead_agg = v[0];
      lane_open_agg = v[0];
      heads_seen    = (int) (lane_heads & 1u);
#pragma unroll
      for (int c = 0; c < 8; ++c)
      {
        ValueT vn[4];
        if (c + 1 < 8)
        {
          *(uint4*) vn = *(const uint4*) (my_row + (((c + 1 + lane_id) & 7) << 2));
        }
#pragma unroll
        for (int j = (c == 0) ? 1 : 0; j < 4; ++j)
        {
          const int elem_idx        = 4 * c + j;
          const bool head           = (lane_heads >> elem_idx) & 1u;
          const ValueT new_open_agg = head ? v[j] : op(lane_open_agg, v[j]);
          const ValueT new_lead_agg = (head || heads_seen > 0) ? lane_lead_agg : op(lane_lead_agg, v[j]);
          if (head && heads_seen > 0)
          {
            out[rank_base + heads_seen - 1] = lane_open_agg;
          }
          heads_seen += (int) head;
          lane_open_agg = new_open_agg;
          lane_lead_agg = new_lead_agg;
        }
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
          v[j] = vn[j];
        }
      }
    }
    else
    {
      const int elem0 = warp_tile_offset + round * (32 * kLaneElems) + lane_id * kLaneElems;
      ValueT v[kLaneElems];
      if constexpr (kLaneElems == 4)
      {
        *(uint4*) v = *(const uint4*) (tile_vals + elem0);
      }
      else if constexpr (kLaneElems == 2)
      {
        *(uint2*) v = *(const uint2*) (tile_vals + elem0);
      }
      else
      {
        v[0] = tile_vals[elem0];
      }
      if constexpr (kLaneElems == 2)
      {
        const bool h0     = (lane_heads & 1u) != 0u;
        const bool h1     = (lane_heads & 2u) != 0u;
        const ValueT both = op(v[0], v[1]);
        lane_lead_agg     = h1 ? v[0] : both; // [start, first head): v0 alone iff e1 is the head
        lane_open_agg     = h1 ? v[1] : both; // since my last head (whole pair when headless)
        heads_seen        = (int) h0 + (int) h1;
        if (h0 && h1)
        {
          out[rank_base] = v[0]; // the run [e0, e1) lives entirely inside my pair
        }
      }
      else
      {
        lane_lead_agg = v[0];
        lane_open_agg = v[0];
        heads_seen    = (int) (lane_heads & 1u);
#pragma unroll
        for (int j = 1; j < kLaneElems; ++j)
        {
          const bool head           = (lane_heads >> j) & 1u;
          const ValueT new_open_agg = head ? v[j] : op(lane_open_agg, v[j]);
          const ValueT new_lead_agg = (head || heads_seen > 0) ? lane_lead_agg : op(lane_lead_agg, v[j]);
          if (head && heads_seen > 0)
          {
            out[rank_base + heads_seen - 1] = lane_open_agg;
          }
          heads_seen += (int) head;
          lane_open_agg = new_open_agg;
          lane_lead_agg = new_lead_agg;
        }
      }
    }
    // <2>'s closed form already leaves lane_open_agg as the tail in every case (incl. headless)
    const ValueT lane_tail_agg = (kLaneElems == 2) ? lane_open_agg : ((heads_seen > 0) ? lane_open_agg : lane_lead_agg);
    const bool has_head        = lane_heads != 0u;
    const int lead_has         = !(lane_heads & 1u); // my pre-first-head piece is empty iff element 0 is a head

    // ---- stitch: ONE masked scan over lane tails, depth bounded by the head distance ----
    ValueT open_agg = lane_tail_agg;
    // at one element per lane the round word IS the head-lane mask: no ballot needed
    const unsigned head_lanes             = (kLaneElems == 1) ? round_word : __ballot_sync(full_mask, has_head);
    const unsigned head_lanes_at_or_below = head_lanes & mask_lanes_at_or_below;
    const int head_distance =
      (head_lanes_at_or_below != 0u) ? (lane_id - (31 - __clz(head_lanes_at_or_below))) : (lane_id + 1);
    const int max_head_distance = __reduce_max_sync(full_mask, head_distance);
    auto scan_steps             = [&](int first_off, int last_off) _CCCL_FORCEINLINE_LAMBDA {
#pragma unroll
      for (int off = 1; off < 32; off <<= 1)
      {
        if (off >= first_off && off <= last_off)
        {
          const unsigned mask_lanes_at_or_below_src = (lane_id >= off) ? ((2u << (lane_id - off)) - 1) : 0u;
          const ValueT from_left                    = __shfl_up_sync(full_mask, open_agg, off);
          const unsigned blockers =
            (head_lanes & mask_lanes_at_or_below & ~mask_lanes_at_or_below_src) | ((lane_id < off) ? 1u : 0u);
          if (blockers == 0)
          {
            open_agg = op(from_left, open_agg);
          }
        }
      }
    };
    if (max_head_distance == 0)
    {
    }
    else if (max_head_distance <= 1)
    {
      scan_steps(1, 1);
    }
    else if (max_head_distance <= 3)
    {
      scan_steps(1, 2);
    }
    else if (max_head_distance <= 7)
    {
      scan_steps(1, 4);
    }
    else
    {
      scan_steps(1, 16);
    }

    //close the runs this round finishes; export the warp tile's lead at its first head
    if constexpr (kLaneElems != 1)
    {
      const ValueT prev_open_agg          = __shfl_up_sync(full_mask, open_agg, 1);
      const unsigned heads_strictly_below = head_lanes & ((1u << lane_id) - 1u);
      const ValueT incoming_agg =
        (kLaneElems == 2)
          ? ((lane_id > 0) ? ((heads_strictly_below != 0u) ? prev_open_agg : op(carry, prev_open_agg)) : carry)
          : ((lane_id > 0) ? ((heads_strictly_below != 0u || !carry_has) ? prev_open_agg : op(carry, prev_open_agg))
                           : carry);
      const bool close_incoming     = has_head && (rank_base >= 1);
      const ValueT incoming_run_agg = lead_has ? op(incoming_agg, lane_lead_agg) : incoming_agg;
      if (close_incoming)
      {
        out[rank_base - 1] = incoming_run_agg;
      }
      if constexpr (kLaneElems >= 4)
        if (!lead_exported && head_lanes != 0u)
        {
          const int first_head_lane = __ffs(head_lanes) - 1;
          const ValueT lead_val     = (lane_id > 0 || carry_has) ? incoming_run_agg : lane_lead_agg;
          lead_out                  = __shfl_sync(full_mask, lead_val, first_head_lane);
          lead_exported             = true;
        }
      }
    }
    else
    {

      if ((round_word & mask_lanes_at_or_below) == 0u && round > 0 && carry_has)
      {
        open_agg = op(carry, open_agg);
      }
      const unsigned next_word = __shfl_sync(full_mask, my_flags, (round + 1) & 31);
      const bool is_end        = (lane_id < 31) ? (((round_word >> (lane_id + 1)) & 1u) != 0u)
                                                : ((round + 1 < kRounds) && ((next_word & 1u) != 0u));
      const int my_run_rank    = run_base + __popc(round_word & mask_lanes_at_or_below) - 1;
      if (is_end && my_run_rank >= 0)
      {
        out[my_run_rank] = open_agg;
      }
    }

    // ---- carry across rounds ----
    const ValueT round_open_agg = __shfl_sync(full_mask, open_agg, 31);
    if constexpr (kLaneElems == 1)
    {
      // the merge above already chained the carry into open_agg: lane 31 holds the complete open fold
      carry     = round_open_agg;
      carry_has = true;
    }
    else if (head_lanes == 0u)
    {
      carry     = carry_has ? op(carry, round_open_agg) : round_open_agg;
      carry_has = true;
    }
    else
    {
      carry     = round_open_agg;
      carry_has = true;
    }
    run_base += round_runs;
  }
  tail_out = carry; // fold since the warp tile's last head (whole tile when head-free)
}
} // namespace detail::reduce_by_key

CUB_NAMESPACE_END
