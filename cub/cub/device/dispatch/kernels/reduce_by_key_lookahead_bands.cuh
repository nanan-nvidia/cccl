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
  const int len     = end - begin;
  const int chunk   = (len > 0) ? (((len + 31) / 32) | 1) : 1;
  const int lo      = begin + lane_id * chunk;
  const int hi      = min(lo + chunk, end);
  const int n_valid = (len > 0) ? ((len + chunk - 1) / chunk) : 0;
  ValueT acc{};
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

// ONE element-directed band; granularity is the only parameter. Each lane owns kLaneElems
// consecutive elements per round (rounds tile the warp tile): a seeded two-ternary walk closes
// lane-interior runs at out[rank_base + hs - 1]; ONE masked scan (depth bounded by the measured
// head distance) stitches lane boundaries; each cross-boundary run closes at out[rank_base - 1];
// a carry chains rounds; the fold since the warp tile's last head returns as tail_out (the
// caller folds the lead span itself). <32> is the register walk (rows rotate in place first),
// <4>/<2> the compressed streams, <1> the row form; <1, false> additionally tracks element
// validity for the input's partial last tile.
template <int items_per_thread, int kLaneElems, bool kFullTile, class ValueT, class OffT, class ReductionOpT>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void stream_band(
  ValueT* __restrict__ d_aggregates,
  const ValueT* __restrict__ tile_vals,
  unsigned my_word,
  OffT global_runs_before_warp_tile,
  int warp_tile_offset,
  int tile_len,
  int lane_id,
  ReductionOpT op,
  ValueT& lead_out, // fold of [warp-tile start, first head); garbage when the lead span is
                    // empty or the warp tile is head-free (the caller's has-flags guard both)
  ValueT& tail_out)
{
  static_assert(items_per_thread <= 32, "one flag word per lane bounds the warp tile");
  static_assert(kLaneElems == 32 || kLaneElems == 4 || kLaneElems == 2 || kLaneElems == 1, "");
  static_assert(kLaneElems == 1 || items_per_thread == 32, "the multi-element forms assume 32x32 warp tiles");
  static_assert(kFullTile || kLaneElems == 1, "partial warp tiles take the one-element form");
  static_assert(kLaneElems == 1 || sizeof(ValueT) == 4, "the vector loads cast through uint4/uint2");
  constexpr int kRounds = items_per_thread / kLaneElems;
  ValueT* const out     = d_aggregates + global_runs_before_warp_tile;
  const unsigned upto_l = (lane_id == 31) ? 0xffffffffu : ((2u << lane_id) - 1);

  if constexpr (kLaneElems == 32)
  {
    // rotate row r's eight 16B quads by (r & 7) IN PLACE so the per-lane column walk below is
    // bank-conflict-free (quads because a 128-bit shared access executes in 4 phases of 8
    // lanes: each phase covers all 32 banks exactly once, both read and write side). <32> only
    // ever runs on the staged smem tile, hence the const_cast.
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

  ValueT carry{}; // fold since the last head seen so far (exact, left-to-right)
  bool carry_has     = false; // empty until a round with valid elements completes
  bool tile_had_head = false; // flips at the tile's first headed round (the lead exports there)
  int word_base      = 0; // heads in words before the current round (uniform)
  // per-granularity unroll (the receipted shapes): full at <32>/<4>, 4-deep at <2>, rolled in
  // fours at <1> -- full unroll of 32 rounds tripled the kernel (SASS gate receipt)
  constexpr int kUnroll = (kLaneElems >= 4) ? kRounds : 4;
#pragma unroll(kUnroll)
  for (int it = 0; it < kRounds; ++it)
  {
    // ---- my kLaneElems head bits + the rank of my first owned run ----
    unsigned nib;
    int rank_base;
    [[maybe_unused]] unsigned round_word   = 0; // <1> emission: my whole round word
    [[maybe_unused]] int round_runs_before = 0; // <1> emission: runs before this round
    if constexpr (kLaneElems == 32)
    {
      nib               = my_word;
      const int my_popc = __popc(my_word);
      typename WarpScan<int>::TempStorage warp_scan_storage;
      int rank_scan;
      WarpScan<int>(warp_scan_storage).InclusiveSum(my_popc, rank_scan);
      rank_base = rank_scan - my_popc;
    }
    else if constexpr (kLaneElems == 4)
    {
      const unsigned w0   = __shfl_sync(full_mask, my_word, 4 * it);
      const unsigned w1   = __shfl_sync(full_mask, my_word, 4 * it + 1);
      const unsigned w2   = __shfl_sync(full_mask, my_word, 4 * it + 2);
      const unsigned w3   = __shfl_sync(full_mask, my_word, 4 * it + 3);
      const unsigned wsel = (lane_id < 8) ? w0 : (lane_id < 16) ? w1 : (lane_id < 24) ? w2 : w3;
      const int bofs      = (lane_id & 7) * 4;
      nib                 = (wsel >> bofs) & 0xFu;
      rank_base           = word_base + ((lane_id >= 8) ? __popc(w0) : 0) + ((lane_id >= 16) ? __popc(w1) : 0)
                + ((lane_id >= 24) ? __popc(w2) : 0) + __popc(wsel & ((1u << bofs) - 1u));
      word_base += __popc(w0) + __popc(w1) + __popc(w2) + __popc(w3);
    }
    else if constexpr (kLaneElems == 2)
    {
      const unsigned w0   = __shfl_sync(full_mask, my_word, 2 * it);
      const unsigned w1   = __shfl_sync(full_mask, my_word, 2 * it + 1);
      const unsigned wsel = (lane_id < 16) ? w0 : w1;
      const int bofs      = (lane_id & 15) * 2;
      nib                 = (wsel >> bofs) & 0x3u;
      rank_base           = word_base + ((lane_id < 16) ? 0 : __popc(w0)) + __popc(wsel & ((1u << bofs) - 1u));
      word_base += __popc(w0) + __popc(w1);
    }
    else
    {
      const unsigned w  = __shfl_sync(full_mask, my_word, it);
      nib               = (w >> lane_id) & 1u;
      round_word        = w;
      round_runs_before = word_base;
      rank_base         = word_base + __popc(w & ((1u << lane_id) - 1u));
      word_base += __popc(w);
    }

    // ---- local walk over my elements: seed at element 0, two-ternary steps after ----
    ValueT pfx, ssh;
    int hs;
    int has = 1; // element validity (partial form only; constant elsewhere)
    if constexpr (kLaneElems == 32)
    {
      const ValueT* const my_row = tile_vals + warp_tile_offset + lane_id * 32;
      ValueT v[4];
      *(uint4*) v = *(const uint4*) (my_row + ((lane_id & 7) << 2));
      pfx         = v[0];
      ssh         = v[0];
      hs          = (int) (nib & 1u);
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
          const int e          = 4 * c + j;
          const bool head      = (nib >> e) & 1u;
          const ValueT new_ssh = head ? v[j] : op(ssh, v[j]);
          const ValueT new_pfx = (head || hs > 0) ? pfx : op(pfx, v[j]);
          if (head && hs > 0)
          {
            out[rank_base + hs - 1] = ssh;
          }
          hs += (int) head;
          ssh = new_ssh;
          pfx = new_pfx;
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
      const int elem0 = warp_tile_offset + it * (32 * kLaneElems) + lane_id * kLaneElems;
      ValueT v[kLaneElems];
      if constexpr (kLaneElems == 4)
      {
        *(uint4*) v = *(const uint4*) (tile_vals + elem0);
      }
      else if constexpr (kLaneElems == 2)
      {
        *(uint2*) v = *(const uint2*) (tile_vals + elem0);
      }
      else if constexpr (!kFullTile)
      {
        has  = (elem0 < tile_len) ? 1 : 0;
        v[0] = has ? tile_vals[elem0] : ValueT{};
      }
      else
      {
        v[0] = tile_vals[elem0];
      }
      pfx = v[0];
      ssh = v[0];
      hs  = (int) (nib & 1u);
#pragma unroll
      for (int j = 1; j < kLaneElems; ++j)
      {
        const bool head      = (nib >> j) & 1u;
        const ValueT new_ssh = head ? v[j] : op(ssh, v[j]);
        const ValueT new_pfx = (head || hs > 0) ? pfx : op(pfx, v[j]);
        if (head && hs > 0)
        {
          out[rank_base + hs - 1] = ssh;
        }
        hs += (int) head;
        ssh = new_ssh;
        pfx = new_pfx;
      }
    }
    const ValueT tail = (hs > 0) ? ssh : pfx; // since my last head (whole span when headless)
    const bool brk    = nib != 0u;
    const int pfx_has = !(nib & 1u); // my pre-first-head piece is empty iff element 0 is a head

    // ---- stitch: ONE masked scan over lane tails, depth bounded by the head distance ----
    ValueT S                      = tail;
    const unsigned pmask          = __ballot_sync(full_mask, brk);
    const unsigned p_at_or_before = pmask & upto_l;
    const int p_dist              = (p_at_or_before != 0u) ? (lane_id - (31 - __clz(p_at_or_before))) : (lane_id + 1);
    const int p_max               = __reduce_max_sync(full_mask, p_dist);
    auto scan_steps               = [&](int first_off, int last_off) _CCCL_FORCEINLINE_LAMBDA {
#pragma unroll
      for (int off = 1; off < 32; off <<= 1)
      {
        if (off >= first_off && off <= last_off)
        {
          const unsigned upto_prev = (lane_id >= off) ? ((2u << (lane_id - off)) - 1) : 0u;
          const ValueT from_left   = shfl_up_sync_wide(S, off);
          int from_has             = 1;
          if constexpr (!kFullTile)
          {
            from_has = __shfl_up_sync(full_mask, has, off);
          }
          // one merged condition, one predicable body, every type and op
          const unsigned t = (pmask & upto_l & ~upto_prev) | ((lane_id < off) ? 1u : 0u) | (from_has ? 0u : 1u);
          if (t == 0)
          {
            S   = has ? op(from_left, S) : from_left;
            has = 1;
          }
        }
      }
    };
    if (p_max == 0)
    {
    }
    else if (p_max <= 1)
    {
      scan_steps(1, 1);
    }
    else if (p_max <= 3)
    {
      scan_steps(1, 2);
    }
    else if (p_max <= 7)
    {
      scan_steps(1, 4);
    }
    else
    {
      scan_steps(1, 16);
    }

    // ---- close the runs this round finishes; export the warp tile's lead at its first head ----
    if constexpr (kLaneElems != 1)
    {
      // deferred incoming close: the run ending at my first head. in_emit (a head exists
      // before this lane's span) guarantees every piece below is nonempty
      const ValueT S_prev   = shfl_up_sync_wide(S, 1);
      const unsigned before = pmask & ((1u << lane_id) - 1u);
      const ValueT P_in     = (lane_id > 0) ? ((before != 0u || !carry_has) ? S_prev : op(carry, S_prev)) : carry;
      const bool in_emit    = brk && (rank_base >= 1);
      const ValueT in_val   = pfx_has ? op(P_in, pfx) : P_in;
      if (in_emit)
      {
        out[rank_base - 1] = in_val;
      }
      if (!tile_had_head && pmask != 0u) // uniform branch: the tile's first headed round
      {
        const int fl          = __ffs(pmask) - 1;
        const ValueT lead_cnd = (lane_id > 0 || carry_has) ? in_val : pfx;
        lead_out              = shfl_sync_wide(lead_cnd, fl);
        tile_had_head         = true;
      }
    }
    else
    {
      // is_end form (a head's pre-first-head piece is empty at this granularity by
      // construction): the lane BEFORE a head stores its own fold, so boundary info travels
      // as a bit, never as a shuffled value (the row stream's receipted emission shape).
      // Lanes with no head at-or-before fold the inter-round carry in FIRST: their open run
      // began in an earlier round, and the store below must carry its whole left part
      if ((round_word & upto_l) == 0u && it > 0 && carry_has)
      {
        S   = has ? op(carry, S) : carry;
        has = 1;
      }
      const unsigned next_word = __shfl_sync(full_mask, my_word, (it + 1) & 31);
      const bool is_end        = (lane_id < 31) ? (((round_word >> (lane_id + 1)) & 1u) != 0u)
                                                : ((it + 1 < kRounds) && ((next_word & 1u) != 0u));
      const int rank_in_round  = round_runs_before + __popc(round_word & upto_l) - 1;
      if (is_end && rank_in_round >= 0)
      {
        out[rank_in_round] = S;
      }
      if (!tile_had_head && pmask != 0u) // uniform branch: the tile's first headed round
      {
        // post-merge, S_prev already contains the inter-round carry chain
        const ValueT S_prev   = shfl_up_sync_wide(S, 1);
        const int fl          = __ffs(pmask) - 1;
        const ValueT lead_cnd = (lane_id > 0) ? S_prev : carry;
        lead_out              = shfl_sync_wide(lead_cnd, fl);
        tile_had_head         = true;
      }
    }

    // ---- carry across rounds ----
    const ValueT S31 = shfl_sync_wide(S, 31);
    int S31_has      = 1;
    if constexpr (!kFullTile)
    {
      S31_has = __shfl_sync(full_mask, has, 31);
    }
    if constexpr (kLaneElems == 1)
    {
      // the merge above already chained the carry into S: lane 31 holds the complete open fold
      if (S31_has)
      {
        carry     = S31;
        carry_has = true;
      }
    }
    else if (pmask == 0u)
    {
      if (S31_has)
      {
        carry     = carry_has ? op(carry, S31) : S31;
        carry_has = true;
      }
    }
    else
    {
      carry     = S31;
      carry_has = S31_has != 0;
    }
  }
  tail_out = carry; // fold since the warp tile's last head (whole tile when head-free)
}
} // namespace detail::reduce_by_key

CUB_NAMESPACE_END
