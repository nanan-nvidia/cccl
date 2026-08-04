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

#include <cub/device/dispatch/kernels/reduce_by_key_lookahead_helpers.cuh>

CUB_NAMESPACE_BEGIN

namespace detail::reduce_by_key::lookahead
{
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

// ROTATED REGISTER-WALK band: the branchless chunk walk with its one receipted flaw removed.
// Lane-contiguous chunks (chunk l == row l) bank-conflict 32-way on every load; rotating each
// row r IN PLACE by r makes the walk's column-step banks (e+l) mod 32 -- conflict-free with ZERO
// extra smem, no shuffles, no padding. The walk itself stays pure FADD/FSEL + one predicated
// store (SASS receipted branchless). MIO per warp tile ~= 96 plain smem ops; no collectives.
template <int items_per_thread, class ValueT, class OffT, class ReductionOpT>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void chunk_reduce_rotated(
  ValueT* __restrict__ d_aggregates,
  ValueT* __restrict__ smem_vals, // the staged tile; rows get ROTATED in place
  unsigned my_word,
  OffT global_runs_before_warp_tile,
  int warp_tile_offset,
  int wt_end,
  int lane_id,
  ReductionOpT op,
  ValueT& lead_out,
  ValueT& tail_out)
{
  static_assert(items_per_thread == 32, "the rotated walk assumes 32x32 warp tiles");
  ValueT* const wt_base = smem_vals + warp_tile_offset;
  // QUAD-granularity rotation (SASS census: the scalar form spent 222 integer ops/wt on rotated
  // addressing + bit tests): rotate row r's eight 16B quads by (r & 7). Eight LDS.128/STS.128
  // pairs replace 32 scalar pairs; the walk then loads four elements per LDS.128 at quad index
  // (c + lane) & 7 -- 4-phase bank-optimal both ways.
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
  const int my_popc = __popc(my_word);
  typename WarpScan<int>::TempStorage warp_scan_storage;
  int rank_scan;
  WarpScan<int>(warp_scan_storage).InclusiveSum(my_popc, rank_scan);
  const OffT gbase = global_runs_before_warp_tile + (rank_scan - my_popc);
  const int valid  = min(max(wt_end - (warp_tile_offset + lane_id * 32), 0), 32);
  ValueT prefix_sum{};
  ValueT sum_since_head{};
  bool seen = false; // any head so far: the COUNT is dead since the indexed
                     // store left (bool drops 2 ISETPs per element)
  const ValueT* const my_row = wt_base + lane_id * 32;
  ValueT* const out          = d_aggregates + gbase;
  // element 0 SEEDS both accumulators outside the loop (no identity element is ever combined
  // in); the e >= 1 step keeps the receipted two-ternary branchless body EXACTLY (an e == 0
  // check inside the step degenerated the predicated store: ncu receipt +44M instructions on
  // the emission line alone)
  // RUNNING output pointer (A/B receipt: the r78 indexed form remats the base 62x from the
  // const bank under today's pressure; the running pointer keeps the region at 925 SASS with
  // zero LDC)
  ValueT* run_out = out;
  auto step       = [&](int e, ValueT v) _CCCL_FORCEINLINE_LAMBDA {
    const bool head      = (my_word >> e) & 1u;
    const bool emit      = head && seen;
    const ValueT new_ssh = head ? v : op(sum_since_head, v);
    const ValueT new_pfx = (head || seen) ? prefix_sum : op(prefix_sum, v);
    if (emit)
    {
      *run_out = sum_since_head;
    }
    run_out += (int) emit;
    seen           = seen || head;
    sum_since_head = new_ssh;
    prefix_sum     = new_pfx;
  };
  auto seed = [&](ValueT v0) _CCCL_FORCEINLINE_LAMBDA {
    sum_since_head = v0;
    prefix_sum     = v0; // never read when element 0 is a head (pfx_has = !(my_word & 1))
    seen           = (my_word & 1u) != 0u;
  };
  // the hot path is full warp tiles only (partial tiles bail to the cold outline before the
  // walk); every lane owns a full 32-element chunk
  {
    ValueT v[4];
    *(uint4*) v = *(const uint4*) (my_row + ((lane_id & 7) << 2));
    seed(v[0]);
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
        step(4 * c + j, v[j]);
      }
#pragma unroll
      for (int j = 0; j < 4; ++j)
      {
        v[j] = vn[j];
      }
    }
  }
  const bool has_head   = seen;
  const ValueT t        = has_head ? sum_since_head : prefix_sum;
  const unsigned bmask  = __ballot_sync(full_mask, has_head);
  const unsigned upto_l = (lane_id == 31) ? 0xffffffffu : ((2u << lane_id) - 1);
  ValueT sseg           = t;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1)
  {
    const unsigned upto_prev = (lane_id >= off) ? ((2u << (lane_id - off)) - 1) : 0u;
    const ValueT from_left   = shfl_up_sync_wide(sseg, off);
    if (lane_id >= off && ((bmask & upto_l & ~upto_prev) == 0u))
    {
      sseg = op(from_left, sseg);
    }
  }
  const ValueT s_prev = shfl_up_sync_wide(sseg, 1);
  // this lane's chunk-lead (prefix_sum) is empty exactly when its element 0 is a head: the
  // pfx-has flag is a bit test, not a tracked register
  const int pfx_has = !(my_word & 1u);
  ValueT incoming{};
  if (has_head)
  {
    incoming = (lane_id > 0) ? (pfx_has ? op(s_prev, prefix_sum) : s_prev) : prefix_sum;
    if ((bmask & ((1u << lane_id) - 1)) != 0)
    {
      d_aggregates[gbase - 1] = incoming;
    }
  }
  const int first_head_lane = __ffs(bmask) - 1; // caller guarantees bmask != 0
  lead_out                  = shfl_sync_wide(incoming, first_head_lane);
  tail_out                  = shfl_sync_wide(sseg, 31);
}

// TWO-LEVEL SEGMENTED SCAN band (the receipted design): pass A runs an independent 5-step
// masked scan per ROW (row-major = conflict-free; NO cross-row carry, so the shuffle chains of
// consecutive rows overlap and the pass is issue-bound), writing local prefixes back over the
// staged values IN PLACE and collecting row totals one-per-lane. Pass B is ONE masked scan over
// the 32 row totals (segment breaks at rows containing heads) = each row's incoming carry.
// Emission is pure lookups, R-proportional: the run ending at position p is
// S_local[p] + (carry[row] iff p precedes the row's first head). Left-to-right numerics.
// The two receipted serializers this replaces: the chunk walk's 32-way bank-conflicted lane
// chunks, and the streaming scan's row-to-row carry chain.
template <int items_per_thread, class ValueT, class OffT>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void scan_lookup_from_flags(
  ValueT* __restrict__ d_aggregates,
  ValueT* smem_vals, // the staged warp tile; OVERWRITTEN with local prefixes
  unsigned my_word,
  OffT global_runs_before_warp_tile,
  int warp_tile_offset,
  int lane_id,
  ValueT* row_carry, // [32] per-warp-tile scratch
  ValueT& lead_out,
  ValueT& tail_out)
{
  static_assert(items_per_thread == 32, "the scan band assumes 32x32 warp tiles");
  const unsigned upto_l = (lane_id == 31) ? 0xffffffffu : ((2u << lane_id) - 1);
  // ---- pass A: independent local row scans, in place; row totals collect one per lane ----
  ValueT my_row_total = ValueT{};
#pragma unroll
  for (int r = 0; r < items_per_thread; ++r)
  {
    const unsigned w = __shfl_sync(full_mask, my_word, r);
    const int loc    = warp_tile_offset + r * 32 + lane_id;
    ValueT incl      = smem_vals[loc];
#pragma unroll
    for (int off = 1; off < 32; off <<= 1)
    {
      const unsigned upto_prev = (lane_id >= off) ? ((2u << (lane_id - off)) - 1) : 0u;
      const ValueT from_left   = __shfl_up_sync(full_mask, incl, off);
      if (lane_id >= off && ((w & upto_l & ~upto_prev) == 0u))
      {
        incl += from_left;
      }
    }
    smem_vals[loc]         = incl; // local prefix (no incoming carry yet)
    const ValueT row_total = __shfl_sync(full_mask, incl, 31); // sum since the row's last head
    if (lane_id == r)
    {
      my_row_total = row_total;
    }
  }
  // ---- pass B: one masked scan across row totals; break at rows that contain a head ----
  const bool row_has_head = (my_word != 0);
  const unsigned rmask    = __ballot_sync(full_mask, row_has_head);
  ValueT rs               = my_row_total;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1)
  {
    const unsigned upto_prev = (lane_id >= off) ? ((2u << (lane_id - off)) - 1) : 0u;
    const ValueT from_left   = __shfl_up_sync(full_mask, rs, off);
    if (lane_id >= off && ((rmask & upto_l & ~upto_prev) == 0u))
    {
      rs += from_left;
    }
  }
  // carry INTO row l = rs of row l-1 (sum since the last head at or before row l-1)
  const ValueT my_carry = __shfl_up_sync(full_mask, rs, 1);
  row_carry[lane_id]    = (lane_id > 0) ? my_carry : ValueT{};
  __syncwarp();
  // ---- emission: lane l owns row l's heads; the run ENDING at head bit b lives at p = b-1 ----
  const int my_popc = __popc(my_word);
  typename WarpScan<int>::TempStorage warp_scan_storage;
  int rank_scan;
  WarpScan<int>(warp_scan_storage).InclusiveSum(my_popc, rank_scan);
  const OffT gbase      = global_runs_before_warp_tile + (rank_scan - my_popc);
  const unsigned w_prev = __shfl_up_sync(full_mask, my_word, 1); // row l-1's word (row -1: garbage, unused)
  ValueT* const out     = d_aggregates + gbase;
  auto s_final          = [&](int p) _CCCL_FORCEINLINE_LAMBDA { // p = absolute pos within the warp tile
    const int prow       = p >> 5;
    const int pbit       = p & 31;
    const unsigned pword = (prow == lane_id) ? my_word : w_prev; // emission only touches rows l and l-1
    const bool pre_head  = (pword & ((2u << pbit) - 1)) == 0u; // no head at or before p in its row
    return smem_vals[warp_tile_offset + p] + (pre_head ? row_carry[prow] : ValueT{});
  };
  const int runs_before_lane = rank_scan - my_popc;
  unsigned rem               = my_word;
  int i                      = 0;
  while (rem != 0)
  {
    const int b = __ffs(rem) - 1;
    rem &= rem - 1;
    // my i-th head is the wt's head number (runs_before_lane + i); every head but the wt's FIRST
    // closes the run ranked one before my base (the first head's incoming span is the wt lead)
    if (runs_before_lane + i > 0)
    {
      out[i - 1] = s_final(lane_id * 32 + b - 1);
    }
    ++i;
  }
  // wt lead: the span before the wt's first head, looked up BY the lane owning that row (the
  // s_final word registers are per-lane)
  const unsigned head_rows  = __ballot_sync(full_mask, my_word != 0);
  const int first_head_lane = __ffs(head_rows) - 1; // caller guarantees runs >= 1
  int fh_bit                = (lane_id == first_head_lane) ? (__ffs(my_word) - 1) : 0;
  fh_bit                    = __shfl_sync(full_mask, fh_bit, first_head_lane);
  const int lead_end        = first_head_lane * 32 + fh_bit - 1;
  const int lead_row        = lead_end >> 5;
  ValueT lead_val{};
  if (lead_end >= 0 && lane_id == lead_row)
  {
    const bool pre = (my_word & ((2u << (lead_end & 31)) - 1)) == 0u;
    lead_val       = smem_vals[warp_tile_offset + lead_end] + (pre ? row_carry[lead_row] : ValueT{});
  }
  lead_out = (lead_end >= 0) ? __shfl_sync(full_mask, lead_val, lead_row & 31) : ValueT{};
  // tail: rs at lane 31 = sum since the wt's last head (rows past the last head have no breaks)
  tail_out = __shfl_sync(full_mask, rs, 31);
}

// single-read per-lane chunk band: lane l owns elements [32l, 32l+32) of the warp tile, which
// is EXACTLY flag word l. Loads are 512B/instruction coalesced (LDG.128 per lane); each lane
// closes its interior runs straight from the register walk; ONE masked segmented scan over the
// lane sums closes every cross-lane run and yields the warp tile's lead and tail as byproducts.
// Replaces the whole-warp/wave/per-lane span-walk bands, whose serial latency chains AND
// separate lead/tail re-reads (up to a full extra pass at seg1024) were the mid-regime tax.
template <int items_per_thread, class ValueT, class OffT, class ReductionOpT>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void chunk_reduce_from_flags(
  ValueT* __restrict__ d_aggregates,
  const ValueT* __restrict__ tile_vals,
  unsigned my_word,
  OffT global_runs_before_warp_tile,
  int warp_tile_offset,
  int wt_end,
  int lane_id,
  ReductionOpT op,
  ValueT& lead_out,
  ValueT& tail_out)
{
  static_assert(items_per_thread == 32, "the chunk band assumes 32x32 warp tiles");
  const int chunk_begin = warp_tile_offset + lane_id * 32;
  const int valid       = min(max(wt_end - chunk_begin, 0), 32);
  const int my_popc     = __popc(my_word);
  typename WarpScan<int>::TempStorage warp_scan_storage;
  int rank_scan;
  WarpScan<int>(warp_scan_storage).InclusiveSum(my_popc, rank_scan);
  const OffT gbase = global_runs_before_warp_tile + (rank_scan - my_popc);
  ValueT prefix_sum{}; // sum of my elements before my first head
  ValueT sum_since_head{};
  int heads_seen      = 0;
  const ValueT* chunk = tile_vals + chunk_begin;
  ValueT* const out   = d_aggregates + gbase; // register base: no per-step LDC rematerialization
  // BRANCHLESS walk (SASS receipt: the if/elif/else form compiled to a BSSY/BRA/BSYNC divergent
  // block PER ELEMENT -- ~5700 cycles/warp tile for compute that costs ~300): selects + one
  // predicated store, nothing for lanes to diverge on
  // e == 0 seeds both accumulators: no identity element is ever combined in
  auto step = [&](int e, ValueT v) _CCCL_FORCEINLINE_LAMBDA {
    const bool head      = (my_word >> e) & 1u;
    const ValueT new_ssh = head ? v : ((e == 0) ? v : op(sum_since_head, v));
    const ValueT new_pfx = (head || heads_seen > 0) ? prefix_sum : ((e == 0) ? v : op(prefix_sum, v));
    if (head && heads_seen > 0)
    {
      out[heads_seen - 1] = sum_since_head; // interior run closes in-lane
    }
    heads_seen += (int) head;
    sum_since_head = new_ssh;
    prefix_sum     = new_pfx;
  };
  if (valid == 32 && sizeof(ValueT) == 4 && (((size_t) chunk & 15) == 0))
  {
#pragma unroll
    for (int c = 0; c < 8; ++c)
    {
      ValueT v4[4];
      *(uint4*) v4 = *(const uint4*) (chunk + 4 * c);
#pragma unroll
      for (int j = 0; j < 4; ++j)
      {
        step(4 * c + j, v4[j]);
      }
    }
  }
  else
  {
    for (int e = 0; e < valid; ++e)
    {
      step(e, chunk[e]);
    }
  }
  const bool has_head   = heads_seen > 0;
  const ValueT t        = has_head ? sum_since_head : prefix_sum; // fold since the last head at or before me
  const unsigned bmask  = __ballot_sync(full_mask, has_head);
  const unsigned upto_l = (lane_id == 31) ? 0xffffffffu : ((2u << lane_id) - 1);
  ValueT s              = t;
  int s_has             = valid > 0; // trailing lanes past wt_end hold no elements
#pragma unroll
  for (int off = 1; off < 32; off <<= 1)
  {
    const unsigned upto_prev = (lane_id >= off) ? ((2u << (lane_id - off)) - 1) : 0u;
    const ValueT from_left   = shfl_up_sync_wide(s, off);
    const int from_has       = __shfl_up_sync(full_mask, s_has, off);
    if (lane_id >= off && ((bmask & upto_l & ~upto_prev) == 0u) && from_has)
    {
      s     = s_has ? op(from_left, s) : from_left;
      s_has = 1;
    }
  }
  const ValueT s_prev = shfl_up_sync_wide(s, 1);
  // my chunk-lead (prefix_sum) is empty exactly when my element 0 is a head
  const int pfx_has = !(my_word & 1u);
  ValueT incoming{};
  int incoming_has = 0;
  if (has_head)
  {
    // the run ENDING at my first head: its rank is one before my first owned run. Lanes below a
    // headed lane are never empty (chunks are contiguous), so s_prev is valid whenever lane > 0.
    incoming     = (lane_id > 0) ? (pfx_has ? op(s_prev, prefix_sum) : s_prev) : prefix_sum;
    incoming_has = (lane_id > 0) | pfx_has;
    if ((bmask & ((1u << lane_id) - 1)) != 0)
    {
      d_aggregates[gbase - 1] = incoming;
    }
  }
  const int first_head_lane = __ffs(bmask) - 1; // caller guarantees bmask != 0
  lead_out                  = shfl_sync_wide(incoming, first_head_lane);
  tail_out                  = shfl_sync_wide(s, 31); // fold since the warp tile's last head
}

// values-only streaming reduce over one warp tile, boundaries from REGISTER flag words (the
// value warps snapshot the flags and run fully detached from the rings). Emits every
// within-warp-tile-closed aggregate; the trailing open sum comes back as the carry. The scan
// loop is the FROZEN form -- lead/tail ride separate passes, never inside it.
template <int items_per_thread, bool full_tile, class ValueT, class OffT, class ReductionOpT>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void stream_values_from_flags(
  ValueT* __restrict__ d_aggregates,
  const ValueT* __restrict__ tile_vals,
  unsigned my_word,
  OffT global_runs_before_warp_tile,
  int warp_tile_offset,
  int tile_len,
  int lane_id,
  ReductionOpT op,
  ValueT& tail_out)
{
  const unsigned upto_l = (lane_id == 31) ? 0xffffffffu : ((2u << lane_id) - 1); // bits [0, lane]
  // ROW-LEVEL output pointer (replaces the entry WarpScan + a per-element SHFL+IADD of
  // runs-before-word, and confines any base rematerialization to once per ROW): the row's
  // emission slots are row_out[popc(w & upto_l) - 1]; rank -1 legally reaches the previous
  // row's last slot (a run closing across the row boundary)
  ValueT* row_out     = d_aggregates + global_runs_before_warp_tile;
  int runs_before_row = 0; // scalar: gates the entering-run skip (global run_idx < 0)
  // CHUNKED preload: 8 rows in flight instead of the whole warp tile -- the full-tile buffer put
  // the kernel at 32+ live registers/lane and SPILLED TO LOCAL under high occupancy (B200
  // receipt: VEmit 16us/warp-tile of local-memory thrash). The scan carry crosses chunks freely.
  constexpr int chunk_rows = (items_per_thread < 8) ? items_per_thread : 8;
  ValueT carry{};
  int carry_has = 0;
  for (int chunk = 0; chunk < items_per_thread; chunk += chunk_rows)
  {
    ValueT row_vals[chunk_rows];
#pragma unroll
    for (int cr = 0; cr < chunk_rows; ++cr)
    {
      const int iter = chunk + cr;
      const int loc  = warp_tile_offset + iter * 32 + lane_id;
      if constexpr (full_tile)
      {
        row_vals[cr] = tile_vals[loc]; // staged tiles are zero-filled past tile_len
      }
      else
      {
        row_vals[cr] = (iter < items_per_thread && loc < tile_len) ? tile_vals[loc] : ValueT{};
      }
    }
#pragma unroll
    for (int cr = 0; cr < chunk_rows; ++cr)
    {
      const int iter = chunk + cr;
      if (iter >= items_per_thread)
      {
        break;
      }
      const unsigned w = __shfl_sync(full_mask, my_word, iter);
      ValueT incl      = row_vals[cr];
      // full tiles are all-valid by construction; only the partial variant tracks validity
      int has = full_tile ? 1 : (int) (warp_tile_offset + iter * 32 + lane_id < tile_len);
      // ADAPTIVE scan depth (dense receipt: the fixed 5-step scan is the MIO wall at ~8 ops/row
      // x 256 rows/tile; step k only matters if some accumulation distance reaches 2^(k-1)).
      // Each lane's distance to its nearest head at-or-before covers spans, carry-in and
      // carry-out regions exactly; the warp max bounds the needed steps. Continuous and general.
      const unsigned at_or_before = w & upto_l;
      const int my_dist           = (at_or_before != 0u) ? (lane_id - (31 - __clz(at_or_before))) : (lane_id + 1);
      const int max_dist          = __reduce_max_sync(full_mask, my_dist);
      // three UNIFORM paths on the bound (receipts: per-step skips cost seg1, a dynamic loop
      // cost seg4, the fixed scan cost seg1/2 -- each form won different cells). Every path is
      // unrolled; running extra steps is always mask-safe, so the branch only affects speed.
      auto scan_steps = [&](int first_off, int last_off) _CCCL_FORCEINLINE_LAMBDA {
#pragma unroll
        for (int off = 1; off < 32; off <<= 1)
        {
          if (off >= first_off && off <= last_off)
          {
            const unsigned upto_prev = (lane_id >= off) ? ((2u << (lane_id - off)) - 1) : 0u;
            const ValueT from_left   = shfl_up_sync_wide(incl, off);
            const int from_has       = full_tile ? 1 : __shfl_up_sync(full_mask, has, off);
            if constexpr (full_tile && sizeof(ValueT) == 4 && ::cuda::std::is_same_v<ValueT, float>
                          && ::cuda::std::is_same_v<ReductionOpT, ::cuda::std::plus<>>)
            {
              // force SETP + @p FADD (2 ops) over nvcc's FADD+FSEL+ISETP (3): the guard folds
              // into the test value with one LOP3 -- nonzero test blocks lanes below the step
              const unsigned t = (w & upto_l & ~upto_prev) | ((lane_id < off) ? 1u : 0u);
              asm volatile("{ .reg .pred p; setp.eq.u32 p, %1, 0; @p add.f32 %0, %0, %2; }"
                           : "+f"(incl)
                           : "r"(t), "f"(from_left));
            }
            else
            {
              if (lane_id >= off && ((w & upto_l & ~upto_prev) == 0u) && from_has)
              {
                incl = has ? op(from_left, incl) : from_left;
                has  = 1;
              }
            }
          }
        }
      };
      if (max_dist == 0)
      {
        // every lane is a head: incl is already its own value
      }
      else if (max_dist <= 3)
      {
        scan_steps(1, 2); // reach 3
      }
      else
      {
        scan_steps(1, 16); // the full pipelined scan
      }
      // iter 0 has no earlier row: the carry is empty there (no identity element is combined in)
      if ((w & upto_l) == 0u && iter > 0 && carry_has)
      {
        incl = has ? op(carry, incl) : carry;
        has  = 1;
      }
      const unsigned next_word = __shfl_sync(full_mask, my_word, (iter + 1) & 31);
      const bool is_end        = (lane_id < 31) ? (((w >> (lane_id + 1)) & 1u) != 0u)
                                                : ((iter + 1 < items_per_thread) && ((next_word & 1u) != 0u));
      const int rank_in_row    = __popc(w & upto_l) - 1;
      if (is_end && runs_before_row + rank_in_row >= 0)
      {
        row_out[rank_in_row] = incl;
      }
      const int row_runs = __popc(w);
      row_out += row_runs;
      runs_before_row += row_runs;
      carry     = shfl_sync_wide(incl, 31);
      carry_has = full_tile ? 1 : __shfl_sync(full_mask, has, 31);
    }
  }
  tail_out = carry; // fold since the warp tile's last head (whole tile when head-free)
}

// QUAD-COMPRESSED STREAM (staged full tiles, 4-byte values): each lane owns FOUR consecutive
// elements -- one LDS.128 and one masked scan per 128 elements. Interior runs (start AND end
// inside the quad) emit from the branchless local walk with purely local values; the single
// possible incoming close per quad defers to after the cross-quad scan (P_in), exactly the
// pair form's shape. At short-span densities the quad-distance bound collapses the scan to
// 0-1 steps. Emissions are bare predicated stores off a hoisted base (SASS receipts).
template <int items_per_thread, class ValueT, class OffT>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void stream_values_quad(
  ValueT* __restrict__ d_aggregates,
  const ValueT* __restrict__ tile_vals, // staged smem, zero-filled past tile_len
  unsigned my_word,
  OffT global_runs_before_warp_tile,
  int warp_tile_offset,
  int lane_id,
  ValueT& tail_out)
{
  static_assert(items_per_thread == 32, "the quad stream assumes 32x32 warp tiles");
  ValueT* const out = d_aggregates + global_runs_before_warp_tile;
  ValueT carry{}; // sum since the last head seen so far (exact, left-to-right)
  int word_base = 0; // heads in words before the current 128-element block (uniform)
#pragma unroll
  for (int it = 0; it < items_per_thread / 4; ++it)
  {
    const unsigned w0   = __shfl_sync(full_mask, my_word, 4 * it);
    const unsigned w1   = __shfl_sync(full_mask, my_word, 4 * it + 1);
    const unsigned w2   = __shfl_sync(full_mask, my_word, 4 * it + 2);
    const unsigned w3   = __shfl_sync(full_mask, my_word, 4 * it + 3);
    const unsigned wsel = (lane_id < 8) ? w0 : (lane_id < 16) ? w1 : (lane_id < 24) ? w2 : w3;
    const int bofs      = (lane_id & 7) * 4;
    const unsigned nib  = (wsel >> bofs) & 0xFu;
    ValueT v[4];
    *(uint4*) v = *(const uint4*) (tile_vals + warp_tile_offset + it * 128 + 4 * lane_id);
    // rank base: heads before my quad in the warp tile
    const int wb = word_base + ((lane_id >= 8) ? __popc(w0) : 0) + ((lane_id >= 16) ? __popc(w1) : 0)
                 + ((lane_id >= 24) ? __popc(w2) : 0);
    const int rank_base = wb + __popc(wsel & ((1u << bofs) - 1u));
    // branchless quad-local walk: interior runs emit NOW (local values); the incoming close
    // defers to post-scan
    ValueT pfx{};
    ValueT ssh{};
    int hs = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j)
    {
      const bool head      = (nib >> j) & 1u;
      const ValueT new_ssh = head ? v[j] : (ssh + v[j]);
      const ValueT new_pfx = (head || hs > 0) ? pfx : (pfx + v[j]);
      if (head && hs > 0)
      {
        out[rank_base + hs - 1] = ssh;
      }
      hs += (int) head;
      ssh = new_ssh;
      pfx = new_pfx;
    }
    const bool brk    = nib != 0u;
    const ValueT tail = (hs > 0) ? ssh : pfx; // since the quad's last head (whole quad if headless)
    // masked scan over quad tails, breaks at quads with heads; adaptive depth in QUAD units
    const unsigned pmask          = __ballot_sync(full_mask, brk);
    const unsigned upto_l         = (lane_id == 31) ? 0xffffffffu : ((2u << lane_id) - 1);
    ValueT S                      = tail;
    const unsigned q_at_or_before = pmask & upto_l;
    const int q_dist              = (q_at_or_before != 0u) ? (lane_id - (31 - __clz(q_at_or_before))) : (lane_id + 1);
    const int q_max               = __reduce_max_sync(full_mask, q_dist);
    auto quad_scan_steps          = [&](int first_off, int last_off) _CCCL_FORCEINLINE_LAMBDA {
#pragma unroll
      for (int off = 1; off < 32; off <<= 1)
      {
        if (off >= first_off && off <= last_off)
        {
          const unsigned upto_prev = (lane_id >= off) ? ((2u << (lane_id - off)) - 1) : 0u;
          const ValueT from_left   = __shfl_up_sync(full_mask, S, off);
          if constexpr (sizeof(ValueT) == 4 && ::cuda::std::is_same_v<ValueT, float>)
          {
            const unsigned t = (pmask & upto_l & ~upto_prev) | ((lane_id < off) ? 1u : 0u);
            asm volatile("{ .reg .pred p; setp.eq.u32 p, %1, 0; @p add.f32 %0, %0, %2; }"
                         : "+f"(S)
                         : "r"(t), "f"(from_left));
          }
          else
          {
            if (lane_id >= off && ((pmask & upto_l & ~upto_prev) == 0u))
            {
              S += from_left;
            }
          }
        }
      }
    };
    if (q_max == 0)
    {
    }
    else if (q_max <= 1)
    {
      quad_scan_steps(1, 1);
    }
    else if (q_max <= 3)
    {
      quad_scan_steps(1, 2);
    }
    else
    {
      quad_scan_steps(1, 16);
    }
    // deferred incoming close: everything since the last head strictly before this quad
    const ValueT S_prev   = __shfl_up_sync(full_mask, S, 1);
    const unsigned before = pmask & ((1u << lane_id) - 1u);
    const ValueT P_in     = ((lane_id > 0) ? S_prev : ValueT{}) + ((before == 0u) ? carry : ValueT{});
    const bool in_emit    = brk && (rank_base >= 1);
    const ValueT in_val   = P_in + pfx;
    if (in_emit)
    {
      out[rank_base - 1] = in_val;
    }
    const ValueT S31 = __shfl_sync(full_mask, S, 31);
    carry            = (pmask == 0u) ? (carry + S31) : S31;
    word_base += __popc(w0) + __popc(w1) + __popc(w2) + __popc(w3);
  }
  tail_out = carry;
}

// PAIR-COMPRESSED STREAM (staged full tiles, 4-byte values): each lane owns elements (2l, 2l+1)
// of a 64-element double-row -- one LDS.64, one masked scan per 64 elements instead of two full
// row iterations. Pair algebra: a pair's carry contribution is the sum since its last head (the
// whole pair when headless); breaks at pairs containing heads. Run ends: e0 ends iff h1 (local
// emit when h0 starts it, incoming-close via the scan otherwise); e1 ends iff the NEXT pair
// opens with a head (its incoming-close handles it). Ranks come from word bit positions, no
// per-lane scans. Emission ranks and the carry are exact left-to-right sums.
template <int items_per_thread, class ValueT, class OffT>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void stream_values_paired(
  ValueT* __restrict__ d_aggregates,
  const ValueT* __restrict__ tile_vals, // staged smem, zero-filled past tile_len
  unsigned my_word,
  OffT global_runs_before_warp_tile,
  int warp_tile_offset,
  int lane_id,
  ValueT& tail_out)
{
  static_assert(items_per_thread == 32, "the paired stream assumes 32x32 warp tiles");
  ValueT* const out = d_aggregates + global_runs_before_warp_tile; // register base: no per-
                                                                   // emission LDC rematerialization
  ValueT carry{}; // sum since the last head seen so far (exact, left-to-right)
  int word_base = 0; // heads in words before the current double-row (uniform)
#pragma unroll 4
  for (int it = 0; it < items_per_thread / 2; ++it)
  {
    const unsigned w0 = __shfl_sync(full_mask, my_word, 2 * it);
    const unsigned w1 = __shfl_sync(full_mask, my_word, 2 * it + 1);
    // lane l's flag bits and elements
    const unsigned wsel = (lane_id < 16) ? w0 : w1;
    const int bofs      = (lane_id & 15) * 2;
    const bool h0       = (wsel >> bofs) & 1u;
    const bool h1       = (wsel >> (bofs + 1)) & 1u;
    ValueT v01[2];
    *(uint2*) v01   = *(const uint2*) (tile_vals + warp_tile_offset + it * 64 + 2 * lane_id);
    const ValueT v0 = v01[0];
    const ValueT v1 = v01[1];
    // pair-local pieces
    const ValueT pre  = h0 ? ValueT{} : v0; // before the pair's first head
    const ValueT tail = h1 ? v1 : (h0 ? (v0 + v1) : (v0 + v1)); // since the pair's last head (whole pair if headless)
    const bool brk    = h0 || h1;
    // masked inclusive scan over pair tails, breaks at pairs with heads: S = sum since the last
    // head at-or-before this pair (within this double-row). ADAPTIVE depth (the row-stream
    // receipt, -27% at its densest cell): the max pair-distance to a break bounds the steps any
    // lane needs; three uniform paths, every path unrolled, extra steps always mask-safe.
    const unsigned pmask          = __ballot_sync(full_mask, brk);
    const unsigned upto_l         = (lane_id == 31) ? 0xffffffffu : ((2u << lane_id) - 1);
    ValueT S                      = tail;
    const unsigned p_at_or_before = pmask & upto_l;
    const int p_dist              = (p_at_or_before != 0u) ? (lane_id - (31 - __clz(p_at_or_before))) : (lane_id + 1);
    const int p_max               = __reduce_max_sync(full_mask, p_dist);
    auto pair_scan_steps          = [&](int first_off, int last_off) _CCCL_FORCEINLINE_LAMBDA {
#pragma unroll
      for (int off = 1; off < 32; off <<= 1)
      {
        if (off >= first_off && off <= last_off)
        {
          const unsigned upto_prev = (lane_id >= off) ? ((2u << (lane_id - off)) - 1) : 0u;
          const ValueT from_left   = __shfl_up_sync(full_mask, S, off);
          if constexpr (sizeof(ValueT) == 4 && ::cuda::std::is_same_v<ValueT, float>)
          {
            const unsigned t = (pmask & upto_l & ~upto_prev) | ((lane_id < off) ? 1u : 0u);
            asm volatile("{ .reg .pred p; setp.eq.u32 p, %1, 0; @p add.f32 %0, %0, %2; }"
                         : "+f"(S)
                         : "r"(t), "f"(from_left));
          }
          else
          {
            if (lane_id >= off && ((pmask & upto_l & ~upto_prev) == 0u))
            {
              S += from_left;
            }
          }
        }
      }
    };
    if (p_max == 0)
    {
      // every pair contains a head: S is already each pair's own tail
    }
    else if (p_max <= 3)
    {
      pair_scan_steps(1, 2); // reach 3
    }
    else if (p_max <= 7)
    {
      pair_scan_steps(1, 4); // reach 7 -- seg4's regime (64% pair-break density)
    }
    else
    {
      pair_scan_steps(1, 16);
    }
    // P_in: the exact sum since the last head STRICTLY before this pair (previous pair's scan
    // value, plus the inter-row carry when no break precedes in this double-row)
    const ValueT S_prev   = __shfl_up_sync(full_mask, S, 1);
    const unsigned before = pmask & ((1u << lane_id) - 1u);
    const ValueT P_in     = ((lane_id > 0) ? S_prev : ValueT{}) + ((before == 0u) ? carry : ValueT{});
    // ranks from word bit positions (base = heads before my element's word)
    const int wb      = word_base + ((lane_id < 16) ? 0 : __popc(w0));
    const int rank_e0 = wb + __popc(wsel & ((1u << bofs) - 1u)); // heads before e0 in the wt
    // a head at e0 closes the incoming run (rank_e0 - 1); a head at e1 closes the run ending at
    // e0 (rank_e0 + h0 - 1): fully local when it started at e0, incoming + v0 otherwise.
    // Values and ranks computed unconditionally; the stores are the ONLY conditional ops
    // (bare `if (b) out[i] = v` compiles to @P STG -- the branchless-walk receipt)
    const bool e0_emit  = h0 && (rank_e0 >= 1);
    const int r1        = rank_e0 + (h0 ? 1 : 0) - 1;
    const bool e1_emit  = h1 && (r1 >= 0);
    const ValueT e1_val = h0 ? v0 : (P_in + v0);
    if (e0_emit)
    {
      out[rank_e0 - 1] = P_in;
    }
    if (e1_emit)
    {
      out[r1] = e1_val;
    }
    // carry update: sum since the last head after this double-row = S at lane 31 (+ carry if the
    // row had no breaks at all)
    const ValueT S31 = __shfl_sync(full_mask, S, 31);
    carry            = (pmask == 0u) ? (carry + S31) : S31;
    word_base += __popc(w0) + __popc(w1);
  }
  tail_out = carry;
}

// warp-parallel strided sum over [begin, end) of the tile's values (lead/tail segments and the
// long-span cooperative walks; every lane participates)
// order-preserving warp fold of [begin, end): lane l folds a CONTIGUOUS chunk, then a
// lane-ascending shfl_down tree combines the chunks. The chunk stride is forced ODD so the
// per-lane smem walks stay bank-conflict-free (a 32-multiple stride lands every lane on one
// bank). No identity element is assumed: empty chunks carry a has-flag. Every combine is
// op(earlier indices, later indices) -- associativity is the only requirement.
template <class ValueT, class ReductionOpT>
_CCCL_DEVICE_API _CCCL_FORCEINLINE ValueT
warp_span_fold(const ValueT* tile_vals, int begin, int end, int lane_id, ReductionOpT op)
{
  const int len     = end - begin;
  const int chunk   = (len > 0) ? (((len + 31) / 32) | 1) : 1; // ODD stride: conflict-free lane walks
  const int lo      = begin + lane_id * chunk;
  const int hi      = min(lo + chunk, end);
  const int n_valid = (len > 0) ? ((len + chunk - 1) / chunk) : 0; // nonempty chunks are a PREFIX
  ValueT acc{};
  if (lo < hi)
  {
    acc = tile_vals[lo]; // the first element seeds: no identity element exists for a general op
    for (int pos = lo + 1; pos < hi; ++pos)
    {
      acc = op(acc, tile_vals[pos]);
    }
  }
  // cub::WarpReduce: ascending shfl_down tree over the first n_valid lanes, associativity-only
  // (order-preserving; lane-ascending = element-ascending). Result is valid on lane 0.
  typename WarpReduce<ValueT>::TempStorage storage;
  return WarpReduce<ValueT>(storage).Reduce(acc, op, n_valid);
}

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
#pragma unroll
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
#pragma unroll
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
} // namespace detail::reduce_by_key::lookahead

CUB_NAMESPACE_END
