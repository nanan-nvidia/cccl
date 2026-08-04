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

// when we have 4-255 runs per warp tile: each lane processes its own 32 elements
template <int items_per_thread, class ValueT, class OffT, class ReductionOpT>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void chunk_reduce_rotated(
  ValueT* __restrict__ d_aggregates,
  ValueT* __restrict__ smem_vals, // the staged tile
  unsigned my_word,
  OffT global_runs_before_warp_tile,
  int warp_tile_offset,
  int wt_end,
  int lane_id,
  ReductionOpT op,
  ValueT& lead_out,
  ValueT& tail_out)
{
  ValueT* const wt_base = smem_vals + warp_tile_offset;
  // first, rotate to avoid bank conflicts
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
} // namespace detail::reduce_by_key

CUB_NAMESPACE_END
