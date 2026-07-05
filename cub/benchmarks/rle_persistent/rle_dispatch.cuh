// Size-dispatched persistent RLE: instantiates persistent_rle.cu TWICE (big-tile + small-tile)
// and picks by tile count. Below ~2 tiles/SM the persistent pipeline never fills (N=2^20 = 128
// big tiles on 148 SMs measured 0.94-0.96x vs CUB at dense segs); smaller tiles restore
// parallelism there while the big-tile champion keeps its tuned geometry at scale.
#pragma once

#define RLE_NS rle_big
#include "persistent_rle.cu"
#undef RLE_NS

#ifndef RLE_SMALL_IPT
#  define RLE_SMALL_IPT 8 // small-path tile = kNumCompWarps * 32 * this (8 -> 2048 elements @4B)
#endif
#ifdef K_IPT
#  undef K_IPT
#endif
#define K_IPT  RLE_SMALL_IPT
#define RLE_NS rle_small
#ifdef RLE_STATIC_ASSIGN
#  undef RLE_STATIC_ASSIGN
#endif
#define RLE_STATIC_ASSIGN 1 // ~1 tile/block: steal machinery is pure exit latency
#include "persistent_rle.cu"
#undef RLE_NS
#undef K_IPT

// mid instantiation: big tile, SHALLOW ring -- at ~1 tile/SM the 5-deep pipeline never fills and
// only inflates the smem carveout (212KB -> 98KB) and block prologue; v49 measured that SMALLER
// tiles lose here (tile count = prefix-chain length = latency at 4MB scale).
#ifndef RLE_MID_STAGES
#  define RLE_MID_STAGES 2
#endif
#ifndef RLE_MID_IPT
#  define RLE_MID_IPT 32 // mid tile = kNumCompWarps * 32 * this
#endif
#ifndef RLE_MID_STATIC
#  define RLE_MID_STATIC 1
#endif
#undef RLE_STATIC_ASSIGN
#define RLE_STATIC_ASSIGN RLE_MID_STATIC
#define K_IPT             RLE_MID_IPT
#define K_STAGES          RLE_MID_STAGES
#define K_POS_STAGES      RLE_MID_STAGES
#define RLE_NS            rle_mid
#include "persistent_rle.cu"
#undef RLE_NS
#undef K_IPT
#undef K_STAGES
#undef K_POS_STAGES
#undef RLE_STATIC_ASSIGN
#define RLE_STATIC_ASSIGN 0 // restore for any later includes

using KeyT     = rle_big::KeyT;
using LenT     = rle_big::LenT;
using NumRunsT = rle_big::NumRunsT;
using OffT     = rle_big::OffT;
using u64      = rle_big::u64;
// harness padding / state sizing must cover BOTH paths: pad to the big tile (a multiple of the
// small tile), size the state array by the small tile count (the larger of the two)
constexpr int kTileSize = rle_big::kTileSize;
static_assert(rle_big::kTileSize % rle_small::kTileSize == 0, "small tile must divide big tile");
static_assert(rle_mid::kTileSize % rle_small::kTileSize == 0, "small tile must divide mid tile");

inline long long rle_state_tiles(long long n)
{
  return (n + rle_small::kTileSize - 1) / rle_small::kTileSize;
}

// dispatch thresholds (big-tile counts): below SMALL -> small tiles (more parallelism at trivial
// sizes); below MID -> big tiles with the shallow ring (latency regime, <=~2 tiles/SM); else the
// deep-ring champion. v49: small tiles at the mid regime LOSE (chain length); tune by sweeping.
#ifndef RLE_DISPATCH_SMALL_TILES
#  define RLE_DISPATCH_SMALL_TILES 33
#endif
#ifndef RLE_DISPATCH_MID_TILES
#  define RLE_DISPATCH_MID_TILES 296
#endif

inline void persistent_rle_dispatch_launch(
  const KeyT* d_keys,
  KeyT* d_unique,
  LenT* d_counts,
  NumRunsT* d_num_runs,
  u64* d_tile_states,
  int* d_tile_counter,
  OffT num_items,
  cudaStream_t stream)
{
  const int big_tiles = (int) ((num_items + rle_big::kTileSize - 1) / rle_big::kTileSize);
  if (big_tiles < RLE_DISPATCH_SMALL_TILES)
  {
    const int small_tiles = (int) ((num_items + rle_small::kTileSize - 1) / rle_small::kTileSize);
    rle_small::persistent_rle_launch(
      d_keys, d_unique, d_counts, d_num_runs, d_tile_states, d_tile_counter, num_items, small_tiles, stream);
  }
  else if (big_tiles < RLE_DISPATCH_MID_TILES)
  {
    const int mid_tiles = (int) ((num_items + rle_mid::kTileSize - 1) / rle_mid::kTileSize);
    rle_mid::persistent_rle_launch(
      d_keys, d_unique, d_counts, d_num_runs, d_tile_states, d_tile_counter, num_items, mid_tiles, stream);
  }
  else
  {
    rle_big::persistent_rle_launch(
      d_keys, d_unique, d_counts, d_num_runs, d_tile_states, d_tile_counter, num_items, big_tiles, stream);
  }
}
