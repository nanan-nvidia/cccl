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
#undef RLE_STATIC_ASSIGN
#define RLE_STATIC_ASSIGN 1 // ~1 tile/block band: steal machinery is pure exit latency
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

// ---- CUB-shaped temp-storage protocol -----------------------------------------------------
// temp layout: [ header: u64 magic | u32 gen | u32 rsvd ][ u64 tile states ... ]
// The init kernel bumps the generation when the header matches (allocate-once-call-many pays a
// tiny kernel, cheaper than stock CUB per-call state init) and clears states exactly once
// otherwise. All state is device-side: graph-capturable, no host statics.
struct RleTempHeader
{
  unsigned long long magic;
  unsigned gen;
  unsigned rsvd;
};
inline constexpr unsigned long long kRleTempMagic = 0x524c455f54454d50ull; // "RLE_TEMP"

__global__ void rle_init_states(RleTempHeader* hdr, u64* states, long long n_states)
{
  const bool fresh = (hdr->magic != kRleTempMagic) || (hdr->gen >= 0xfffffff0u);
  if (fresh)
  {
    const long long stride = (long long) gridDim.x * blockDim.x;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < n_states; i += stride)
    {
      states[i] = 0;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
      hdr->magic = kRleTempMagic;
      hdr->gen   = 1; // stale zeroed words carry gen 0 and never match gen >= 1
    }
  }
  else if (blockIdx.x == 0 && threadIdx.x == 0)
  {
    hdr->gen = hdr->gen + 1;
  }
}

inline size_t persistent_rle_temp_bytes(long long num_items)
{
  return sizeof(RleTempHeader) + (size_t) rle_state_tiles(num_items) * sizeof(u64);
}

inline void persistent_rle_dispatch_launch(
  const KeyT* d_keys,
  KeyT* d_unique,
  LenT* d_counts,
  NumRunsT* d_num_runs,
  u64* d_tile_states,
  const unsigned* d_launch_gen,
  OffT num_items,
  cudaStream_t stream)
{
  const int big_tiles = (int) ((num_items + rle_big::kTileSize - 1) / rle_big::kTileSize);
  if (big_tiles < RLE_DISPATCH_SMALL_TILES)
  {
    const int small_tiles = (int) ((num_items + rle_small::kTileSize - 1) / rle_small::kTileSize);
    rle_small::persistent_rle_launch(
      d_keys, d_unique, d_counts, d_num_runs, d_tile_states, d_launch_gen, num_items, small_tiles, stream);
  }
  else if (big_tiles < RLE_DISPATCH_MID_TILES)
  {
    const int mid_tiles = (int) ((num_items + rle_mid::kTileSize - 1) / rle_mid::kTileSize);
    rle_mid::persistent_rle_launch(
      d_keys, d_unique, d_counts, d_num_runs, d_tile_states, d_launch_gen, num_items, mid_tiles, stream);
  }
  else
  {
    rle_big::persistent_rle_launch(
      d_keys, d_unique, d_counts, d_num_runs, d_tile_states, d_launch_gen, num_items, big_tiles, stream);
  }
}

// CUB-shaped entry point: two-phase (null d_temp_storage -> required size), stream-ordered,
// graph-capturable. Mirrors the cub::DeviceRunLengthEncode::Encode calling convention.
inline cudaError_t persistent_rle_encode(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  const KeyT* d_keys,
  KeyT* d_unique,
  LenT* d_counts,
  NumRunsT* d_num_runs,
  OffT num_items,
  cudaStream_t stream = 0)
{
  const size_t required = persistent_rle_temp_bytes((long long) num_items);
  if (d_temp_storage == nullptr)
  {
    temp_storage_bytes = required;
    return cudaSuccess;
  }
  if (temp_storage_bytes < required)
  {
    return cudaErrorInvalidValue;
  }
  auto* hdr                = (RleTempHeader*) d_temp_storage;
  u64* states              = (u64*) (hdr + 1);
  const long long n_states = rle_state_tiles((long long) num_items);
  rle_init_states<<<128, 256, 0, stream>>>(hdr, states, n_states);
  persistent_rle_dispatch_launch(d_keys, d_unique, d_counts, d_num_runs, states, &hdr->gen, num_items, stream);
  return cudaGetLastError();
}
