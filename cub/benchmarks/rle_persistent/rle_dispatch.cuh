// Size-dispatched RLE encode: ONE persistent instantiation (the tuned Blackwell kernel) above a
// tile-count threshold, stock cub::DeviceRunLengthEncode::Encode below it. In the small/mid band
// the problem is end-to-end launch/latency-bound and the persistent machinery cannot beat a thin
// kernel by construction (measured: v48-v53b); above it the persistent kernel wins 1.3-5.2x
// across the full type matrix. This mirrors how a CUB policy integration would dispatch.
#pragma once

#include <cub/device/device_run_length_encode.cuh>

#include <algorithm>

#define RLE_NS rle_impl
#include "persistent_rle.cu"
#undef RLE_NS

using KeyT     = rle_impl::KeyT;
using LenT     = rle_impl::LenT;
using NumRunsT = rle_impl::NumRunsT;
using OffT     = rle_impl::OffT;
using u64      = rle_impl::u64;

constexpr int kTileSize = rle_impl::kTileSize;

inline long long rle_state_tiles(long long n)
{
  return (n + kTileSize - 1) / kTileSize;
}

// tile count below which stock CUB runs. Boundary sweep (v55): 512 tiles still shows one sub-par
// cell (2^22 seg2 = 0.97x); 1024 tiles is clean -- parity below (stock == stock), >=1.17x above.
#ifndef RLE_STOCK_TILES
#  define RLE_STOCK_TILES 1024
#endif

// ---- temp-storage protocol -------------------------------------------------------------------
// persistent path: [ header: u64 magic | u32 gen | u32 rsvd ][ u64 tile states ... ]
// The init kernel bumps the generation when the header matches (allocate-once-call-many pays a
// tiny kernel, cheaper than stock CUB per-call state init) and clears states exactly once
// otherwise -- including after the same allocation was used by the stock path (magic mismatch).
// All state is device-side: graph-capturable, no host statics.
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
  // size query must cover BOTH paths (the same allocation may serve either across calls)
  size_t cub_bytes = 0;
  cub::DeviceRunLengthEncode::Encode(nullptr, cub_bytes, d_keys, d_unique, d_counts, d_num_runs, num_items, stream);
  const size_t pers_bytes = sizeof(RleTempHeader) + (size_t) rle_state_tiles((long long) num_items) * sizeof(u64);
  const size_t required   = std::max(cub_bytes, pers_bytes);
  if (d_temp_storage == nullptr)
  {
    temp_storage_bytes = required;
    return cudaSuccess;
  }
  if (temp_storage_bytes < required)
  {
    return cudaErrorInvalidValue;
  }
  const long long tiles = rle_state_tiles((long long) num_items);
  if (tiles < RLE_STOCK_TILES)
  {
    return cub::DeviceRunLengthEncode::Encode(
      d_temp_storage, temp_storage_bytes, d_keys, d_unique, d_counts, d_num_runs, num_items, stream);
  }
  auto* hdr   = (RleTempHeader*) d_temp_storage;
  u64* states = (u64*) (hdr + 1);
  rle_init_states<<<128, 256, 0, stream>>>(hdr, states, tiles);
  rle_impl::persistent_rle_launch(
    d_keys, d_unique, d_counts, d_num_runs, states, &hdr->gen, num_items, (int) tiles, stream);
  return cudaGetLastError();
}
