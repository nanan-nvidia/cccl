// Size-dispatched RLE encode: ONE persistent instantiation (the tuned Blackwell kernel) above a
// tile-count threshold, stock cub::DeviceRunLengthEncode::Encode below it. In the small/mid band
// the problem is end-to-end launch/latency-bound and the persistent machinery cannot beat a thin
// kernel by construction (measured: v48-v53b); above it the persistent kernel wins 1.3-5.2x
// across the full type matrix. This mirrors how a CUB policy integration would dispatch.
#pragma once

#include <cub/device/device_run_length_encode.cuh>

#include <algorithm>

#include "persistent_rle.cu"

using KeyT     = rle_impl::KeyT;
using LenT     = rle_impl::LenT;
using NumRunsT = rle_impl::NumRunsT;
using OffT     = rle_impl::OffT;
using u64      = rle_impl::u64;
using StateT   = rle_impl::StateT;

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
// persistent path: [ header: u64 magic | u32 launch_gen | u32 rsvd ][ StateT tile states ... ]
// The init kernel bumps the generation when the header matches (allocate-once-call-many pays a
// tiny kernel, cheaper than stock CUB per-call state init) and clears states exactly once
// otherwise -- including after the same allocation was used by the stock path (magic mismatch).
// RLE_STATE32 builds additionally re-clear at every 15-call tag-cycle restart (see state_tag).
// All state is device-side: graph-capturable, no host statics.
struct RleTempHeader
{
  unsigned long long magic;
  unsigned launch_gen;
  unsigned rsvd;
};
inline constexpr unsigned long long kRleTempMagic = 0x524c455f54454d50ull; // "RLE_TEMP"

// SINGLE block: the clear/bump decision must be uniform across all clearing threads, and a multi-
// block grid cannot read the header while block 0 rewrites it without a race (harmless for the
// 64-bit almost-never-wraps tag, fatal for the 15-call cycle: a torn decision = a partial clear).
// One block stages the decision through smem and __syncthreads. Clears are rare (once per
// allocation; every 15th call under RLE_STATE32) and small (4-8B/tile), so one SM suffices.
__global__ void rle_init_states(RleTempHeader* hdr, StateT* states, long long n_states)
{
  __shared__ unsigned s_gen;
  __shared__ bool s_clear;
  if (threadIdx.x == 0)
  {
    const bool fresh = (hdr->magic != kRleTempMagic) || (hdr->launch_gen >= 0xfffffff0u);
    const unsigned g = fresh ? 1u : hdr->launch_gen + 1u;
    s_gen            = g;
#if RLE_STATE32
    s_clear = fresh || ((g - 1u) % 15u == 0u); // tag cycle restarts at this call
#else
    s_clear = fresh;
#endif
  }
  __syncthreads();
  if (s_clear)
  {
    for (long long i = threadIdx.x; i < n_states; i += blockDim.x)
    {
      states[i] = 0; // tag 0 never matches a live tag (>= 1)
    }
  }
  if (threadIdx.x == 0)
  {
    hdr->magic      = kRleTempMagic;
    hdr->launch_gen = s_gen;
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
  const size_t pers_bytes = sizeof(RleTempHeader) + (size_t) rle_state_tiles((long long) num_items) * sizeof(StateT);
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
  auto* hdr      = (RleTempHeader*) d_temp_storage;
  StateT* states = (StateT*) (hdr + 1);
  rle_init_states<<<1, 256, 0, stream>>>(hdr, states, tiles);
  rle_impl::persistent_rle_launch(
    d_keys, d_unique, d_counts, d_num_runs, states, &hdr->launch_gen, num_items, (int) tiles, stream);
  return cudaGetLastError();
}
