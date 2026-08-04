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
#include <cub/device/dispatch/kernels/kernel_rle_encode_lookahead.cuh>
#include <cub/device/dispatch/tuning/tuning_rle_encode.cuh>
#include <cub/util_arch.cuh>
#include <cub/util_macro.cuh>
#include <cub/warp/warp_reduce.cuh>
#include <cub/warp/warp_scan.cuh>

#include <cuda/atomic>
#include <cuda/ptx>
#include <cuda/std/bit>
#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

CUB_NAMESPACE_BEGIN

namespace detail::reduce_by_key::lookahead
{
// requires PTX ISA 9.2 (CUDA 13.2): the load warp uses the cp.async.bulk .ignore_oob qualifier
namespace ptx = ::cuda::ptx;

// shared warpspeed machinery comes from the RLE lookahead kernel this one is derived from;
// only the RBK-specific pieces are defined below
using rle::encode::clc_next_tile_id;
using rle::encode::compute_head_flags;
using rle::encode::load_tile_keys;
using rle::encode::nth_set_bit;
using rle::encode::poll_and_fold;
using rle::encode::poll_fold_windows;
using rle::encode::PrefixT;
using rle::encode::reduce_and_publish_tile_state;
using rle::encode::RingCursorT;
using rle::encode::RunSpanT;
using rle::encode::scan_warp_tile_run_counts;
using rle::encode::stage_head_positions;
using rle::encode::swizzle_xor_stride32;
using rle::encode::tile_published;
using rle::encode::TilePartialStateT;
using rle::encode::wait_parity;
using rle::encode::WarpTileRunScanT;

_CCCL_HOST_DEVICE_API constexpr int num_total_threads(const RleLookaheadPolicy& policy)
{
  // keys-only pass: load + compute + poll + key stores (values run in their own kernel)
  const int num_total_warps = 1 /*load*/ + policy.compute_warps + 1 /*poll*/ + policy.compute_warps /*key store*/;
  return num_total_warps * 32;
}

constexpr unsigned full_mask = 0xffffffffu;

// per-tile VALUE RECORD, written with plain stores by the value warps
// and read by the cleanup kernel after the main kernel completes
template <class ValueT>
struct TileValueRecordT
{
  ValueT open_agg; // sum after the tile's last head (whole-tile sum when head-free)
  ValueT lead_agg; // sum before the tile's first head (whole-tile sum when head-free)
};

// values loader: plain ignore_oob TMA of the tile's values into the ring slot -- no pad, no
// skip (values have no predecessor semantics); the last tile ZERO-FILLS past tile_len, so the
// consumer bands never need tail guards (zeros are additive identity)
template <int tile_size, class ValueT>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void load_tile_values(
  ValueT* slot,
  const ValueT* d_values,
  int tile_id,
  int tile_len,
  bool last_tile,
  ::cuda::std::uint64_t* full_bar,
  int lane_id)
{
  if (lane_id == 0)
  {
    const unsigned nbytes     = (unsigned) ((size_t) tile_len * sizeof(ValueT));
    const unsigned span_bytes = (nbytes + 15u) & ~15u;
    ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, full_bar, span_bytes);
    ptx::cp_async_bulk_ignore_oob(
      ptx::space_shared,
      ptx::space_global,
      slot,
      d_values + (size_t) tile_id * tile_size,
      span_bytes,
      0u,
      last_tile ? (span_bytes - nbytes) : 0u,
      full_bar);
  }
  __syncwarp();
}

// __shfl_* has no >8B overloads: wide types (int128) shuffle as 64-bit halves
template <class T>
_CCCL_DEVICE_API _CCCL_FORCEINLINE T shfl_sync_wide(T v, int src_lane)
{
  if constexpr (sizeof(T) <= 8)
  {
    return __shfl_sync(full_mask, v, src_lane);
  }
  else
  {
    struct Halves
    {
      ::cuda::std::uint64_t h[sizeof(T) / 8];
    };
    auto hv = ::cuda::std::bit_cast<Halves>(v);
#pragma unroll
    for (int i = 0; i < (int) (sizeof(T) / 8); ++i)
    {
      hv.h[i] = __shfl_sync(full_mask, hv.h[i], src_lane);
    }
    return ::cuda::std::bit_cast<T>(hv);
  }
}

template <class T>
_CCCL_DEVICE_API _CCCL_FORCEINLINE T shfl_up_sync_wide(T v, unsigned delta)
{
  if constexpr (sizeof(T) <= 8)
  {
    return __shfl_up_sync(full_mask, v, delta);
  }
  else
  {
    struct Halves
    {
      ::cuda::std::uint64_t h[sizeof(T) / 8];
    };
    auto hv = ::cuda::std::bit_cast<Halves>(v);
#pragma unroll
    for (int i = 0; i < (int) (sizeof(T) / 8); ++i)
    {
      hv.h[i] = __shfl_up_sync(full_mask, hv.h[i], delta);
    }
    return ::cuda::std::bit_cast<T>(hv);
  }
}

template <class T>
_CCCL_DEVICE_API _CCCL_FORCEINLINE T shfl_down_sync_wide(T v, unsigned delta)
{
  if constexpr (sizeof(T) <= 8)
  {
    return __shfl_down_sync(full_mask, v, delta);
  }
  else
  {
    struct Halves
    {
      ::cuda::std::uint64_t h[sizeof(T) / 8];
    };
    auto hv = ::cuda::std::bit_cast<Halves>(v);
#pragma unroll
    for (int i = 0; i < (int) (sizeof(T) / 8); ++i)
    {
      hv.h[i] = __shfl_down_sync(full_mask, hv.h[i], delta);
    }
    return ::cuda::std::bit_cast<T>(hv);
  }
}

template <class T>
_CCCL_DEVICE_API _CCCL_FORCEINLINE T shfl_xor_sync_wide(T v, int mask)
{
  if constexpr (sizeof(T) <= 8)
  {
    return __shfl_xor_sync(full_mask, v, mask);
  }
  else
  {
    struct Halves
    {
      ::cuda::std::uint64_t h[sizeof(T) / 8];
    };
    auto hv = ::cuda::std::bit_cast<Halves>(v);
#pragma unroll
    for (int i = 0; i < (int) (sizeof(T) / 8); ++i)
    {
      hv.h[i] = __shfl_xor_sync(full_mask, hv.h[i], mask);
    }
    return ::cuda::std::bit_cast<T>(hv);
  }
}
} // namespace detail::reduce_by_key::lookahead

CUB_NAMESPACE_END
