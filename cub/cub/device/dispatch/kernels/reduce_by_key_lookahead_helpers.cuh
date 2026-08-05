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

namespace detail::reduce_by_key
{
namespace ptx = ::cuda::ptx;

using rle::encode::clc_next_tile_id;
using rle::encode::compute_head_flags;
using rle::encode::HeadFlagDecodeT;
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
} // namespace detail::reduce_by_key

CUB_NAMESPACE_END
