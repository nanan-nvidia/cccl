// Standalone harness shim over the reduce-by-key lookahead kernel (the sibling of the rle encode
// lookahead kernel; RBK = RLE with an open AGGREGATE instead of an open length). Presents the
// campaign harness interface (rbk_impl::winner_config + persistent_rbk_encode) so the bench,
// verify, and hammer drivers run unchanged.
#pragma once

#include <cub/device/device_reduce.cuh>
#include <cub/device/dispatch/kernels/kernel_reduce_by_key_lookahead.cuh>
#ifdef K_FUSED
#  include <cub/device/dispatch/kernels/kernel_reduce_by_key_fused.cuh>
#endif

#include <cuda/std/algorithm>
#include <cuda/std/functional>

namespace rbk_impl
{
namespace rbk_kernels = CUB_NS_QUALIFIER::detail::reduce_by_key::lookahead;

using CountStateT = rbk_kernels::CountStateT;
template <class ValueT>
using TileValueRecordT = rbk_kernels::TileValueRecordT<ValueT>;
template <class OffT>
using TilePrefixT = rbk_kernels::PrefixT<OffT>;

// tile count below which stock CUB runs
constexpr int kStockDispatchTiles = 1024;

template <class KeyT, class ValueT, int kIptOverride = 0, int kStagesOverride = 0>
struct winner_config
{
  // ring bytes stay constant across key widths (RLE lookahead selector: 8192x4B = 4096x8B = 2048x16B)
  static constexpr int kIPT =
    (kIptOverride != 0) ? kIptOverride : (sizeof(KeyT) >= 16 ? 8 : (sizeof(KeyT) == 8 ? 16 : 32));
  static constexpr int kNumCompWarps = 8;
  static constexpr int kStages       = (kStagesOverride != 0) ? kStagesOverride : 6; // key ring depth
  static constexpr int kPosStages    = (kStages + 1) / 2;
#ifdef K_POLL_MLP
  static constexpr int kPollMlp = K_POLL_MLP;
#else
  static constexpr int kPollMlp = 5;
#endif
#ifdef K_FW_THRESH
  static constexpr int kFlagStagingThreshold = K_FW_THRESH;
#else
  static constexpr int kFlagStagingThreshold = 32;
#endif
  static constexpr int kWarpTileSize = 32 * kIPT;
  static constexpr int kTileSize     = kNumCompWarps * kWarpTileSize;

  static constexpr CUB_NS_QUALIFIER::RleLookaheadPolicy kPolicy{
    kIPT, kNumCompWarps, kStages, kPosStages, kPollMlp, kFlagStagingThreshold};

  struct Policy
  {
    CUB_NS_QUALIFIER::RleLookaheadPolicy lookahead;
  };

  struct Selector
  {
    constexpr Policy operator()(::cuda::compute_capability) const
    {
      return {kPolicy};
    }
  };

  // positions are gone: the dyn smem is the keys ring only; the policy's own accounting still
  // charges the dead pos ring (it routed the S=6 arm to stock via the opt-in gate)
  static constexpr size_t kDynSmem =
    (size_t) kStages * kPolicy.slot_stride((int) sizeof(KeyT), (int) alignof(KeyT)) * sizeof(KeyT);
};

template <class Config>
inline long long rbk_state_tiles(long long n)
{
  return (n + Config::kTileSize - 1) / Config::kTileSize;
}

#ifdef K_FUSED
#  include "rbk_dispatch_fused.cuh" // supplies persistent_rbk_encode for the fused prototype
#else
template <class Config, class KeyT, class ValueT, class NumRunsT, class OffT, class ReductionOpT = ::cuda::std::plus<>>
inline cudaError_t persistent_rbk_encode(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  const KeyT* d_keys,
  const ValueT* d_values,
  KeyT* d_unique,
  ValueT* d_aggregates,
  NumRunsT* d_num_runs,
  OffT num_items,
  cudaStream_t stream       = 0,
  ReductionOpT reduction_op = {})
{
  using RecordT = TileValueRecordT<ValueT>;
  // size query must cover BOTH paths (the same allocation may serve either across calls)
  size_t cub_bytes = 0;
  cub::DeviceReduce::ReduceByKey(
    nullptr, cub_bytes, d_keys, d_unique, d_values, d_aggregates, d_num_runs, reduction_op, num_items, stream);
  const long long q_tiles = rbk_state_tiles<Config>((long long) num_items);
  // each carve is rounded up to 16B: the flag-words bulk prefetch requires a 16B-aligned base
  // (24B TileValueRecordT with 8B values left it 8-mod-16 -> misaligned address)
  const auto align16 = [](size_t b) {
    return (b + 15) & ~(size_t) 15;
  };
  const size_t pers_bytes =
    align16((size_t) q_tiles * sizeof(CountStateT)) + align16((size_t) q_tiles * sizeof(RecordT))
    + align16((size_t) q_tiles * sizeof(TilePrefixT<OffT>));

  const size_t required = cuda::std::max(cub_bytes, pers_bytes);
  if (d_temp_storage == nullptr)
  {
    temp_storage_bytes = required;
    return cudaSuccess;
  }
  if (temp_storage_bytes < required)
  {
    return cudaErrorInvalidValue;
  }
  const long long tiles = rbk_state_tiles<Config>((long long) num_items);
  int device = 0, cc_major = 0, smem_optin = 0;
  cudaError_t error = cudaGetDevice(&device);
  if (error == cudaSuccess)
  {
    error = cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, device);
  }
  if (error == cudaSuccess)
  {
    error = cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  }
  if (error != cudaSuccess)
  {
    return error;
  }
  // PROTOTYPE: 4-byte values only; staging is required; unsupported inputs ERROR OUT
  if constexpr (sizeof(ValueT) != 4)
  {
    return cudaErrorInvalidValue;
  }
#  ifdef RBK_VK_KEYS_GLOBAL
  constexpr size_t kValBlockSmemGate = (size_t) Config::kTileSize * sizeof(ValueT);
#  else
  constexpr size_t kValBlockSmemGate =
    ((((size_t) Config::kTileSize * sizeof(ValueT)) + 15) & ~(size_t) 15)
    + (size_t) Config::kPolicy.slot_stride((int) sizeof(KeyT), (int) alignof(KeyT)) * sizeof(KeyT);
#  endif
  if ((((size_t) d_values & 15) != 0) || (size_t) smem_optin < kValBlockSmemGate)
  {
    return cudaErrorInvalidValue;
  }
  if (tiles < kStockDispatchTiles || tiles > 0x7fffffff || cc_major < 10 || (size_t) smem_optin < Config::kDynSmem)
  {
    return cub::DeviceReduce::ReduceByKey(
      d_temp_storage,
      temp_storage_bytes,
      d_keys,
      d_unique,
      d_values,
      d_aggregates,
      d_num_runs,
      reduction_op,
      num_items,
      stream);
  }
  auto* count_states    = (CountStateT*) d_temp_storage;
  auto* value_records   = (RecordT*) ((char*) count_states + align16(tiles * sizeof(CountStateT)));
  auto* tile_prefixes   = (TilePrefixT<OffT>*) ((char*) value_records + align16(tiles * sizeof(RecordT)));
  const int init_blocks = (int) ((tiles + 255) / 256);
  // only the tagged COUNT states need clearing; the value records are plain outputs of the main
  // kernel, synchronized by the launch boundary
  rbk_kernels::DeviceReduceByKeyLookaheadInitKernel<typename Config::Selector>
    <<<init_blocks, 256, 0, stream>>>(count_states, tiles);

  auto* kernel = rbk_kernels::DeviceReduceByKeyLookaheadKernel<typename Config::Selector, KeyT, ValueT, NumRunsT, OffT>;
  error        = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) Config::kDynSmem);
  if (error != cudaSuccess)
  {
    return error;
  }
  constexpr int threads = rbk_kernels::num_total_threads(Config::kPolicy);
  kernel<<<(int) tiles, threads, Config::kDynSmem, stream>>>(
    d_keys,
    d_values,
    d_unique,
    d_aggregates,
    d_num_runs,
    count_states,
    tile_prefixes,
    num_items,
    (int) tiles,
    Config::kStages,
    Config::kPosStages,
    /*keys_staged=*/true);
  error = cudaPeekAtLastError();
  if (error != cudaSuccess)
  {
    return error;
  }
  // PASS 2: values. Persistent TMA-ring kernel when the values base is 16B-aligned and the ring
  // fits the opt-in smem; the block-per-tile kernel is the fallback (misaligned bases, small smem)
  constexpr size_t kValDynSmem = (size_t) Config::kStages * Config::kTileSize * sizeof(ValueT);
  // staged mode: one bulk TMA per block pulls the tile's values + flag words into smem (IKET
  // receipt: bands 3.7x faster from smem; the 6-block occupancy is the latency pipeline). The
  // smem is values + flags per block.
  constexpr size_t kValBlockSmem = kValBlockSmemGate;
  auto* vkernel =
    rbk_kernels::DeviceReduceByKeyLookaheadValueKernel<typename Config::Selector, KeyT, ValueT, OffT, ReductionOpT>;
  error = cudaFuncSetAttribute(vkernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) kValBlockSmem);
  if (error != cudaSuccess)
  {
    return error;
  }
  vkernel<<<(int) tiles, Config::kNumCompWarps * 32, kValBlockSmem, stream>>>(
    d_keys, d_values, d_aggregates, tile_prefixes, value_records, num_items, (int) tiles, reduction_op);
  error = cudaPeekAtLastError();
  if (error != cudaSuccess)
  {
    return error;
  }
  // PASS 3: boundary cleanup (one warp per tile with a pending cross-tile close)
  const int cleanup_blocks = (int) ((tiles * 32 + 255) / 256);
  rbk_kernels::DeviceReduceByKeyLookaheadCleanupKernel<KeyT, ValueT, OffT, ReductionOpT>
    <<<cleanup_blocks, 256, 0, stream>>>(
      d_aggregates, value_records, tile_prefixes, d_keys, Config::kTileSize, (int) tiles, reduction_op);
  return cudaPeekAtLastError();
}
#endif // K_FUSED
} // namespace rbk_impl
