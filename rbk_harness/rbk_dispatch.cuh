#pragma once

#include <cub/device/device_reduce.cuh>
#include <cub/device/dispatch/kernels/kernel_reduce_by_key_lookahead.cuh>

#include <cuda/std/algorithm>
#include <cuda/std/functional>

namespace rbk_impl
{
namespace rbk_kernels = CUB_NS_QUALIFIER::detail::reduce_by_key;

using TilePartialStateT = rbk_kernels::TilePartialStateT;
template <class ValueT>
using TileValueRecordT = rbk_kernels::TileValueRecordT<ValueT>;
template <class OffT>
using TilePrefixT = rbk_kernels::PrefixT<OffT>;

template <class KeyT, class ValueT>
struct winner_config
{
  static constexpr int kIPT                  = 32;
  static constexpr int kNumCompWarps         = 8;
  static constexpr int kStages               = 6; // key ring depth
  static constexpr int kPosStages            = (kStages + 1) / 2;
  static constexpr int kPollMlp              = 5;
  static constexpr int kFlagStagingThreshold = 32;
  static constexpr int kWarpTileSize         = 32 * kIPT;
  static constexpr int kTileSize             = kNumCompWarps * kWarpTileSize;

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

  static constexpr size_t kDynSmem =
    (size_t) kStages * kPolicy.slot_stride((int) sizeof(KeyT), (int) alignof(KeyT)) * sizeof(KeyT);
};

template <class Config>
inline long long rbk_state_tiles(long long n)
{
  return (n + Config::kTileSize - 1) / Config::kTileSize;
}

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
  ReductionOpT reduction_op = {},
  float* pass_ms            = nullptr) // when set: wall ms of the 4 launches (init/keys/values/cleanup)
{
  cudaEvent_t pass_ev[5];
  if (pass_ms)
  {
    for (auto& e : pass_ev)
    {
      cudaEventCreate(&e);
    }
    cudaEventRecord(pass_ev[0], stream);
  }
  using RecordT           = TileValueRecordT<ValueT>;
  const long long q_tiles = rbk_state_tiles<Config>((long long) num_items);
  const auto align16      = [](size_t b) {
    return (b + 15) & ~(size_t) 15;
  };
  const size_t pers_bytes =
    align16((size_t) q_tiles * sizeof(TilePartialStateT)) + align16((size_t) q_tiles * sizeof(RecordT))
    + align16((size_t) q_tiles * sizeof(TilePrefixT<OffT>))
    + align16((size_t) q_tiles * (size_t) Config::kNumCompWarps * 32 * sizeof(unsigned));

  const size_t required = pers_bytes;
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
  // PROTOTYPE (Blackwell-only): keys <= 4 bytes, 4-byte values, 16B-aligned values base;
  // anything else errors out
  if constexpr (sizeof(KeyT) > 4 || sizeof(ValueT) != 4)
  {
    return cudaErrorInvalidValue;
  }
  if (((size_t) d_values & 15) != 0)
  {
    return cudaErrorInvalidValue;
  }
  cudaError_t error     = cudaSuccess;
  auto* count_states    = (TilePartialStateT*) d_temp_storage;
  auto* value_records   = (RecordT*) ((char*) count_states + align16(tiles * sizeof(TilePartialStateT)));
  auto* tile_prefixes   = (TilePrefixT<OffT>*) ((char*) value_records + align16(tiles * sizeof(RecordT)));
  auto* flag_words      = (unsigned*) ((char*) tile_prefixes + align16(tiles * sizeof(TilePrefixT<OffT>)));
  const int init_blocks = (int) ((tiles + 255) / 256);
  rbk_kernels::DeviceReduceByKeyLookaheadInitKernel<typename Config::Selector>
    <<<init_blocks, 256, 0, stream>>>(count_states, tiles);
  if (pass_ms)
  {
    cudaEventRecord(pass_ev[1], stream);
  }

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
    flag_words,
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
  if (pass_ms)
  {
    cudaEventRecord(pass_ev[2], stream);
  }
  // PASS 2: values
  constexpr size_t kValDynSmem   = (size_t) Config::kStages * Config::kTileSize * sizeof(ValueT);
  constexpr size_t kValBlockSmem = (size_t) Config::kTileSize * sizeof(ValueT);
  auto* vkernel =
    rbk_kernels::DeviceReduceByKeyLookaheadValueKernel<typename Config::Selector, ValueT, OffT, ReductionOpT>;
  error = cudaFuncSetAttribute(vkernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) kValBlockSmem);
  if (error != cudaSuccess)
  {
    return error;
  }
  vkernel<<<(int) tiles, Config::kNumCompWarps * 32, kValBlockSmem, stream>>>(
    d_values, d_aggregates, flag_words, tile_prefixes, value_records, num_items, (int) tiles, reduction_op);
  error = cudaPeekAtLastError();
  if (error != cudaSuccess)
  {
    return error;
  }
  if (pass_ms)
  {
    cudaEventRecord(pass_ev[3], stream);
  }
  // PASS 3: boundary cleanup (one warp per tile with a pending cross-tile close)
  const int cleanup_blocks = (int) ((tiles * 32 + 255) / 256);
  rbk_kernels::DeviceReduceByKeyLookaheadCleanupKernel<KeyT, ValueT, OffT, ReductionOpT>
    <<<cleanup_blocks, 256, 0, stream>>>(
      d_aggregates, value_records, tile_prefixes, d_keys, Config::kTileSize, (int) tiles, reduction_op);
  if (pass_ms)
  {
    cudaEventRecord(pass_ev[4], stream);
    cudaEventSynchronize(pass_ev[4]);
    for (int p = 0; p < 4; ++p)
    {
      cudaEventElapsedTime(&pass_ms[p], pass_ev[p], pass_ev[p + 1]);
    }
    for (auto& e : pass_ev)
    {
      cudaEventDestroy(e);
    }
  }
  return cudaPeekAtLastError();
}
} // namespace rbk_impl
