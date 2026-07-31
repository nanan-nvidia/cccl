// Standalone harness shim over the reduce-by-key lookahead kernel (the sibling of the rle encode
// lookahead kernel; RBK = RLE with an open AGGREGATE instead of an open length). Presents the
// campaign harness interface (rbk_impl::winner_config + persistent_rbk_encode) so the bench,
// verify, and hammer drivers run unchanged.
#pragma once

#include <cub/device/device_reduce.cuh>
#include <cub/device/dispatch/kernels/kernel_reduce_by_key_lookahead.cuh>

#include <cuda/std/algorithm>
#include <cuda/std/functional>

namespace rbk_impl
{
namespace rbk_kernels = CUB_NS_QUALIFIER::detail::reduce_by_key::lookahead;

using CountStateT = rbk_kernels::CountStateT;
template <class ValueT, class OffT>
using TileValueRecordT = rbk_kernels::TileValueRecordT<ValueT, OffT>;
template <class OffT>
using TilePrefixT = rbk_kernels::PrefixT<OffT>;

// tile count below which stock CUB runs
constexpr int kStockDispatchTiles = 1024;

template <class KeyT, class ValueT, int kIptOverride = 0, int kStagesOverride = 0>
struct winner_config
{
  static constexpr int kIPT          = (kIptOverride != 0) ? kIptOverride : 32;
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

// CUB temp storage is caller scratch with no contents contract between calls, so the states are
// cleared on EVERY launch (same as stock CUB's init kernels)
template <class StateT>
__global__ void rbk_init_states(StateT* states, long long n_states)
{
  const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_states)
  {
    states[i] = StateT{}; // tag 0 never matches the published tag (1)
  }
}

template <class Config, class KeyT, class ValueT, class NumRunsT, class OffT>
inline cudaError_t persistent_rbk_encode(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  const KeyT* d_keys,
  const ValueT* d_values,
  KeyT* d_unique,
  ValueT* d_aggregates,
  NumRunsT* d_num_runs,
  OffT num_items,
  cudaStream_t stream = 0)
{
  // size query must cover BOTH paths (the same allocation may serve either across calls)
  size_t cub_bytes = 0;
  cub::DeviceReduce::ReduceByKey(
    nullptr, cub_bytes, d_keys, d_unique, d_values, d_aggregates, d_num_runs, cuda::std::plus<>{}, num_items, stream);
  const long long q_tiles = rbk_state_tiles<Config>((long long) num_items);
  const size_t pers_bytes =
    (size_t) q_tiles
    * (sizeof(CountStateT) + sizeof(TileValueRecordT<ValueT, OffT>) + sizeof(TilePrefixT<OffT>)
       + (size_t) Config::kNumCompWarps * 32 * sizeof(unsigned));
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
      cuda::std::plus<>{},
      num_items,
      stream);
  }
  auto* count_states    = (CountStateT*) d_temp_storage;
  auto* value_records   = (TileValueRecordT<ValueT, OffT>*) (count_states + tiles);
  auto* tile_prefixes   = (TilePrefixT<OffT>*) (value_records + tiles);
  auto* flag_words      = (unsigned*) (tile_prefixes + tiles);
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
  // PASS 2: values. Persistent TMA-ring kernel when the values base is 16B-aligned and the ring
  // fits the opt-in smem; the block-per-tile kernel is the fallback (misaligned bases, small smem)
  constexpr size_t kValDynSmem = (size_t) Config::kStages * Config::kTileSize * sizeof(ValueT);
  // staged mode: one bulk TMA per block pulls the tile's values + flag words into smem (IKET
  // receipt: bands 3.7x faster from smem; the 6-block occupancy is the latency pipeline). The
  // smem is values + flags per block.
  constexpr size_t kValBlockSmem =
    (size_t) Config::kTileSize * sizeof(ValueT) + (size_t) Config::kNumCompWarps * 32 * sizeof(unsigned);
  const bool vals_staged = (((size_t) d_values & 15) == 0) && (size_t) smem_optin >= kValBlockSmem;
  auto* vkernel          = rbk_kernels::DeviceReduceByKeyLookaheadValueKernel<typename Config::Selector, ValueT, OffT>;
  error = cudaFuncSetAttribute(vkernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) kValBlockSmem);
  if (error != cudaSuccess)
  {
    return error;
  }
  int v_occ = 0, sm_count = 0;
  error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &v_occ, vkernel, Config::kNumCompWarps * 32, vals_staged ? kValBlockSmem : 0);
  if (error != cudaSuccess)
  {
    return error;
  }
  error = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
  if (error != cudaSuccess)
  {
    return error;
  }
  const int v_grid = (int) cuda::std::min<long long>(tiles, (long long) v_occ * sm_count);
  vkernel<<<v_grid, Config::kNumCompWarps * 32, vals_staged ? kValBlockSmem : 0, stream>>>(
    d_values, d_aggregates, flag_words, tile_prefixes, value_records, num_items, (int) tiles, vals_staged);
  error = cudaPeekAtLastError();
  if (error != cudaSuccess)
  {
    return error;
  }
  // PASS 3: boundary cleanup (one warp per tile with a pending cross-tile close)
  const int cleanup_blocks = (int) ((tiles * 32 + 255) / 256);
  rbk_kernels::DeviceReduceByKeyLookaheadCleanupKernel<<<cleanup_blocks, 256, 0, stream>>>(
    d_aggregates, value_records, (int) tiles);
  return cudaPeekAtLastError();
}
} // namespace rbk_impl
