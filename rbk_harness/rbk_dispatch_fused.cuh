// FUSED SINGLE-PASS PROTOTYPE (quarantined): keys + values in one block-per-tile kernel,
// temp storage O(tiles). Built only under -DK_FUSED. This fragment is included by
// rbk_dispatch.cuh INSIDE namespace rbk_impl (after the shared aliases/helpers), which selects it in place
// of the two-pass persistent_rbk_encode. The two-pass implementation never references this file.
#pragma once

using FusedStateT = rbk_kernels::FusedStateT;

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
  using RecordT = rbk_kernels::FusedValueRecordT<ValueT, OffT>;
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
    align16((size_t) q_tiles * sizeof(FusedStateT)) + align16((size_t) q_tiles * sizeof(RecordT))
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
  constexpr size_t kValBlockSmemGate =
    (size_t) Config::kTileSize * sizeof(ValueT) + (size_t) Config::kNumCompWarps * 32 * sizeof(unsigned);
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
  // FUSED SINGLE PASS: one block-per-tile kernel, head flags never leave the SM; temp = O(tiles)
  if ((((size_t) d_keys & 15) != 0)) // 16B head-pad TMA needs an aligned keys base (prototype)
  {
    return cudaErrorInvalidValue;
  }
  auto* fused_states    = (FusedStateT*) d_temp_storage;
  auto* value_records   = (RecordT*) ((char*) fused_states + align16(tiles * sizeof(FusedStateT)));
  auto* tile_prefixes   = (TilePrefixT<OffT>*) ((char*) value_records + align16(tiles * sizeof(RecordT)));
  const int init_blocks = (int) ((tiles + 255) / 256);
  rbk_init_states<<<init_blocks, 256, 0, stream>>>(fused_states, tiles);
  constexpr int kPadElems = 16 / (int) sizeof(KeyT);
  (void) kPadElems;
#ifdef RBK_FUSED_KEYS_GLOBAL
  constexpr size_t kFusedSmem = (size_t) Config::kTileSize * sizeof(ValueT);
#else
  constexpr size_t kFusedSmem = ((((size_t) (Config::kTileSize + kPadElems) * sizeof(KeyT)) + 15) & ~(size_t) 15)
                              + (size_t) Config::kTileSize * sizeof(ValueT);
#endif
  if ((size_t) smem_optin < kFusedSmem)
  {
    return cudaErrorInvalidValue;
  }
  auto* fkernel =
    rbk_kernels::DeviceReduceByKeyFusedKernel<typename Config::Selector, KeyT, ValueT, NumRunsT, OffT, ReductionOpT>;
  error = cudaFuncSetAttribute(fkernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) kFusedSmem);
  if (error != cudaSuccess)
  {
    return error;
  }
  fkernel<<<(int) tiles, Config::kNumCompWarps * 32, kFusedSmem, stream>>>(
    d_keys,
    d_values,
    d_unique,
    d_aggregates,
    d_num_runs,
    fused_states,
    tile_prefixes,
    value_records,
    num_items,
    (int) tiles,
    reduction_op);
  error = cudaPeekAtLastError();
  if (error != cudaSuccess)
  {
    return error;
  }
  const int cleanup_blocks = (int) ((tiles * 32 + 255) / 256);
  rbk_kernels::DeviceReduceByKeyFusedCleanupKernel<ValueT, OffT, ReductionOpT>
    <<<cleanup_blocks, 256, 0, stream>>>(d_aggregates, value_records, (int) tiles, reduction_op);
  return cudaPeekAtLastError();
}
