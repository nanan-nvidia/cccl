// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_segmented_radix_sort.cuh>
#include <cub/agent/agent_sub_warp_merge_sort.cuh>
#include <cub/detail/device_double_buffer.cuh>
#include <cub/device/dispatch/dispatch_common.cuh>
#include <cub/device/dispatch/tuning/tuning_segmented_sort.cuh>
#include <cub/util_arch.cuh>
#include <cub/util_device.cuh>
#include <cub/warp/warp_reduce.cuh>

#include <cuda/__cmath/ceil_div.h>

CUB_NAMESPACE_BEGIN
namespace detail::segmented_sort
{
// Type used to index within segments within a single invocation
using local_segment_index_t = ::cuda::std::uint32_t;
// Type used for total number of segments and to index within segments globally
using global_segment_offset_t = ::cuda::std::int64_t;

// Maximum resident threads per SM of the architecture the device pass compiles for.
_CCCL_HOST_DEVICE constexpr int max_threads_per_sm()
{
#if !defined(__CUDA_ARCH__)
  return 2048;
#elif __CUDA_ARCH__ == 750
  return 1024;
#elif (__CUDA_ARCH__ >= 860 && __CUDA_ARCH__ < 900) || __CUDA_ARCH__ >= 1200
  return 1536;
#else
  return 2048;
#endif
}

_CCCL_HOST_DEVICE constexpr int small_kernel_min_blocks_per_sm(int threads_per_block, bool keys_only)
{
  return (keys_only ? max_threads_per_sm() : max_threads_per_sm() / 2) / threads_per_block;
}

// Match the two-blocks-per-SM occupancy ptxas achieves for the loop-free kernel at these key sizes.
template <typename KeyT, typename ValueT>
_CCCL_HOST_DEVICE constexpr int large_kernel_min_blocks_per_sm()
{
  const bool keys_only = ::cuda::std::is_same_v<ValueT, NullType>;
  return (keys_only && (sizeof(KeyT) == 1 || sizeof(KeyT) >= 8)) ? 2 : 1;
}

template <typename OffsetT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
struct LargeSegmentsSelectorT
{
  OffsetT value{};
  BeginOffsetIteratorT d_offset_begin{};
  EndOffsetIteratorT d_offset_end{};
  global_segment_offset_t base_segment_offset{};

#if !_CCCL_COMPILER(NVRTC)
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE
  LargeSegmentsSelectorT(OffsetT value, BeginOffsetIteratorT d_offset_begin, EndOffsetIteratorT d_offset_end)
      : value(value)
      , d_offset_begin(d_offset_begin)
      , d_offset_end(d_offset_end)
  {}
#endif // !_CCCL_COMPILER(NVRTC)

  _CCCL_DEVICE _CCCL_FORCEINLINE bool operator()(local_segment_index_t segment_id) const
  {
    // NOLINTNEXTLINE(bugprone-misplaced-widening-cast)
    const OffsetT segment_size =
      d_offset_end[base_segment_offset + segment_id] - d_offset_begin[base_segment_offset + segment_id];
    return segment_size > value;
  }
};

template <typename OffsetT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
struct SmallSegmentsSelectorT
{
  OffsetT value{};
  BeginOffsetIteratorT d_offset_begin{};
  EndOffsetIteratorT d_offset_end{};
  global_segment_offset_t base_segment_offset{};

#if !_CCCL_COMPILER(NVRTC)
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE
  SmallSegmentsSelectorT(OffsetT value, BeginOffsetIteratorT d_offset_begin, EndOffsetIteratorT d_offset_end)
      : value(value)
      , d_offset_begin(d_offset_begin)
      , d_offset_end(d_offset_end)
  {}
#endif // !_CCCL_COMPILER(NVRTC)

  _CCCL_DEVICE _CCCL_FORCEINLINE bool operator()(local_segment_index_t segment_id) const
  {
    // NOLINTNEXTLINE(bugprone-misplaced-widening-cast)
    const OffsetT segment_size =
      d_offset_end[base_segment_offset + segment_id] - d_offset_begin[base_segment_offset + segment_id];
    return segment_size < value;
  }
};

/**
 * @brief Fallback kernel, in case there's not enough segments to
 *        take advantage of partitioning.
 *
 * In this case, the sorting method is still selected based on the segment size.
 * If a single warp can sort the segment, the algorithm will use the sub-warp
 * merge sort. Otherwise, the algorithm will use the in-shared-memory version of
 * block radix sort. If data don't fit into shared memory, the algorithm will
 * use in-global-memory radix sort.
 *
 * @param[in] d_keys_in_orig
 *   Input keys buffer
 *
 * @param[out] d_keys_out_orig
 *   Output keys buffer
 *
 * @param[in,out] d_keys_double_buffer
 *   Double keys buffer
 *
 * @param[in] d_values_in_orig
 *   Input values buffer
 *
 * @param[out] d_values_out_orig
 *   Output values buffer
 *
 * @param[in,out] d_values_double_buffer
 *   Double values buffer
 *
 * @param[in] d_begin_offsets
 *   Random-access input iterator to the sequence of beginning offsets of length
 *   `num_segments`, such that `d_begin_offsets[i]` is the first element of the
 *   i-th data segment in `d_keys_*` and `d_values_*`
 *
 * @param[in] d_end_offsets
 *   Random-access input iterator to the sequence of ending offsets of length
 *   `num_segments`, such that `d_end_offsets[i]-1` is the last element of the
 *   i-th data segment in `d_keys_*` and `d_values_*`.
 *   If `d_end_offsets[i]-1 <= d_begin_offsets[i]`, the i-th segment is
 *   considered empty.
 */
template <SortOrder Order,
          typename PolicySelector,
          typename KeyT,
          typename ValueT,
          typename BeginOffsetIteratorT,
          typename EndOffsetIteratorT,
          typename OffsetT>
#if _CCCL_HAS_CONCEPTS()
  requires segmented_sort_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
__launch_bounds__(current_policy<PolicySelector>().large_segment.threads_per_block)
  _CCCL_KERNEL_ATTRIBUTES void DeviceSegmentedSortFallbackKernel(
    const KeyT* d_keys_in_orig,
    KeyT* d_keys_out_orig,
    device_double_buffer<KeyT> d_keys_double_buffer,
    const ValueT* d_values_in_orig,
    ValueT* d_values_out_orig,
    device_double_buffer<ValueT> d_values_double_buffer,
    const BeginOffsetIteratorT d_begin_offsets,
    const EndOffsetIteratorT d_end_offsets)
{
  static constexpr SegmentedSortPolicy active_policy = current_policy<PolicySelector>();
  static constexpr auto large_policy                 = active_policy.large_segment;
  using LargeSegmentPolicyT                          = detail::agent_radix_sort_downsweep_policy<
                             0,
                             0,
                             void,
                             large_policy.load_algorithm,
                             large_policy.load_modifier,
                             large_policy.rank_algorithm,
                             large_policy.scan_algorithm,
                             large_policy.radix_bits,
                             NoScaling<large_policy.threads_per_block, large_policy.items_per_thread>>;
  static constexpr auto medium_policy = active_policy.medium_segment;
  using MediumPolicyT                 = agent_sub_warp_merge_sort_policy<
                    medium_policy.threads_per_block,
                    medium_policy.threads_per_warp,
                    medium_policy.items_per_thread,
                    medium_policy.load_algorithm,
                    medium_policy.load_modifier,
                    medium_policy.store_algorithm>;

  const auto segment_id = static_cast<local_segment_index_t>(blockIdx.x);
  OffsetT segment_begin = d_begin_offsets[segment_id];
  OffsetT segment_end   = d_end_offsets[segment_id];
  OffsetT num_items     = segment_end - segment_begin;

  if (num_items <= 0)
  {
    return;
  }

  using AgentSegmentedRadixSortT =
    radix_sort::AgentSegmentedRadixSort<Order == SortOrder::Descending, LargeSegmentPolicyT, KeyT, ValueT, OffsetT>;

  using WarpReduceT = cub::WarpReduce<KeyT>;

  using AgentWarpMergeSortT =
    sub_warp_merge_sort::AgentSubWarpSort<Order == SortOrder::Descending, MediumPolicyT, KeyT, ValueT, OffsetT>;

  __shared__ union
  {
    typename AgentSegmentedRadixSortT::TempStorage block_sort;
    typename WarpReduceT::TempStorage warp_reduce;
    typename AgentWarpMergeSortT::TempStorage medium_warp_sort;
  } temp_storage;

  constexpr bool keys_only = ::cuda::std::is_same_v<ValueT, NullType>;
  AgentSegmentedRadixSortT agent(num_items, temp_storage.block_sort);

  constexpr int begin_bit = 0;
  constexpr int end_bit   = sizeof(KeyT) * 8;

  constexpr int cacheable_tile_size = LargeSegmentPolicyT::BLOCK_THREADS * LargeSegmentPolicyT::ITEMS_PER_THREAD;

  d_keys_in_orig += segment_begin;
  d_keys_out_orig += segment_begin;

  if (!keys_only)
  {
    d_values_in_orig += segment_begin;
    d_values_out_orig += segment_begin;
  }

  if (num_items <= MediumPolicyT::ITEMS_PER_TILE)
  {
    // Sort by a single warp
    if (threadIdx.x < MediumPolicyT::WARP_THREADS)
    {
      AgentWarpMergeSortT(temp_storage.medium_warp_sort)
        .ProcessSegment(num_items, d_keys_in_orig, d_keys_out_orig, d_values_in_orig, d_values_out_orig);
    }
  }
  else if (num_items < cacheable_tile_size)
  {
    // Sort by a CTA if data fits into shared memory
    agent.ProcessSinglePass(begin_bit, end_bit, d_keys_in_orig, d_values_in_orig, d_keys_out_orig, d_values_out_orig);
  }
  else
  {
    // Sort by a CTA with multiple reads from global memory
    int current_bit = begin_bit;
    int pass_bits   = (::cuda::std::min) (int{LargeSegmentPolicyT::RADIX_BITS}, (end_bit - current_bit));

    d_keys_double_buffer = device_double_buffer<KeyT>(
      d_keys_double_buffer.current() + segment_begin, d_keys_double_buffer.alternate() + segment_begin);

    if (!keys_only)
    {
      d_values_double_buffer = device_double_buffer<ValueT>(
        d_values_double_buffer.current() + segment_begin, d_values_double_buffer.alternate() + segment_begin);
    }

    agent.ProcessIterative(
      current_bit,
      pass_bits,
      d_keys_in_orig,
      d_values_in_orig,
      d_keys_double_buffer.current(),
      d_values_double_buffer.current());
    current_bit += pass_bits;

    _CCCL_PRAGMA_NOUNROLL()
    while (current_bit < end_bit)
    {
      pass_bits = (::cuda::std::min) (int{LargeSegmentPolicyT::RADIX_BITS}, (end_bit - current_bit));

      __syncthreads();
      agent.ProcessIterative(
        current_bit,
        pass_bits,
        d_keys_double_buffer.current(),
        d_values_double_buffer.current(),
        d_keys_double_buffer.alternate(),
        d_values_double_buffer.alternate());

      d_keys_double_buffer.swap();
      d_values_double_buffer.swap();
      current_bit += pass_bits;
    }
  }
}

/**
 * @brief Single kernel for moderate size (less than a few thousand items)
 *        segments.
 *
 * This kernel allocates a sub-warp per segment. Therefore, this kernel assigns
 * a single thread block to multiple segments. Segments fall into two
 * categories. An architectural warp usually sorts segments in the medium-size
 * category, while a few threads sort segments in the small-size category. Since
 * segments are partitioned, we know the last thread block index assigned to
 * sort medium-size segments. A particular thread block can check this number to
 * find out which category it was assigned to sort. In both cases, the
 * merge sort is used.
 *
 * @param[in] num_segments
 *   Total number of segments in this invocation
 *
 * @param[in] d_group_sizes
 *   Device array holding the number of large (`d_group_sizes[0]`) and small
 *   (`d_group_sizes[1]`) segments produced by the partitioning stage
 *
 * @param[in] d_small_segments_indices
 *   Small segments mapping of length `d_group_sizes[1]`, such that
 *   `d_small_segments_indices[i]` is the input segment index
 *
 * @param[in] d_large_and_medium_segments_indices
 *   Large and medium segments mapping of length @p num_segments, with large
 *   segment indices at the front and medium segment indices at the back
 *
 * @param[in] d_keys_in_orig
 *   Input keys buffer
 *
 * @param[out] d_keys_out_orig
 *   Output keys buffer
 *
 * @param[in] d_values_in_orig
 *   Input values buffer
 *
 * @param[out] d_values_out_orig
 *   Output values buffer
 *
 * @param[in] d_begin_offsets
 *   Random-access input iterator to the sequence of beginning offsets of length
 *   `num_segments`, such that `d_begin_offsets[i]` is the first element of the
 *   <em>i</em><sup>th</sup> data segment in `d_keys_*` and `d_values_*`
 *
 * @param[in] d_end_offsets
 *   Random-access input iterator to the sequence of ending offsets of length
 *   `num_segments`, such that `d_end_offsets[i]-1` is the last element of the
 *   <em>i</em><sup>th</sup> data segment in `d_keys_*` and `d_values_*`. If
 *   `d_end_offsets[i]-1 <= d_begin_offsets[i]`, the <em>i</em><sup>th</sup> is
 *   considered empty.
 */
template <SortOrder Order,
          typename PolicySelector,
          typename KeyT,
          typename ValueT,
          typename BeginOffsetIteratorT,
          typename EndOffsetIteratorT,
          typename OffsetT>
#if _CCCL_HAS_CONCEPTS()
  requires segmented_sort_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
__launch_bounds__(current_policy<PolicySelector>().small_segment.threads_per_block,
                  small_kernel_min_blocks_per_sm(current_policy<PolicySelector>().small_segment.threads_per_block,
                                                 ::cuda::std::is_same_v<ValueT, NullType>))
  _CCCL_KERNEL_ATTRIBUTES void DeviceSegmentedSortKernelSmall(
    const local_segment_index_t num_segments,
    const local_segment_index_t* const d_group_sizes,
    [[maybe_unused]] local_segment_index_t* const d_ticket,
    const local_segment_index_t* const d_small_segments_indices,
    const local_segment_index_t* const d_large_and_medium_segments_indices,
    const KeyT* const d_keys_in,
    KeyT* const d_keys_out,
    const ValueT* const d_values_in,
    ValueT* const d_values_out,
    const BeginOffsetIteratorT d_begin_offsets,
    const EndOffsetIteratorT d_end_offsets)
{
  using local_segment_index_t = local_segment_index_t;

  static constexpr SegmentedSortPolicy active_policy = current_policy<PolicySelector>();
  static constexpr auto small_policy                 = active_policy.small_segment;
  using SmallPolicyT                                 = agent_sub_warp_merge_sort_policy<
                                    small_policy.threads_per_block,
                                    small_policy.threads_per_warp,
                                    small_policy.items_per_thread,
                                    small_policy.load_algorithm,
                                    small_policy.load_modifier,
                                    small_policy.store_algorithm>;
  static constexpr auto medium_policy = active_policy.medium_segment;
  using MediumPolicyT                 = agent_sub_warp_merge_sort_policy<
                    medium_policy.threads_per_block,
                    medium_policy.threads_per_warp,
                    medium_policy.items_per_thread,
                    medium_policy.load_algorithm,
                    medium_policy.load_modifier,
                    medium_policy.store_algorithm>;

  using MediumAgentWarpMergeSortT =
    sub_warp_merge_sort::AgentSubWarpSort<Order == SortOrder::Descending, MediumPolicyT, KeyT, ValueT, OffsetT>;

  using SmallAgentWarpMergeSortT =
    sub_warp_merge_sort::AgentSubWarpSort<Order == SortOrder::Descending, SmallPolicyT, KeyT, ValueT, OffsetT>;

  constexpr auto segments_per_medium_block = static_cast<local_segment_index_t>(MediumPolicyT::SEGMENTS_PER_BLOCK);

  constexpr auto segments_per_small_block = static_cast<local_segment_index_t>(SmallPolicyT::SEGMENTS_PER_BLOCK);

  // The segment counts are only known on the device, so the grid may be smaller than the number of blocks
  // required to process all segments. Blocks claim work from the ticket counter until all of it is done.
  const local_segment_index_t small_segments  = d_group_sizes[1];
  const local_segment_index_t medium_segments = num_segments - d_group_sizes[0] - small_segments;

  const local_segment_index_t* const d_medium_segments_indices =
    d_large_and_medium_segments_indices + (num_segments - medium_segments);

  // Each warp claims work on its own, and the two roles get segregated (non-overlapping) storage,
  // so no block-wide barrier is ever needed and no team ever waits for another.
  __shared__ typename MediumAgentWarpMergeSortT::TempStorage medium_storage[segments_per_medium_block];
  __shared__ typename SmallAgentWarpMergeSortT::TempStorage small_storage[segments_per_small_block];

  constexpr auto medium_teams_per_warp     = static_cast<local_segment_index_t>(32 / MediumPolicyT::WARP_THREADS);
  constexpr auto small_teams_per_warp      = static_cast<local_segment_index_t>(32 / SmallPolicyT::WARP_THREADS);
  const local_segment_index_t medium_units = ::cuda::ceil_div(medium_segments, medium_teams_per_warp);
  const local_segment_index_t total_units  = medium_units + ::cuda::ceil_div(small_segments, small_teams_per_warp);

  // Chunked claims amortize the ticket latency; the chunk adapts so every warp still gets several claims.
  const local_segment_index_t total_warps = gridDim.x * (blockDim.x / 32);
  const local_segment_index_t warp_chunk =
    (::cuda::std::min) ((::cuda::std::max) (total_units / (8 * total_warps), local_segment_index_t{1}),
                        local_segment_index_t{8});
  while (true)
  {
    local_segment_index_t unit{};
    if ((threadIdx.x & 31) == 0)
    {
      unit = atomicAdd(d_ticket, warp_chunk);
    }
    unit = __shfl_sync(0xffffffff, unit, 0);
    if (unit >= total_units)
    {
      break;
    }

    const local_segment_index_t unit_end = (::cuda::std::min) (unit + warp_chunk, total_units);
    for (; unit < unit_end; ++unit)
    {
      if (unit < medium_units)
      {
        const local_segment_index_t medium_segment_id =
          unit * medium_teams_per_warp + (threadIdx.x & 31) / MediumPolicyT::WARP_THREADS;

        if (medium_segment_id < medium_segments)
        {
          const local_segment_index_t global_segment_id = d_medium_segments_indices[medium_segment_id];

          const OffsetT segment_begin = d_begin_offsets[global_segment_id];
          const OffsetT segment_end   = d_end_offsets[global_segment_id];
          const OffsetT num_items     = segment_end - segment_begin;

          MediumAgentWarpMergeSortT(medium_storage[threadIdx.x / MediumPolicyT::WARP_THREADS])
            .ProcessSegment(num_items,
                            d_keys_in + segment_begin,
                            d_keys_out + segment_begin,
                            d_values_in + segment_begin,
                            d_values_out + segment_begin);
        }
      }
      else
      {
        const local_segment_index_t small_segment_id =
          (unit - medium_units) * small_teams_per_warp + (threadIdx.x & 31) / SmallPolicyT::WARP_THREADS;

        if (small_segment_id < small_segments)
        {
          const local_segment_index_t global_segment_id = d_small_segments_indices[small_segment_id];

          const OffsetT segment_begin = d_begin_offsets[global_segment_id];
          const OffsetT segment_end   = d_end_offsets[global_segment_id];
          const OffsetT num_items     = segment_end - segment_begin;

          SmallAgentWarpMergeSortT(small_storage[threadIdx.x / SmallPolicyT::WARP_THREADS])
            .ProcessSegment(num_items,
                            d_keys_in + segment_begin,
                            d_keys_out + segment_begin,
                            d_values_in + segment_begin,
                            d_values_out + segment_begin);
        }
      }
    }
  }
}

/**
 * @brief Single kernel for large size (more than a few thousand items) segments.
 *
 * @param[in] d_group_sizes
 *   Device array whose first element holds the number of large segments
 *   produced by the partitioning stage
 *
 * @param[in] d_ticket
 *   Device counter used by blocks to claim segments; must be zero before the
 *   kernel starts
 *
 * @param[in] d_segments_indices
 *   Large segments mapping of length `d_group_sizes[0]`, such that
 *   `d_segments_indices[i]` is the input segment index
 *
 * @param[in] d_keys_in_orig
 *   Input keys buffer
 *
 * @param[out] d_keys_out_orig
 *   Output keys buffer
 *
 * @param[in] d_values_in_orig
 *   Input values buffer
 *
 * @param[out] d_values_out_orig
 *   Output values buffer
 *
 * @param[in] d_begin_offsets
 *   Random-access input iterator to the sequence of beginning offsets of length
 *   `num_segments`, such that `d_begin_offsets[i]` is the first element of the
 *   <em>i</em><sup>th</sup> data segment in `d_keys_*` and `d_values_*`
 *
 * @param[in] d_end_offsets
 *   Random-access input iterator to the sequence of ending offsets of length
 *   `num_segments`, such that `d_end_offsets[i]-1` is the last element of the
 *   <em>i</em><sup>th</sup> data segment in `d_keys_*` and `d_values_*`. If
 *   `d_end_offsets[i]-1 <= d_begin_offsets[i]`, the <em>i</em><sup>th</sup> is
 *   considered empty.
 */
template <SortOrder Order,
          typename PolicySelector,
          typename KeyT,
          typename ValueT,
          typename BeginOffsetIteratorT,
          typename EndOffsetIteratorT,
          typename OffsetT>
#if _CCCL_HAS_CONCEPTS()
  requires segmented_sort_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
__launch_bounds__(current_policy<PolicySelector>().large_segment.threads_per_block,
                  large_kernel_min_blocks_per_sm<KeyT, ValueT>())
  _CCCL_KERNEL_ATTRIBUTES void DeviceSegmentedSortKernelLarge(
    const local_segment_index_t* const d_group_sizes,
    local_segment_index_t* const d_ticket,
    const local_segment_index_t* const d_segments_indices,
    const KeyT* const d_keys_in_orig,
    KeyT* const d_keys_out_orig,
    device_double_buffer<KeyT> d_keys_double_buffer,
    const ValueT* const d_values_in_orig,
    ValueT* const d_values_out_orig,
    device_double_buffer<ValueT> d_values_double_buffer,
    const BeginOffsetIteratorT d_begin_offsets,
    const EndOffsetIteratorT d_end_offsets)
{
  static constexpr SegmentedSortRadixSortPolicy large_policy = current_policy<PolicySelector>().large_segment;
  using LargeSegmentPolicyT                                  = detail::agent_radix_sort_downsweep_policy<
                                     0,
                                     0,
                                     void,
                                     large_policy.load_algorithm,
                                     large_policy.load_modifier,
                                     large_policy.rank_algorithm,
                                     large_policy.scan_algorithm,
                                     large_policy.radix_bits,
                                     NoScaling<large_policy.threads_per_block, large_policy.items_per_thread>>;

  using local_segment_index_t = local_segment_index_t;

  using AgentSegmentedRadixSortT =
    radix_sort::AgentSegmentedRadixSort<Order == SortOrder::Descending, LargeSegmentPolicyT, KeyT, ValueT, OffsetT>;

  __shared__ typename AgentSegmentedRadixSortT::TempStorage storage;
  __shared__ local_segment_index_t block_segment_index[2];

  constexpr int small_tile_size = LargeSegmentPolicyT::BLOCK_THREADS * LargeSegmentPolicyT::ITEMS_PER_THREAD;
  constexpr int begin_bit       = 0;
  constexpr int end_bit         = sizeof(KeyT) * 8;
  constexpr bool keys_only      = ::cuda::std::is_same_v<ValueT, NullType>;

  // The grid may be smaller than the number of segments: blocks claim segments through the ticket
  // counter, one iteration ahead into the other half of the ping-pong buffer.
  const local_segment_index_t large_segments = d_group_sizes[0];
  if (threadIdx.x == 0)
  {
    block_segment_index[0] = atomicAdd(d_ticket, local_segment_index_t{1});
  }
  __syncthreads();
  for (int parity = 0;; parity ^= 1)
  {
    const local_segment_index_t segment_index = block_segment_index[parity];
    if (segment_index >= large_segments)
    {
      return;
    }
    if (threadIdx.x == 0)
    {
      // All threads passed the previous iteration's barrier, so the other slot is free to overwrite
      block_segment_index[parity ^ 1] = atomicAdd(d_ticket, local_segment_index_t{1});
    }

    const local_segment_index_t global_segment_id = d_segments_indices[segment_index];
    const OffsetT segment_begin                   = d_begin_offsets[global_segment_id];
    const OffsetT segment_end                     = d_end_offsets[global_segment_id];
    const OffsetT num_items                       = segment_end - segment_begin;

    AgentSegmentedRadixSortT agent(num_items, storage);

    const KeyT* d_keys_in = d_keys_in_orig + segment_begin;
    KeyT* d_keys_out      = d_keys_out_orig + segment_begin;

    const ValueT* d_values_in = d_values_in_orig;
    ValueT* d_values_out      = d_values_out_orig;

    if (!keys_only)
    {
      d_values_in  = d_values_in_orig + segment_begin;
      d_values_out = d_values_out_orig + segment_begin;
    }

    if (num_items < small_tile_size)
    {
      // Sort in shared memory if the segment fits into it
      agent.ProcessSinglePass(begin_bit, end_bit, d_keys_in, d_values_in, d_keys_out, d_values_out);
    }
    else
    {
      // Sort reading global memory multiple times
      int current_bit = begin_bit;
      int pass_bits   = (::cuda::std::min) (int{LargeSegmentPolicyT::RADIX_BITS}, (end_bit - current_bit));

      // The double buffers stay in the kernel parameters; staging them in shared memory regresses multipass.
      device_double_buffer<KeyT> keys_double_buffer(
        d_keys_double_buffer.current() + segment_begin, d_keys_double_buffer.alternate() + segment_begin);

      device_double_buffer<ValueT> values_double_buffer = d_values_double_buffer;
      if (!keys_only)
      {
        values_double_buffer = device_double_buffer<ValueT>(
          d_values_double_buffer.current() + segment_begin, d_values_double_buffer.alternate() + segment_begin);
      }

      agent.ProcessIterative(
        current_bit, pass_bits, d_keys_in, d_values_in, keys_double_buffer.current(), values_double_buffer.current());
      current_bit += pass_bits;

      _CCCL_PRAGMA_NOUNROLL()
      while (current_bit < end_bit)
      {
        pass_bits = (::cuda::std::min) (int{LargeSegmentPolicyT::RADIX_BITS}, (end_bit - current_bit));

        __syncthreads();
        agent.ProcessIterative(
          current_bit,
          pass_bits,
          keys_double_buffer.current(),
          values_double_buffer.current(),
          keys_double_buffer.alternate(),
          values_double_buffer.alternate());

        keys_double_buffer.swap();
        values_double_buffer.swap();
        current_bit += pass_bits;
      }
    }

    // Make the shared memory and the claimed index reusable before the next iteration
    __syncthreads();
  }
}
} // namespace detail::segmented_sort
CUB_NAMESPACE_END
