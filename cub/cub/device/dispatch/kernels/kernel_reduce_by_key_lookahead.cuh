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

#include <cub/device/dispatch/kernels/reduce_by_key_lookahead_bands.cuh>

CUB_NAMESPACE_BEGIN

namespace detail::reduce_by_key
{
// we aim for 1 block/SM since it is easier to manage resources: we do not need to worry about occupancy anymore
template <typename PolicySelector, class KeyT, class ValueT, class NumRunsT, class OffT>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void device_reduce_by_key_lookahead_body(
  const KeyT* __restrict__ d_keys,
  const ValueT* __restrict__ d_values,
  KeyT* __restrict__ d_unique,
  ValueT* __restrict__ d_aggregates,
  NumRunsT* __restrict__ d_num_runs,
  TilePartialStateT* __restrict__ count_states,
  unsigned* __restrict__ d_flag_words, // [num_tiles][compute_warps*32]: the persisted run structure
  PrefixT<OffT>* __restrict__ d_tile_prefix, // [num_tiles]: packed (run prefix, last headed tile)
  OffT num_items,
  int num_tiles,
  int key_ring_stages,
  int pos_ring_stages,
  bool keys_staged)
{
  static constexpr RleLookaheadPolicy policy = current_policy<PolicySelector>().lookahead;
  static_assert(16 % sizeof(KeyT) == 0, "KeyT size must be a power of two <= 16");
  static_assert(alignof(KeyT) <= 16, "Alignment <= 16");
  static_assert(policy.items_per_thread >= 1 && policy.items_per_thread <= 32, "items_per_thread must be in [1, 32]");
  static_assert(policy.compute_warps >= 1 && policy.compute_warps <= 31, "compute_warps must be in [1, 31]");
  static_assert(policy.key_ring_stages >= 1, "at least one pipeline stage");
  static_assert(policy.pos_ring_stages >= 1 && 2 * policy.pos_ring_stages >= policy.key_ring_stages,
                "pos ring parity wait aliases unless 2*pos_ring_stages >= key_ring_stages");
  static_assert(policy.floor_pos_ring_stages() <= policy.pos_ring_stages
                  && 2 * policy.floor_pos_ring_stages() >= policy.floor_key_ring_stages(),
                "the unstaged floor configuration must satisfy the pos ring parity bound");
  static_assert(policy.floor_dyn_smem_bytes() + RleLookaheadPolicy::static_smem_budget
                  <= RleLookaheadPolicy::default_smem_per_block,
                "the unstaged floor configuration must launch within the default shared memory limit on every device");
  static_assert(policy.tile_size() <= 0xffff && policy.tile_size() <= 32768,
                "tile_size must fit the 16-bit state words and signed 16-bit staged positions");
  static_assert(num_total_threads(policy) <= 1024, "a CTA is capped at 1024 threads");
  static_assert(policy.buf_per_lane() * ((int) sizeof(KeyT) + 4) <= 64,
                "reg-buf rounds must fit the 64B/lane register budget");
  static_assert(::cuda::std::is_integral_v<OffT> && policy.tile_size() <= ::cuda::std::numeric_limits<OffT>::max(),
                "OffT must be an integer type wide enough for one tile");
  constexpr int items_per_thread = policy.items_per_thread;
  constexpr int compute_warps    = policy.compute_warps;
  // TWO-PASS DESIGN: this kernel is KEYS-ONLY (flags, counts, prefixes, unique keys); the value
  // pass is a separate pipeline-free kernel fed by the persisted flag words and tile prefixes
  constexpr int key_store_warps        = compute_warps;
  constexpr int max_key_ring_stages    = policy.key_ring_stages;
  constexpr int max_pos_ring_stages    = policy.pos_ring_stages;
  constexpr int flag_staging_threshold = policy.flag_staging_threshold;
  // in the regressed case: always stage positions, so the store warps never run the flag-decode drain vvv
  const int staging_threshold  = keys_staged ? flag_staging_threshold : 0;
  constexpr int warp_tile_size = policy.warp_tile_size();
  constexpr int tile_size      = policy.tile_size();
  constexpr int slot_pad       = policy.slot_pad((int) sizeof(KeyT));
  constexpr int slot_stride    = policy.slot_stride((int) sizeof(KeyT), (int) alignof(KeyT));
  using PrefixT                = reduce_by_key::PrefixT<OffT>;
  // the dense band's fused-stream crossover (runs per warp tile); below it, per-run span walks
  // are output-proportional and cheaper
#ifdef RBK_STREAM_DIV
  constexpr int stream_threshold = policy.warp_tile_size() / RBK_STREAM_DIV;
#else
  // B200 receipt: /4 pulls the seg4 band into the stream and costs nothing anywhere else
  constexpr int stream_threshold = policy.warp_tile_size() / 4;
#endif
  // [key_ring_stages][tile_size] input keys
  // [key_ring_stages][tile_size] int16 staged head positions
  extern __shared__ char smem_raw[];
  KeyT* const tile_buf = (KeyT*) smem_raw;
  __shared__ int tile_id_buf[max_key_ring_stages]; // which global tile each ring slot holds (LOAD gets it with
                                                   // try_cancel)
  __shared__ int warp_run_counts[max_key_ring_stages][compute_warps]; // per compute warp run counts
  __shared__ int warp_last_heads[max_key_ring_stages][compute_warps]; // per compute warp last head idx (-1 if none)
  __shared__ unsigned head_flag_buf[max_key_ring_stages][compute_warps * 32]; // staged head-flag words
  __shared__ unsigned word_mask[max_key_ring_stages][compute_warps]; // which of a wt's 32 flag words hold heads

  // for POLL to pass STORE the packed (run_count prefix, open aggregate) pair
  __shared__ PrefixT prefix_packed[max_key_ring_stages];

  // LOAD --full--> COMPUTE & POLL
  // COMPUTE(all warps) --computed--> COMPUTE w0, then cw0 calculates & publishes this tile's aggregate to the global
  // POLL --prefixed--> STORE
  // STORE --empty--> LOAD & POLL
  __shared__ ::cuda::std::uint64_t full[max_key_ring_stages];
  __shared__ ::cuda::std::uint64_t computed[max_key_ring_stages], prefixed[max_key_ring_stages],
    empty[max_key_ring_stages];
  // COMPUTE warp w --staged_warp_tile[w]--> STORE: we arrive per warp tile handoff
  // i.e. store warps start working to drain a warp-tile as soon as ITS positions are staged
  __shared__ ::cuda::std::uint64_t staged_warp_tile[max_key_ring_stages][compute_warps];

  // try_cancel writes a 16-byte response into clc_resp + completes clc_bar's tx.
  __shared__ __align__(16) uint4 clc_resp;
  __shared__ ::cuda::std::uint64_t clc_bar;
  static_assert(sizeof(tile_id_buf) + sizeof(warp_run_counts) + sizeof(warp_last_heads) + sizeof(head_flag_buf)
                    + sizeof(word_mask) + sizeof(prefix_packed) + sizeof(full) + sizeof(computed) + sizeof(prefixed)
                    + sizeof(empty) + sizeof(staged_warp_tile) + sizeof(clc_resp) + sizeof(clc_bar)
                  <= RleLookaheadPolicy::static_smem_budget,
                "static shared memory exceeds the budget assumed by the floor launch guarantee");

  const int thr_id         = threadIdx.x;
  const int lane_id        = thr_id & 31;
  const int blk_id         = blockIdx.x;
  const unsigned base_skip = (alignof(KeyT) < 16) ? ((unsigned) (size_t) d_keys & 15u) : 0u;
  const int skip_elems     = (int) (base_skip / sizeof(KeyT));
  if (thr_id == 0)
  {
    for (int slot_id = 0; slot_id < max_key_ring_stages; ++slot_id)
    {
      ptx::mbarrier_init(&full[slot_id], 1);
      ptx::mbarrier_init(&computed[slot_id], compute_warps); // every compute warp arrives
      ptx::mbarrier_init(&prefixed[slot_id], 1);
      ptx::mbarrier_init(&empty[slot_id], key_store_warps);
      for (int cw = 0; cw < compute_warps; ++cw)
      {
        ptx::mbarrier_init(&staged_warp_tile[slot_id][cw], 1); // that compute warp's lane0
      }
    }

    ptx::mbarrier_init(&clc_bar, 1); // 1 arrival
  }
  // normal smem writes (e.g. mbarrier_init) go through the generic proxy
  // the TMA operations access shared memory through the async proxy. these are separate visibility domains,
  // so the init writes are not automatically visible to TMA.
  ptx::fence_proxy_async(ptx::space_shared);
  __syncthreads();

  constexpr warpspeed::SquadDesc squadLoad{0, 1};
  constexpr warpspeed::SquadDesc squadCompute{1, compute_warps};
  constexpr warpspeed::SquadDesc squadPoll{2, 1};
  constexpr warpspeed::SquadDesc squadKeyStore{3, key_store_warps};
  constexpr warpspeed::SquadDesc squads[] = {squadLoad, squadCompute, squadPoll, squadKeyStore};

  warpspeed::squadDispatch(
    warpspeed::getSpecialRegisters(), squads, [&](warpspeed::Squad squad) _CCCL_FORCEINLINE_LAMBDA {
      // if you are load
      if (squad == squadLoad)
      {
        // CLC tile assignment: gen0 tile = this CTA's launch id (blockIdx.x)
        int tile_id = blk_id;
        if (lane_id == 0)
        {
          // 16 is the try_cancel byte tx
          ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &clc_bar, 16);
          ptx::clusterlaunchcontrol_try_cancel(&clc_resp, &clc_bar);
        }
        RingCursorT key_ring;
        for (int pipeline_gen = 0;; ++pipeline_gen, key_ring.advance(key_ring_stages))
        {
          const int slot_id = key_ring.slot; // which slot is this?
          if (pipeline_gen >= key_ring_stages)
          {
            // need to wait for slot to be free
            wait_parity(&empty[slot_id], key_ring.parity ^ 1u);
          }
          if (lane_id == 0)
          {
            tile_id_buf[slot_id] = tile_id;
          }
          if (tile_id >= num_tiles)
          {
            if (lane_id == 0)
            {
              ptx::mbarrier_arrive(&full[slot_id]);
            }
            __syncwarp();
            break;
          }
          // over-fetch one 16B chunk to the left, so that we get last tiles last key
          // tile 0 has no predecessor and skips the over-fetch
          const bool first_tile = (tile_id == 0);
          const int tile_len    = (int) min((OffT) tile_size, num_items - (OffT) tile_id * tile_size);
          load_tile_keys<tile_size, slot_pad>(
            tile_buf + (size_t) slot_id * slot_stride,
            d_keys,
            tile_id,
            tile_len,
            first_tile,
            tile_id == num_tiles - 1,
            base_skip,
            &full[slot_id],
            lane_id,
            keys_staged);
          // consume the prefetched cancel, this is ok since it should be fast to get next cancelled id
          tile_id = clc_next_tile_id(clc_resp, clc_bar, pipeline_gen, num_tiles, lane_id);
        }
      }
      // if you are compute
      else if (squad == squadCompute)
      {
        const int compute_warp_id  = squad.warpRank();
        const int warp_tile_offset = compute_warp_id * warp_tile_size;
        RingCursorT key_ring;
        for (int pipeline_gen = 0;; ++pipeline_gen, key_ring.advance(key_ring_stages))
        {
          const int slot_id = key_ring.slot;
          wait_parity(&full[slot_id], key_ring.parity);
          const int tile_id = tile_id_buf[slot_id];
          if (tile_id >= num_tiles)
          {
            if (lane_id == 0)
            {
              // consumers wait computed + this warp-tile's staged_warp_tile, so arrive both
              ptx::mbarrier_arrive(&computed[slot_id]);
              ptx::mbarrier_arrive(&staged_warp_tile[slot_id][compute_warp_id]);
            }
            break;
          }
          // slot is ready! compute is FLAGS-ONLY (the disaggregation design): the count chain
          // publishes at flag-fold and never waits on a single value read
          const int tile_len  = (int) min((OffT) tile_size, num_items - (OffT) tile_id * tile_size);
          int local_run_count = 0;
          unsigned my_flags;
          if (keys_staged)
          {
            const KeyT* key_buf = tile_buf + (size_t) slot_id * slot_stride + slot_pad;
            my_flags            = compute_head_flags<items_per_thread, false>(
              key_buf, warp_tile_offset, tile_len, tile_id, lane_id, skip_elems);
          }
          else
          {
            // vvv regressed case: we load compute flags straight from global vvv
            const KeyT* key_buf = d_keys + (size_t) tile_id * tile_size;
            my_flags =
              compute_head_flags<items_per_thread, true>(key_buf, warp_tile_offset, tile_len, tile_id, lane_id, 0);
            // ^^^ regressed case ^^^
          }
          local_run_count = __reduce_add_sync(full_mask, __popc(my_flags));
          // last head in this warp tile (feeds the tile's open_len publish): highest nonempty
          // word, then its highest bit -- the ballot is warp-uniform so the branch is converged
          const unsigned nonempty_chunk_mask = __ballot_sync(full_mask, my_flags != 0u);
          int warp_last_head                 = -1;
          if (nonempty_chunk_mask)
          {
            const int last_chunk           = 31 - __clz(nonempty_chunk_mask);
            const unsigned last_chunk_mask = __shfl_sync(full_mask, my_flags, last_chunk);
            warp_last_head                 = warp_tile_offset + last_chunk * 32 + 31 - __clz(last_chunk_mask);
          }
          if (lane_id == 0)
          {
            warp_run_counts[slot_id][compute_warp_id] = local_run_count;
            warp_last_heads[slot_id][compute_warp_id] = warp_last_head;
            ptx::mbarrier_arrive(&computed[slot_id]); // each compute warp arrives
          }
          // warp 0 waits all compute warp arrivals so that every warp's results are visible,
          // then folds and publishes the tile's COUNT state
          if (compute_warp_id == 0)
          {
            wait_parity(&computed[slot_id], key_ring.parity);
            reduce_and_publish_tile_state<compute_warps>(
              count_states, tile_id, tile_len, warp_run_counts[slot_id], warp_last_heads[slot_id], lane_id);
          }
          // no position staging: consumers decode run boundaries from the flag words. They also
          // PERSIST to global: they are the whole run structure the value pass needs (n/8 bytes
          // of temp, the documented cost of the two-pass design)
          head_flag_buf[slot_id][compute_warp_id * 32 + lane_id]                                 = my_flags;
          d_flag_words[(size_t) tile_id * (compute_warps * 32) + compute_warp_id * 32 + lane_id] = my_flags;
          const unsigned nonempty_words = __ballot_sync(full_mask, my_flags != 0);
          if (lane_id == 0)
          {
            word_mask[slot_id][compute_warp_id] = nonempty_words;
          }
          __syncwarp();
          if (lane_id == 0)
          {
            ptx::mbarrier_arrive(&staged_warp_tile[slot_id][compute_warp_id]); // this warp-tile's flags/positions ready
          }
        }
      }
      // if you are poll
      else if (squad == squadPoll)
      {
        int last_seen_tile_id             = 0;
        OffT last_seen_prefix_run_count   = 0;
        OffT last_seen_prefix_open_length = 0;
        int poll_dense_mode               = 1;
        RingCursorT key_ring;
        for (int pipeline_gen = 0;; ++pipeline_gen, key_ring.advance(key_ring_stages))
        {
          const int slot_id = key_ring.slot;
          wait_parity(&full[slot_id], key_ring.parity);
          const int tile_id = tile_id_buf[slot_id];
          if (tile_id >= num_tiles)
          {
            if (lane_id == 0)
            {
              ptx::mbarrier_arrive(&prefixed[slot_id]);
            }
            break;
          }
          OffT curr_prefix_run_count;
          OffT curr_prefix_open_length;
          poll_and_fold<PolicySelector>(
            count_states,
            tile_id,
            last_seen_tile_id,
            last_seen_prefix_run_count,
            last_seen_prefix_open_length,
            lane_id,
            poll_dense_mode,
            curr_prefix_run_count,
            curr_prefix_open_length);
          __syncwarp();
          if (lane_id == 0)
          {
            prefix_packed[slot_id] = PrefixT::pack(curr_prefix_run_count, curr_prefix_open_length);
            ptx::mbarrier_arrive(&prefixed[slot_id]); // prefix ready, key stores + value warps may proceed
          }
        }
      }
      // if you are a KEY-STORE warp: drain keys only, so the key ring recycles at key-pass speed
      else if (squad == squadKeyStore)
      {
        const int key_warp_idx = squad.warpRank();
        RingCursorT key_ring;
        for (int pipeline_gen = 0;; ++pipeline_gen, key_ring.advance(key_ring_stages))
        {
          const int slot_id = key_ring.slot;
          wait_parity(&computed[slot_id], key_ring.parity);
          const int tile_id = tile_id_buf[slot_id];
          if (tile_id >= num_tiles)
          {
            if (lane_id == 0)
            {
              ptx::mbarrier_arrive(&empty[slot_id]);
            }
            break;
          }
          const auto [lane_warp_tile_run_count, lane_runs_before_warp_tile] =
            scan_warp_tile_run_counts<compute_warps>(warp_run_counts[slot_id], lane_id);
          wait_parity(&prefixed[slot_id], key_ring.parity);
          const OffT curr_prefix_run_count = prefix_packed[slot_id].run_count();
          // the count-side epilogue (total run count) rides with the key squad -- it is pure
          // count data and must not wait for any value work
          const int tile_total_runs =
            __shfl_sync(full_mask, lane_runs_before_warp_tile + lane_warp_tile_run_count, compute_warps - 1);
          if (key_warp_idx == 0 && lane_id == 0)
          {
            // the packed word carries the run prefix AND the last headed tile: everything the
            // value pass + cleanup need; neither runs any lookahead of its own
            d_tile_prefix[tile_id] = prefix_packed[slot_id];
            if (tile_id == num_tiles - 1)
            {
              *d_num_runs = (NumRunsT) (curr_prefix_run_count + tile_total_runs);
            }
          }
          // the drain reads keys via LDS only when the pointer is compile-time provably SMEM: the
          // old runtime staged/global ternary forced generic LD.E on every key read (SASS
          // receipt: the flat ~36ns/word KDrain at every density)
          auto drain_tiles = [&](auto staged_tag) _CCCL_FORCEINLINE_LAMBDA {
            constexpr bool wt_keys_staged = decltype(staged_tag)::value;
            const KeyT* tile_keys         = wt_keys_staged ? tile_buf + (size_t) slot_id * slot_stride + slot_pad
                                                           : d_keys + (size_t) tile_id * tile_size;
            const int key_skip            = wt_keys_staged ? skip_elems : 0;
            for (int warp_tile_id = key_warp_idx; warp_tile_id < compute_warps; warp_tile_id += key_store_warps)
            {
              const int warp_tile_run_count   = __shfl_sync(full_mask, lane_warp_tile_run_count, warp_tile_id);
              const int runs_before_warp_tile = __shfl_sync(full_mask, lane_runs_before_warp_tile, warp_tile_id);
              if (warp_tile_run_count == 0)
              {
                continue;
              }
              wait_parity(&staged_warp_tile[slot_id][warp_tile_id], key_ring.parity);
              const OffT global_runs_before_warp_tile = curr_prefix_run_count + runs_before_warp_tile;
              const int warp_tile_offset              = warp_tile_id * warp_tile_size;
#ifdef K_DRAIN_THRESH
              constexpr int word_serial_threshold = K_DRAIN_THRESH;
#else
            constexpr int word_serial_threshold = 256;
#endif
              if (warp_tile_run_count >= staging_threshold && warp_tile_run_count <= word_serial_threshold)
              {
                // word-serial band (profiler receipt: KDrain flat 1.38us/gen at seg4-16 = the broadcast
                // band's fixed 2-shuffles-per-word loop on the MIO pipe): each lane owns its own
                // flag word and extracts its runs serially -- no shuffles, no sync collectives, so
                // the divergence is safe; one smem read + one store per RUN
                const unsigned my_word = head_flag_buf[slot_id][warp_tile_id * 32 + lane_id];
                const int my_popc      = __popc(my_word);
                typename WarpScan<int>::TempStorage warp_scan_storage;
                int word_scan;
                WarpScan<int>(warp_scan_storage).InclusiveSum(my_popc, word_scan);
                int rank     = word_scan - my_popc;
                unsigned rem = my_word;
                while (rem != 0)
                {
                  const int bit = __ffs(rem) - 1;
                  rem &= rem - 1;
                  d_unique[global_runs_before_warp_tile + rank] =
                    tile_keys[warp_tile_offset + lane_id * 32 + bit + key_skip];
                  ++rank;
                }
              }
              else if (warp_tile_run_count >= staging_threshold)
              {
                // broadcast band: replay the flag words IN ORDER from smem (same-address LDS is a
                // free 32-lane broadcast) and emit at head lanes; the word's output base is a
                // warp-uniform running sum, so the loop has NO shuffles at all. Actives within an
                // iteration write consecutive unique slots = coalesced.
                const unsigned upto_l = (lane_id == 31) ? 0xffffffffu : ((2u << lane_id) - 1);
                int word_base         = 0;
#pragma unroll
                for (int iter = 0; iter < items_per_thread; ++iter)
                {
                  const unsigned w = head_flag_buf[slot_id][warp_tile_id * 32 + iter];
                  if ((w >> lane_id) & 1u)
                  {
                    const int run_idx                                = word_base + __popc(w & upto_l) - 1;
                    const int loc                                    = warp_tile_offset + iter * 32 + lane_id;
                    d_unique[global_runs_before_warp_tile + run_idx] = tile_keys[loc + key_skip];
                  }
                  word_base += __popc(w);
                }
              }
              else
              {
                // sparse band: lane-parallel masked word-serial. Compute published which words
                // hold heads (one ballot, outside the scan loop); lane i owns the i-th LIVE word
                // (<32 runs => <32 live words, always one round), so all live words load in ONE
                // parallel LDS round -- the serial-walk form paid one exposed LDS latency per
                // word and regressed seg64 2x (receipted). Rank base via warp scan; each lane
                // extracts its own word's heads serially (no sync collectives, divergence safe).
                const unsigned live_mask = word_mask[slot_id][warp_tile_id];
                const int num_live       = __popc(live_mask);
                if (num_live <= 4)
                {
                  // near-empty sub-band: at ~2 live words the serial replay's couple of exposed
                  // LDS latencies beat the lane-parallel form's warp-scan setup (B200 receipt:
                  // 193 vs 223us at seg1024); the serial form's loss starts at ~8 live words
                  const unsigned upto_l = (lane_id == 31) ? 0xffffffffu : ((2u << lane_id) - 1);
                  unsigned live         = live_mask;
                  int word_base         = 0;
                  while (live != 0)
                  {
                    const int iter = __ffs(live) - 1;
                    live &= live - 1;
                    const unsigned w = head_flag_buf[slot_id][warp_tile_id * 32 + iter];
                    if ((w >> lane_id) & 1u)
                    {
                      const int run_idx                                = word_base + __popc(w & upto_l) - 1;
                      const int loc                                    = warp_tile_offset + iter * 32 + lane_id;
                      d_unique[global_runs_before_warp_tile + run_idx] = tile_keys[loc + key_skip];
                    }
                    word_base += __popc(w);
                  }
                  continue;
                }
                unsigned my_word = 0;
                int my_word_idx  = 0;
                if (lane_id < num_live)
                {
                  my_word_idx = (int) __fns(live_mask, 0, lane_id + 1);
                  my_word     = head_flag_buf[slot_id][warp_tile_id * 32 + my_word_idx];
                }
                const int my_popc = __popc(my_word);
                typename WarpScan<int>::TempStorage warp_scan_storage;
                int word_scan;
                WarpScan<int>(warp_scan_storage).InclusiveSum(my_popc, word_scan);
                int rank     = word_scan - my_popc;
                unsigned rem = my_word;
                while (rem != 0)
                {
                  const int bit = __ffs(rem) - 1;
                  rem &= rem - 1;
                  d_unique[global_runs_before_warp_tile + rank] =
                    tile_keys[warp_tile_offset + my_word_idx * 32 + bit + key_skip];
                  ++rank;
                }
              }
            }
          };
          if (keys_staged)
          {
            drain_tiles(::cuda::std::true_type{});
          }
          else
          {
            drain_tiles(::cuda::std::false_type{});
          }
          __syncwarp();
          if (lane_id == 0)
          {
            // keys drained: the ring recycles without waiting a single value operation
            ptx::mbarrier_arrive(&empty[slot_id]);
          }
        }
      }
    });
}

template <class PolicySelector, class StateT>
_CCCL_KERNEL_ATTRIBUTES void DeviceReduceByKeyLookaheadInitKernel(StateT* states, ::cuda::std::int64_t n_states)
{
  const ::cuda::std::int64_t i = (::cuda::std::int64_t) blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_states)
  {
    states[i] = StateT{};
  }
}

// THE VALUE PASS: pipeline-free. One 8-warp block per tile; the persisted flag words are the
// whole run structure, the persisted tile prefix is the whole cross-tile coordination. No rings,
// no barriers beyond __syncthreads, no lookahead, no work stealing -- latency hides behind plain
// occupancy. Emits every within-tile-closed aggregate + the boundary record for the cleanup pass.
template <typename PolicySelector, class ValueT, class OffT, class ReductionOpT = ::cuda::std::plus<>>
#ifdef RBK_VBLOCKS
__launch_bounds__(current_policy<PolicySelector>().lookahead.compute_warps * 32, RBK_VBLOCKS)
#else
__launch_bounds__(current_policy<PolicySelector>().lookahead.compute_warps * 32, 6)
#endif
  _CCCL_KERNEL_ATTRIBUTES void DeviceReduceByKeyLookaheadValueKernel(
    const ValueT* __restrict__ d_values,
    ValueT* __restrict__ d_aggregates,
    const unsigned* __restrict__ d_flag_words,
    const PrefixT<OffT>* __restrict__ d_tile_prefix,
    TileValueRecordT<ValueT>* __restrict__ value_records,
    OffT num_items,
    int num_tiles,
    ReductionOpT reduction_op = {})
{
  {
    static constexpr RleLookaheadPolicy policy = current_policy<PolicySelector>().lookahead;
    constexpr int items_per_thread             = policy.items_per_thread;
    constexpr int compute_warps                = policy.compute_warps;
    constexpr int warp_tile_size               = policy.warp_tile_size();
    constexpr int tile_size                    = policy.tile_size();
#ifdef RBK_STREAM_DIV
    constexpr int stream_threshold = warp_tile_size / RBK_STREAM_DIV;
#else
    // B200 receipt: /4 pulls the seg4 band into the stream and costs nothing anywhere else
    constexpr int stream_threshold = warp_tile_size / 4;
#endif
    __shared__ int wt_counts[compute_warps];
    __shared__ ValueT wt_leads[compute_warps];
    __shared__ ValueT wt_tails[compute_warps];
    __shared__ int wt_leads_has[compute_warps];
    __shared__ int wt_tails_has[compute_warps];
    __shared__ ValueT wt_row_carry[compute_warps][32];

    const int tile_id = (int) blockIdx.x;
    if (tile_id >= num_tiles)
    {
      return;
    }
    // STAGED mode (profiler receipt: the bands run 3.7x faster per warp tile from smem, and 6-block
    // occupancy is the latency pipeline): one bulk TMA pulls the tile's values AND flag words into
    // this block's smem before the bands run. Unstaged (misaligned values base): the old global
    // path with the fire-and-forget L2 prefetch (v6b receipt).
    extern __shared__ char v_smem_raw[];
    ValueT* const staged_vals = (ValueT*) v_smem_raw;
    __shared__ ::cuda::std::uint64_t staged_bar;
#ifdef K_VSTAGE_T
    constexpr int stage_run_threshold = K_VSTAGE_T;
#else
    constexpr int stage_run_threshold = 512;
#endif
    const int stage_len = (int) min((OffT) tile_size, num_items - (OffT) tile_id * tile_size);
    if (threadIdx.x == 0)
    {
      {
        ptx::mbarrier_init(&staged_bar, 1);
        ptx::fence_proxy_async(ptx::space_shared);
      }
      if (stage_run_threshold == 0)
      {
        // always-stage shape: the TMA IS the fetch; issue it NOW so setup rides under the flight
        const unsigned vbytes = (unsigned) ((size_t) stage_len * sizeof(ValueT));
        const unsigned vspan  = (vbytes + 15u) & ~15u;
        ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &staged_bar, vspan);
        ptx::cp_async_bulk_ignore_oob(
          ptx::space_shared,
          ptx::space_global,
          staged_vals,
          d_values + (size_t) tile_id * tile_size,
          vspan,
          0u,
          vspan - vbytes,
          &staged_bar);
      }
      else
      {
        // routed/unstaged shapes: L2 prefetch (v6b receipt)
        const char* vbase        = (const char*) (d_values + (size_t) tile_id * tile_size);
        const char* vbase16      = (const char*) ((size_t) vbase & ~(size_t) 15);
        const unsigned vpf_bytes = (unsigned) (((size_t) stage_len * sizeof(ValueT)) & ~(size_t) 15);
        if (vpf_bytes > 0)
        {
          asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(vbase16), "r"(vpf_bytes) : "memory");
        }
      }
      const char* fbase = (const char*) (d_flag_words + (size_t) tile_id * (compute_warps * 32));
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(fbase),
                   "r"((unsigned) (compute_warps * 32 * sizeof(unsigned)))
                   : "memory");
    }
    __syncthreads(); // staged_bar init is visible to every warp past this point
    // trace STEADY-STATE blocks, not the first wave: tiles < 8 run on an uncontended GPU and
    // understate every range by ~1.4x (background-agent receipt: block lifetimes 11.4us steady vs
    // 7.3 first-wave; effective concurrency 5.43/SM)
    const int wt                  = (int) (threadIdx.x >> 5);
    const int lane_id             = (int) (threadIdx.x & 31);
    const int tile_len            = (int) min((OffT) tile_size, num_items - (OffT) tile_id * tile_size);
    const unsigned my_word        = d_flag_words[(size_t) tile_id * (compute_warps * 32) + wt * 32 + lane_id];
    const int warp_tile_run_count = __reduce_add_sync(full_mask, __popc(my_word));
    if (lane_id == 0)
    {
      wt_counts[wt] = warp_tile_run_count;
    }
    __syncthreads();
    const auto [lane_warp_tile_run_count, lane_runs_before_warp_tile] =
      scan_warp_tile_run_counts<compute_warps>(wt_counts, lane_id);
    const int runs_before_warp_tile         = __shfl_sync(full_mask, lane_runs_before_warp_tile, wt);
    const OffT curr_prefix_run_count        = d_tile_prefix[tile_id].run_count();
    const OffT global_runs_before_warp_tile = curr_prefix_run_count + runs_before_warp_tile;
    const int warp_tile_offset              = wt * warp_tile_size;
    const int wt_end                        = min(warp_tile_offset + warp_tile_size, tile_len);
    // boundary sums + the banded within-warp-tile emission (the receipted forms). The tag lambda
    // keeps the smem pointer compile-time provable (the LD.E-vs-LDS receipt from the keys drain)
    ValueT wt_lead{};
    ValueT wt_tail{};
    auto emit_bands = [&](const ValueT* tile_vals, auto staged_tag) _CCCL_FORCEINLINE_LAMBDA {
      constexpr bool from_smem = decltype(staged_tag)::value;
      if (warp_tile_run_count == 0 || (from_smem && warp_tile_run_count < 4))
      {
        // whole-warp span band: a 2-run warp tile paid the full rotate + ~390-inst walk (SASS
        // census); two coalesced span sums cost ~60. All 32 lanes walk each span together.
        // run_count is warp-uniform, so the head-free case legally skips the decoder and does
        // its one fold (the unconditional decoder cost +2.1% at 2^12: B200 receipt)
        if (warp_tile_run_count == 0)
        {
          wt_lead = warp_span_fold(tile_vals, warp_tile_offset, wt_end, lane_id, reduction_op);
          wt_tail = wt_lead; // head-free: the whole warp tile leads AND trails
        }
        else
        {
          const HeadFlagDecodeT dec(my_word, lane_id);
          const RunSpanT lane_run = dec.decode_run(lane_id < warp_tile_run_count ? lane_id : 0);
          for (int run_idx = 0; run_idx + 1 < warp_tile_run_count; ++run_idx)
          {
            const int head = __shfl_sync(full_mask, lane_run.head_pos_in_warp_tile, run_idx);
            const int next = __shfl_sync(full_mask, lane_run.next_head_pos, run_idx);
            const ValueT agg =
              warp_span_fold(tile_vals, warp_tile_offset + head, warp_tile_offset + next, lane_id, reduction_op);
            if (lane_id == 0)
            {
              d_aggregates[global_runs_before_warp_tile + run_idx] = agg;
            }
          }
          const int first_head = dec.decode_run(0).head_pos_in_warp_tile;
          const int last_head  = dec.decode_run(warp_tile_run_count - 1).head_pos_in_warp_tile;
          wt_lead = warp_span_fold(tile_vals, warp_tile_offset, warp_tile_offset + first_head, lane_id, reduction_op);
          wt_tail = warp_span_fold(tile_vals, warp_tile_offset + last_head, wt_end, lane_id, reduction_op);
        }
      }
      else if (from_smem && sizeof(ValueT) == 4 && items_per_thread == 32 && warp_tile_run_count < stream_threshold)
      {
        // (4-byte values, 32x32 FULL warp tiles only: the rotate/loads cast through uint4;
        // partial warp tiles take the one-element form)
        if constexpr (from_smem && sizeof(ValueT) == 4 && items_per_thread == 32)
        {
          // lead fold FIRST: <32> rotates the rows in place, so linear reads must precede it
          const HeadFlagDecodeT dec(my_word, lane_id);
          const RunSpanT first_run = dec.decode_run(0);
          wt_lead                  = warp_span_fold(
            tile_vals, warp_tile_offset, warp_tile_offset + first_run.head_pos_in_warp_tile, lane_id, reduction_op);
          stream_band<items_per_thread, 32, true>(
            d_aggregates,
            tile_vals,
            my_word,
            global_runs_before_warp_tile,
            warp_tile_offset,
            tile_len,
            lane_id,
            reduction_op,
            wt_tail);
        }
      }
      else if (warp_tile_run_count >= stream_threshold || sizeof(ValueT) != 4 || items_per_thread != 32)
      {
        // stream density router (B200 receipts): quad wins to ~half density (seg4 -9.4%), pair
        // from there to ~3/4 (seg2, quad's interior-emission cost +14% there), the adaptive row
        // stream at near-all-heads (seg1). Any 4-byte type, any associative op (the plus<float>
        // asm inside is instruction selection; partial warp tiles take the row stream)
        int stream_form = 0; // 0 = row
        if constexpr (from_smem && sizeof(ValueT) == 4 && items_per_thread == 32)
        {
          stream_form = (warp_tile_run_count < warp_tile_size / 2) ? 2
                      : (warp_tile_run_count < (7 * warp_tile_size) / 8)
                        ? 1
                        : 0;
        }
        if (stream_form == 2)
        {
          if constexpr (from_smem && sizeof(ValueT) == 4 && items_per_thread == 32)
          {
            stream_band<items_per_thread, 4, true>(
              d_aggregates,
              tile_vals,
              my_word,
              global_runs_before_warp_tile,
              warp_tile_offset,
              tile_len,
              lane_id,
              reduction_op,
              wt_tail);
          }
        }
        else if (stream_form == 1)
        {
          if constexpr (from_smem && sizeof(ValueT) == 4 && items_per_thread == 32)
          {
            stream_band<items_per_thread, 2, true>(
              d_aggregates,
              tile_vals,
              my_word,
              global_runs_before_warp_tile,
              warp_tile_offset,
              tile_len,
              lane_id,
              reduction_op,
              wt_tail);
          }
        }
        else
        {
          stream_band<items_per_thread, 1, true>(
            d_aggregates,
            tile_vals,
            my_word,
            global_runs_before_warp_tile,
            warp_tile_offset,
            tile_len,
            lane_id,
            reduction_op,
            wt_tail);
        }
        const HeadFlagDecodeT dec(my_word, lane_id);
        const RunSpanT first_run = dec.decode_run(0);
        wt_lead                  = warp_span_fold(
          tile_vals, warp_tile_offset, warp_tile_offset + first_run.head_pos_in_warp_tile, lane_id, reduction_op);
      }
      // (no final else: the hot path is staged-full only; every other shape bailed to the cold
      // outline above)
    };
    const int tile_total_runs =
      __shfl_sync(full_mask, lane_runs_before_warp_tile + lane_warp_tile_run_count, compute_warps - 1);
    // TILE-level hot gate, provably warp-uniform (kernel param + blockIdx-derived): a cold CALL
    // inside the band branch chain broke ptxas' convergence proof and dual-compiled every
    // collective band (WARPSYNC/ENDCOLLECTIVE safe copies = the +30% executed-instruction tax)
    if (tile_len == tile_size && tile_total_runs >= stage_run_threshold)
    {
      if (stage_run_threshold > 0 && threadIdx.x == 0)
      {
        const unsigned vbytes = (unsigned) ((size_t) stage_len * sizeof(ValueT));
        const unsigned vspan  = (vbytes + 15u) & ~15u;
        ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &staged_bar, vspan);
        ptx::cp_async_bulk_ignore_oob(
          ptx::space_shared,
          ptx::space_global,
          staged_vals,
          d_values + (size_t) tile_id * tile_size,
          vspan,
          0u,
          vspan - vbytes,
          &staged_bar);
      }
      wait_parity(&staged_bar, 0);
      emit_bands(staged_vals, ::cuda::std::true_type{});
    }
    else
    {
      // the LAST (short) tile, still STAGED (the ignore-oob TMA clamps the copy): row stream at
      // every density with tile_len validity, from smem. Block-uniform branch: no dual-compile.
      wait_parity(&staged_bar, 0);
      if (warp_tile_run_count == 0)
      {
        wt_lead = warp_span_fold(staged_vals, warp_tile_offset, wt_end, lane_id, reduction_op);
        wt_tail = wt_lead;
      }
      else
      {
        stream_band<items_per_thread, 1, false>(
          d_aggregates,
          staged_vals,
          my_word,
          global_runs_before_warp_tile,
          warp_tile_offset,
          tile_len,
          lane_id,
          reduction_op,
          wt_tail);
        const HeadFlagDecodeT dec(my_word, lane_id);
        const RunSpanT first_run = dec.decode_run(0);
        wt_lead                  = warp_span_fold(
          staged_vals, warp_tile_offset, warp_tile_offset + first_run.head_pos_in_warp_tile, lane_id, reduction_op);
      }
    }
    if (lane_id == 0)
    {
      wt_leads[wt] = wt_lead;
      wt_tails[wt] = wt_tail;
      // structural has: the tail is empty only on an element-free warp tile; the lead is empty
      // when the warp tile's first element is a head (lane 0 owns iter 0's ballot)
      const int wt_nonempty = wt_end > warp_tile_offset;
      wt_tails_has[wt]      = wt_nonempty;
      wt_leads_has[wt]      = (warp_tile_run_count == 0) ? wt_nonempty : (int) !(my_word & 1u);
    }
    __syncthreads();
    // warp 0: the record + the within-tile boundary closes (aggregate chains over the wt sums)
    if (wt == 0)
    {
      const int lane_count                    = (lane_id < compute_warps) ? wt_counts[lane_id] : 0;
      const unsigned nonempty_warp_tiles_mask = __ballot_sync(full_mask, lane_count > 0);
      const bool any_head                     = (nonempty_warp_tiles_mask != 0);
      const bool is_last_tile                 = (tile_id == num_tiles - 1);
      {
        const int last_headed  = any_head ? (31 - __clz(nonempty_warp_tiles_mask)) : 0;
        const int first_headed = any_head ? (__ffs(nonempty_warp_tiles_mask) - 1) : compute_warps;
        // element-holding warp tiles are a PREFIX (trailing wts past tile_len are empty), so
        // both folds are contiguous smem ranges: lane i takes the i-th element; cub::WarpReduce
        // handles the valid prefix (order-preserving ascending tree, result on lane 0)
        const int n_wt_nonempty = min(compute_warps, (tile_len - 1) / warp_tile_size + 1);
        const int n_open        = n_wt_nonempty - last_headed; // >= 1 on a live tile
        const int n_lead        = min(first_headed, n_wt_nonempty);
        typename WarpReduce<ValueT>::TempStorage fold_storage;
        ValueT open_agg =
          WarpReduce<ValueT>(fold_storage)
            .Reduce((lane_id < n_open) ? wt_tails[last_headed + lane_id] : ValueT{}, reduction_op, n_open);
        ValueT lead_agg = WarpReduce<ValueT>(fold_storage)
                            .Reduce((lane_id < n_lead) ? wt_tails[lane_id] : ValueT{}, reduction_op, n_lead);
        if (lane_id == 0)
        {
          int lead_has = n_lead > 0;
          if (any_head && wt_leads_has[first_headed])
          {
            lead_agg = lead_has ? reduction_op(lead_agg, wt_leads[first_headed]) : wt_leads[first_headed];
            lead_has = 1;
          }
          TileValueRecordT<ValueT> rec;
          rec.open_agg           = open_agg; // never empty on a live tile: it always contains its last head
          rec.lead_agg           = lead_agg;
          value_records[tile_id] = rec;
        }
      }
      if (lane_id < compute_warps && lane_count > 0)
      {
        const unsigned later = nonempty_warp_tiles_mask >> (lane_id + 1);
        // this warp's scan already gave lane i warp-tile i's runs-before
        const OffT last_run_global_idx = curr_prefix_run_count + lane_runs_before_warp_tile + lane_count - 1;
        if (later)
        {
          const int next_headed = lane_id + 1 + __ffs(later) - 1;
          // a headed warp tile's own tail always has (it contains its last head)
          ValueT closing = wt_tails[lane_id];
          for (int w2 = lane_id + 1; w2 < next_headed; ++w2)
          {
            if (wt_tails_has[w2])
            {
              closing = reduction_op(closing, wt_tails[w2]); // head-free middles: whole folds
            }
          }
          if (wt_leads_has[next_headed])
          {
            closing = reduction_op(closing, wt_leads[next_headed]);
          }
          d_aggregates[last_run_global_idx] = closing;
        }
        else if (is_last_tile)
        {
          ValueT closing = wt_tails[lane_id];
          for (int w2 = lane_id + 1; w2 < compute_warps; ++w2)
          {
            if (wt_tails_has[w2])
            {
              closing = reduction_op(closing, wt_tails[w2]);
            }
          }
          d_aggregates[last_run_global_idx] = closing;
        }
        // else: open into the next tile -- the cleanup pass closes it
      }
    }
  }
}

// boundary cleanup: one warp per tile that owes an entering-run close. Runs AFTER the main
// kernel (the launch boundary synchronizes), reads plain records, writes each cross-tile run's
// aggregate = open(from) + whole-sums of the head-free tiles between + lead(this). Every window
// is disjoint, sums fold in fixed order (deterministic), total work is O(num_tiles) amortized.
// The record holds only the two value sums; the bookkeeping is derived here: a tile owes a
// close iff runs exist before it AND it closes them (own head, or input end); the destination
// is the entering run's rank; the window start is the prefix's last-tile-with-runs; the lead
// span is empty iff the tile's first element is a head (its key differs from its predecessor).
template <class KeyT, class ValueT, class OffT, class ReductionOpT = ::cuda::std::plus<>>
_CCCL_KERNEL_ATTRIBUTES void DeviceReduceByKeyLookaheadCleanupKernel(
  ValueT* d_aggregates,
  const TileValueRecordT<ValueT>* value_records,
  const PrefixT<OffT>* d_tile_prefix,
  const KeyT* d_keys,
  int tile_size,
  int num_tiles,
  ReductionOpT reduction_op = {})
{
  const int warp_global = (int) ((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
  const int lane_id     = (int) (threadIdx.x & 31);
  if (warp_global >= num_tiles)
  {
    return;
  }
  const OffT runs_before = d_tile_prefix[warp_global].run_count();
  if (runs_before == 0) // no entering run (covers tile 0)
  {
    return;
  }
  const bool is_last_tile = warp_global == num_tiles - 1;
  if (!is_last_tile && d_tile_prefix[warp_global + 1].run_count() == runs_before)
  {
    return; // head-free middle tile: the run passes through, a later tile closes it
  }
  // window start: the entering run's head sits at global position t*tile_size - open_len
  // (open_len includes the head), and every tile between it and t is full, so its tile index
  // is plain division
  const OffT open_len = d_tile_prefix[warp_global].open_len();
  const int from      = (int) (((OffT) warp_global * tile_size - open_len) / tile_size);
  ValueT carry{};
  for (int base = from; base < warp_global; base += 32)
  {
    // open_agg of the window start is its tail chain; the head-free tiles after it contribute
    // their whole-tile folds (== their open_agg). Valid lanes are a contiguous prefix, so the
    // lane-ascending shfl_down tree (= tile-ascending ordered fold) rides a count predicate
    // (CUB WarpReduceShfl shape).
    const int n_valid = min(32, warp_global - base);
    const int t       = base + lane_id;
    typename WarpReduce<ValueT>::TempStorage storage;
    const ValueT part = WarpReduce<ValueT>(storage).Reduce(
      (lane_id < n_valid) ? value_records[t].open_agg : ValueT{}, reduction_op, n_valid);
    // part is valid on lane 0 only; so is carry -- lane 0 does the final store
    carry = (base == from) ? part : reduction_op(carry, part); // the window is never empty
  }
  if (lane_id == 0)
  {
    // lead span empty iff the tile's first element is a head: compare it with its predecessor
    const int lead_has = (d_keys[(size_t) warp_global * tile_size] == d_keys[(size_t) warp_global * tile_size - 1]);
    d_aggregates[runs_before - 1] = lead_has ? reduction_op(carry, value_records[warp_global].lead_agg) : carry;
  }
}

template <typename PolicySelector>
[[nodiscard]] _CCCL_HOST_DEVICE_API _CCCL_CONSTEVAL int get_device_reduce_by_key_lookahead_launch_bounds() noexcept
{
  return num_total_threads(current_policy<PolicySelector>().lookahead);
}

// need a variable template for clang in CUDA mode to avoid:
// error: 'launch_bounds' attribute requires parameter 0 to be an integer constant
template <typename PolicySelector>
inline constexpr int device_reduce_by_key_lookahead_launch_bounds =
  get_device_reduce_by_key_lookahead_launch_bounds<PolicySelector>();

template <typename PolicySelector, class KeyT, class ValueT, class NumRunsT, class OffT>
__launch_bounds__(device_reduce_by_key_lookahead_launch_bounds<PolicySelector>, 1)
  _CCCL_KERNEL_ATTRIBUTES void DeviceReduceByKeyLookaheadKernel(
    const KeyT* __restrict__ d_keys,
    const ValueT* __restrict__ d_values,
    KeyT* __restrict__ d_unique,
    ValueT* __restrict__ d_aggregates,
    NumRunsT* __restrict__ d_num_runs,
    TilePartialStateT* count_states,
    unsigned* d_flag_words,
    PrefixT<OffT>* d_tile_prefix,
    OffT num_items,
    int num_tiles,
    int key_ring_stages,
    int pos_ring_stages,
    bool keys_staged)
{
  NV_IF_TARGET(
    NV_PROVIDES_SM_100,
    (device_reduce_by_key_lookahead_body<PolicySelector>(
       d_keys,
       d_values,
       d_unique,
       d_aggregates,
       d_num_runs,
       count_states,
       d_flag_words,
       d_tile_prefix,
       num_items,
       num_tiles,
       key_ring_stages,
       pos_ring_stages,
       keys_staged);))
}
} // namespace detail::reduce_by_key

CUB_NAMESPACE_END
