// Correctness gate: persistent_rle vs cub::DeviceRunLengthEncode across regimes.
// Covers: single tile, partial tail tiles, dense (seg=1), mid, long runs spanning many tiles
// (head-free tiles + cross-tile open-run closing), and the full bench size.
// d_keys is allocated padded to a tile multiple, matching the kernel's input contract.
#include <cub/device/device_run_length_encode.cuh>

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "rle_dispatch.cuh"

#define CHECK_CUDA(call)                                                                   \
  do                                                                                       \
  {                                                                                        \
    cudaError_t e_ = (call);                                                               \
    if (e_ != cudaSuccess)                                                                 \
    {                                                                                      \
      std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
      std::exit(2);                                                                        \
    }                                                                                      \
  } while (0)

// segment values drawn as ints then cast to KeyT; adjacency-distinctness is enforced POST-cast so
// narrow key types (int8) can't accidentally merge adjacent segments through wraparound.
static std::vector<KeyT> gen_keys(long long n, int max_seg, unsigned seed)
{
  std::vector<KeyT> k((size_t) n);
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> seg(1, max_seg), kd(0, 1000000);
  long long i = 0;
  KeyT prev   = KeyT(-1);
  while (i < n)
  {
    int run = seg(rng);
    KeyT v  = KeyT(kd(rng));
    for (int tries = 0; v == prev && tries < 8; ++tries)
    {
      v = KeyT(kd(rng));
    }
    prev        = v;
    long long e = std::min<long long>(i + run, n);
    for (; i < e; ++i)
    {
      k[i] = v;
    }
  }
  return k;
}

template <class T>
static double dbg(T v)
{
  if constexpr (cuda::std::is_arithmetic_v<T>)
  {
    return (double) v;
  }
  else
  {
    return 0.0; // complex/other: mismatch positions matter, values don't print
  }
}

static bool run_case(long long n, int max_seg, unsigned seed, bool sampled = false)
{
  const size_t pad = (size_t) n; // EXACT allocation: the bounded-TMA tail must never over-read
  auto h           = gen_keys(n, max_seg, seed);

  KeyT *dk{}, *du{};
  LenT* dc{};
  NumRunsT* dn{};
  void* dtemp{};
  CHECK_CUDA(cudaMalloc(&dk, sizeof(KeyT) * pad));
  CHECK_CUDA(cudaMemset(dk, 0, sizeof(KeyT) * pad));
  CHECK_CUDA(cudaMemcpy(dk, h.data(), sizeof(KeyT) * (size_t) n, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMalloc(&du, sizeof(KeyT) * (size_t) n));
  CHECK_CUDA(cudaMalloc(&dc, sizeof(LenT) * (size_t) n));
  CHECK_CUDA(cudaMalloc(&dn, sizeof(NumRunsT)));
  size_t temp_bytes = 0;
  persistent_rle_encode(nullptr, temp_bytes, dk, du, dc, dn, (OffT) n);
  CHECK_CUDA(cudaMalloc(&dtemp, temp_bytes));
  CHECK_CUDA(cudaMemset(dtemp, 0xAB, temp_bytes)); // garbage: the magic-word path must cold-init

  // reference
  void* tmp   = nullptr;
  size_t tbsz = 0;
  cub::DeviceRunLengthEncode::Encode(tmp, tbsz, dk, du, dc, dn, (OffT) n);
  CHECK_CUDA(cudaMalloc(&tmp, tbsz));
  cub::DeviceRunLengthEncode::Encode(tmp, tbsz, dk, du, dc, dn, (OffT) n);
  CHECK_CUDA(cudaDeviceSynchronize());
  NumRunsT refR_w = -1;
  CHECK_CUDA(cudaMemcpy(&refR_w, dn, sizeof(NumRunsT), cudaMemcpyDeviceToHost));
  const long long refR = (long long) refR_w;
  // sampled mode (huge dense inputs): compare num_runs exactly + a window at each end -- the tail
  // window exercises run indices past 2^31 without 2x-full-output host mirrors
  const long long win   = sampled ? std::min<long long>(refR, 1 << 20) : refR;
  const long long tail0 = refR - win;
  std::vector<KeyT> ref_u((size_t) win), ref_u2((size_t) (sampled ? win : 0));
  std::vector<LenT> ref_c((size_t) win), ref_c2((size_t) (sampled ? win : 0));
  CHECK_CUDA(cudaMemcpy(ref_u.data(), du, sizeof(KeyT) * win, cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(ref_c.data(), dc, sizeof(LenT) * win, cudaMemcpyDeviceToHost));
  if (sampled)
  {
    CHECK_CUDA(cudaMemcpy(ref_u2.data(), du + tail0, sizeof(KeyT) * win, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(ref_c2.data(), dc + tail0, sizeof(LenT) * win, cudaMemcpyDeviceToHost));
  }

  // ours -- repeated launches on the SAME (never re-cleared) tile_state array: round 1 exercises
  // the zeroed-at-alloc path, round 2 exercises gen-tag invalidation of round 1's stale states.
  // outputs are poisoned before each round so stale data can't mask a missing write.
  constexpr int kVerifyRounds = 2;
  bool ok                     = true;
  for (int round = 0; round < kVerifyRounds; ++round)
  {
    CHECK_CUDA(cudaMemset(du, 0xEE, sizeof(KeyT) * (size_t) n));
    CHECK_CUDA(cudaMemset(dc, 0xEE, sizeof(LenT) * (size_t) n));
    CHECK_CUDA(cudaMemset(dn, 0xEE, sizeof(NumRunsT)));
    CHECK_CUDA(persistent_rle_encode(dtemp, temp_bytes, dk, du, dc, dn, (OffT) n, 0));
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    NumRunsT gotR_w = -1;
    CHECK_CUDA(cudaMemcpy(&gotR_w, dn, sizeof(NumRunsT), cudaMemcpyDeviceToHost));
    const long long gotR = (long long) gotR_w;
    if (gotR != refR)
    {
      std::printf("  round %d: num_runs MISMATCH: got %lld ref %lld\n", round, gotR, refR);
      ok = false;
      continue;
    }
    std::vector<KeyT> got_u((size_t) win), got_u2((size_t) (sampled ? win : 0));
    std::vector<LenT> got_c((size_t) win), got_c2((size_t) (sampled ? win : 0));
    CHECK_CUDA(cudaMemcpy(got_u.data(), du, sizeof(KeyT) * win, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(got_c.data(), dc, sizeof(LenT) * win, cudaMemcpyDeviceToHost));
    if (sampled)
    {
      CHECK_CUDA(cudaMemcpy(got_u2.data(), du + tail0, sizeof(KeyT) * win, cudaMemcpyDeviceToHost));
      CHECK_CUDA(cudaMemcpy(got_c2.data(), dc + tail0, sizeof(LenT) * win, cudaMemcpyDeviceToHost));
      for (long long i = 0; i < win; ++i)
      {
        if (!(got_u2[i] == ref_u2[i]) || got_c2[i] != ref_c2[i])
        {
          std::printf("  round %d TAIL run %lld mismatch\n", round, tail0 + i);
          ok = false;
          break;
        }
      }
    }
    int shown = 0;
    for (long long i = 0; i < win && shown < 5; ++i)
    {
      if (!(got_u[i] == ref_u[i]) || got_c[i] != ref_c[i])
      {
        std::printf(
          "  round %d run %lld: got (u=%g,c=%lld) ref (u=%g,c=%lld)\n",
          round,
          i,
          dbg(got_u[i]),
          (long long) got_c[i],
          dbg(ref_u[i]),
          (long long) ref_c[i]);
        ok = false;
        ++shown;
      }
    }
  }
  std::printf("%-8s n=%-12lld max_seg=%-8d runs=%-11lld\n", ok ? "PASS" : "FAIL", n, max_seg, refR);

  cudaFree(tmp);
  cudaFree(dk);
  cudaFree(du);
  cudaFree(dc);
  cudaFree(dn);
  cudaFree(dtemp);
  return ok;
}

int main(int argc, char** argv)
{
  const bool huge = (argc > 1) && argv[1][0] == 'h'; // 2^32-scale cases (wide-offset builds, big GPUs)

  struct Case
  {
    long long n;
    int max_seg;
  };
  const Case cases[] = {
    {200000, 2}, // mid-dispatch band at local tile geometry (caught the mid-tile-count bug)
    {150000, 3}, // mid band, different tile alignment
    {kTileSize, 1}, // single tile, dense
    {kTileSize, 1000000}, // single tile, one run
    {3 * kTileSize, 1}, // few tiles, dense
    {3 * kTileSize + 7, 1}, // partial tail tile, dense
    {3 * kTileSize + 1, 1000000}, // partial tail, open run crossing into tail
    {(1 << 20) + 12345, 2}, // partial tail, mid
    {1 << 22, 4}, // mid
    {1 << 22, 32}, // mid
    {1 << 24, 4096}, // long: head-free tiles
    {(1 << 24) + 8191, 1000000}, // very long runs + partial tail
    {1 << 28, 1}, // full bench size, dense
    {1 << 28, 1048576}, // full bench size, longest regime
  };
  int fails = 0;
  for (const Case& c : cases)
  {
    for (unsigned seed : {1u, 42u})
    {
      fails += run_case(c.n, c.max_seg, seed) ? 0 : 1;
    }
  }
  if (huge)
  {
    static_assert(sizeof(OffT) == 8 || true, "");
    if (sizeof(OffT) < 8)
    {
      std::printf("huge requested but OffT is 32-bit -- skipping\n");
    }
    else
    {
      fails += run_case((1ll << 32) + 12345, 1000000, 7u) ? 0 : 1; // element offsets past 2^31
      fails += run_case((1ll << 32), 1, 7u, /*sampled=*/true) ? 0 : 1; // 4G runs: indices past 2^31
    }
  }
  std::printf(fails ? "*** %d FAILURES ***\n" : "ALL PASS\n", fails);
  return fails ? 1 : 0;
}
