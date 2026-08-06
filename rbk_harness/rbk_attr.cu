// per-pass wall-time attribution: ./rbk_attr <pattern>... where pattern is rN/eN/alt255_256/
// zipf/uN (uN = uniform max_seg N). Prints avg ms of init/keys/values/cleanup per pattern.
#include <cub/detail/choose_offset.cuh>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "rbk_dispatch.cuh"
#include "rbk_keygen.h"

using KeyT   = int;
using ValueT = float;

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

static std::vector<KeyT> pattern_keys(long long n, const std::string& pat)
{
  if (pat == "alt255_256")
  {
    return gen_keys_skew<KeyT>(n, 0, true, 1u);
  }
  if (pat == "zipf")
  {
    return gen_keys_zipf<KeyT>(n, 1u);
  }
  const int v = std::atoi(pat.c_str() + 1);
  if (pat[0] == 'u')
  {
    return gen_keys<KeyT>(n, v, 1u);
  }
  if (pat[0] == 'c')
  {
    return gen_keys_columnar<KeyT>(n, v, 1u);
  }
  return (pat[0] == 'e') ? gen_keys_even<KeyT>(n, v, 1u) : gen_keys_skew<KeyT>(n, v, false, 1u);
}

int main(int argc, char** argv)
{
  using config_t      = rbk_impl::winner_config<KeyT, ValueT>;
  using NumRunsT      = cub::detail::choose_signed_offset_t<int>;
  const long long n   = 1LL << 28;
  constexpr int kWarm = 3, kIters = 20;

  std::printf("%-11s %9s %9s %9s %9s %9s\n", "pattern", "init_ms", "keys_ms", "values_ms", "clean_ms", "total_ms");
  for (int a = 1; a < argc; ++a)
  {
    const std::string pat = argv[a];
    auto h                = pattern_keys(n, pat);
    long long cpuR        = 0;
    for (long long i = 0; i < n; ++i)
    {
      cpuR += (i == 0 || h[i] != h[i - 1]) ? 1 : 0;
    }
    std::vector<ValueT> hv((size_t) n);
    std::mt19937 vrng(2u);
    std::uniform_real_distribution<float> vd(0.0f, 1.0f);
    for (auto& x : hv)
    {
      x = vd(vrng);
    }
    const size_t pad = (size_t) ((n + config_t::kTileSize - 1) / config_t::kTileSize) * config_t::kTileSize;
    KeyT *dk{}, *du{};
    ValueT *dv{}, *da{};
    NumRunsT* dn{};
    void* dtemp{};
    size_t tempb = 0;
    CHECK_CUDA(cudaMalloc(&dk, sizeof(KeyT) * pad));
    CHECK_CUDA(cudaMemset(dk, 0, sizeof(KeyT) * pad));
    CHECK_CUDA(cudaMalloc(&dv, sizeof(ValueT) * pad));
    CHECK_CUDA(cudaMemset(dv, 0, sizeof(ValueT) * pad));
    CHECK_CUDA(cudaMalloc(&du, sizeof(KeyT) * (size_t) n));
    CHECK_CUDA(cudaMalloc(&da, sizeof(ValueT) * (size_t) n));
    CHECK_CUDA(cudaMalloc(&dn, sizeof(NumRunsT)));
    rbk_impl::persistent_rbk_encode<config_t>(nullptr, tempb, dk, dv, du, da, dn, (int) n, 0);
    CHECK_CUDA(cudaMalloc(&dtemp, tempb));
    CHECK_CUDA(cudaMemcpy(dk, h.data(), sizeof(KeyT) * (size_t) n, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dv, hv.data(), sizeof(ValueT) * (size_t) n, cudaMemcpyHostToDevice));

    for (int w = 0; w < kWarm; ++w)
    {
      CHECK_CUDA(rbk_impl::persistent_rbk_encode<config_t>(dtemp, tempb, dk, dv, du, da, dn, (int) n, 0));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    NumRunsT got = 0;
    CHECK_CUDA(cudaMemcpy(&got, dn, sizeof(got), cudaMemcpyDeviceToHost));
    if ((long long) got != cpuR)
    {
      std::printf("*** %s CORRECTNESS FAIL got=%lld cpu=%lld ***\n", pat.c_str(), (long long) got, cpuR);
      std::exit(3);
    }

    float acc[4] = {};
    for (int it = 0; it < kIters; ++it)
    {
      float ms[4] = {};
      CHECK_CUDA(rbk_impl::persistent_rbk_encode<config_t>(
        dtemp, tempb, dk, dv, du, da, dn, (int) n, 0, ::cuda::std::plus<>{}, ms));
      for (int p = 0; p < 4; ++p)
      {
        acc[p] += ms[p];
      }
    }
    std::printf(
      "%-11s %9.3f %9.3f %9.3f %9.3f %9.3f\n",
      pat.c_str(),
      acc[0] / kIters,
      acc[1] / kIters,
      acc[2] / kIters,
      acc[3] / kIters,
      (acc[0] + acc[1] + acc[2] + acc[3]) / kIters);
    cudaFree(dk);
    cudaFree(dv);
    cudaFree(du);
    cudaFree(da);
    cudaFree(dn);
    cudaFree(dtemp);
  }
  return 0;
}
