#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "rbk_dispatch.cuh"
int main(int argc, char** argv)
{
  using K        = int;
  using V        = float;
  using config_t = rbk_impl::winner_config<K, V>;
  long long n    = (argc > 1) ? atoll(argv[1]) : 4194304;
  int max_seg    = (argc > 2) ? atoi(argv[2]) : 32;
  int rounds     = (argc > 3) ? atoi(argv[3]) : 40;
  std::vector<K> h((size_t) n);
  std::mt19937 rng(1);
  std::uniform_int_distribution<int> seg(1, max_seg), kd(0, 1000000);
  long long i = 0;
  K prev      = -1;
  while (i < n)
  {
    int r = seg(rng);
    K v   = (K) kd(rng);
    while (v == prev)
    {
      v = (K) kd(rng);
    }
    prev = v;
    for (long long e = std::min(i + r, n); i < e; ++i)
    {
      h[i] = v;
    }
  }
  std::vector<V> hv((size_t) n);
  std::mt19937 vr(2);
  std::uniform_real_distribution<float> vd(0.f, 1.f);
  for (auto& x : hv)
  {
    x = vd(vr);
  }
  std::vector<K> ru;
  std::vector<V> ra; // cpu ref
  for (long long j = 0; j < n; ++j)
  {
    if (j == 0 || h[j] != h[j - 1])
    {
      ru.push_back(h[j]);
      ra.push_back(0.f);
    }
    ra.back() += hv[j];
  }
  K *dk, *du;
  V *dv, *da;
  int* dn;
  void* tmp  = nullptr;
  size_t tb  = 0;
  size_t pad = (size_t) ((n + config_t::kTileSize - 1) / config_t::kTileSize) * config_t::kTileSize;
  cudaMalloc(&dk, 4 * pad);
  cudaMalloc(&dv, 4 * pad);
  cudaMalloc(&du, 4 * n);
  cudaMalloc(&da, 4 * n);
  cudaMalloc(&dn, 4);
  rbk_impl::persistent_rbk_encode<config_t>(nullptr, tb, dk, dv, du, da, dn, (int) n);
  cudaMalloc(&tmp, tb);
  cudaMemset(tmp, 0xAB, tb);
  cudaMemcpy(dk, h.data(), 4 * n, cudaMemcpyHostToDevice);
  cudaMemcpy(dv, hv.data(), 4 * n, cudaMemcpyHostToDevice);
  std::vector<K> gu(ru.size());
  std::vector<V> ga(ru.size());
  int fails = 0;
  for (int r = 0; r < rounds; ++r)
  {
    cudaMemset(du, 0xCC, 4 * n);
    cudaMemset(da, 0xCC, 4 * n);
    rbk_impl::persistent_rbk_encode<config_t>(tmp, tb, dk, dv, du, da, dn, (int) n);
    cudaDeviceSynchronize();
    int got;
    cudaMemcpy(&got, dn, 4, cudaMemcpyDeviceToHost);
    if ((size_t) got != ru.size())
    {
      printf("round %d: R got %d ref %zu\n", r, got, ru.size());
      ++fails;
      continue;
    }
    cudaMemcpy(gu.data(), du, 4 * ru.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(ga.data(), da, 4 * ru.size(), cudaMemcpyDeviceToHost);
    int bad = 0;
    for (size_t j = 0; j < ru.size() && bad < 3; ++j)
    {
      if (gu[j] != ru[j] || std::fabs(ga[j] - ra[j]) > 1e-3f * (1.f + std::fabs(ra[j])))
      {
        printf("round %d run %zu: got(u=%d,a=%g) ref(u=%d,a=%g)\n", r, j, gu[j], (double) ga[j], ru[j], (double) ra[j]);
        ++bad;
      }
    }
    if (bad)
    {
      ++fails;
    }
  }
  printf("%s: %d/%d rounds failed\n", fails ? "FLAKY" : "CLEAN", fails, rounds);
  return 0;
}
