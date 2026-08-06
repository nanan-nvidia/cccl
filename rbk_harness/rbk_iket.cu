// IKET driver for the two-pass kernels: one full dispatch at (n, max_seg)
#include "rbk_dispatch.cuh"
#include <cstdio>
#include <vector>
#include <random>
int main(int argc, char** argv)
{
  using K = int; using V = float;
  using config_t   = rbk_impl::winner_config<K, V>;
  const long long n = (argc > 1) ? atoll(argv[1]) : (1ll << 26);
  const int max_seg = (argc > 2) ? atoi(argv[2]) : 1;
  std::vector<K> h((size_t) n); std::mt19937 rng(1);
  std::uniform_int_distribution<int> seg(1, max_seg), kd(0, 1000000);
  long long i = 0; K prev = -1;
  while (i < n) { int r = seg(rng); K v = (K) kd(rng); while (v == prev) v = (K) kd(rng); prev = v;
    for (long long e = std::min(i + r, n); i < e; ++i) h[i] = v; }
  std::vector<V> hv((size_t) n, 1.0f);
  K *dk, *du; V *dv, *da; int* dn; void* tmp = nullptr; size_t tb = 0;
  size_t pad = (size_t)((n + config_t::kTileSize - 1) / config_t::kTileSize) * config_t::kTileSize;
  cudaMalloc(&dk, 4 * pad); cudaMalloc(&dv, 4 * pad); cudaMalloc(&du, 4 * n); cudaMalloc(&da, 4 * n); cudaMalloc(&dn, 4);
  rbk_impl::persistent_rbk_encode<config_t>(nullptr, tb, dk, dv, du, da, dn, (int) n);
  cudaMalloc(&tmp, tb);
  cudaMemcpy(dk, h.data(), 4 * n, cudaMemcpyHostToDevice);
  cudaMemcpy(dv, hv.data(), 4 * n, cudaMemcpyHostToDevice);
  rbk_impl::persistent_rbk_encode<config_t>(tmp, tb, dk, dv, du, da, dn, (int) n);
  cudaDeviceSynchronize();
  int got; cudaMemcpy(&got, dn, 4, cudaMemcpyDeviceToHost);
  printf("n=%lld max_seg=%d runs=%d err=%s\n", n, max_seg, got, cudaGetErrorString(cudaGetLastError()));
  return 0;
}
