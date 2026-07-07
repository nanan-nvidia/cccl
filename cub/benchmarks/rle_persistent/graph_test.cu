#include <cstdio>
#include <random>
#include <vector>

#include "rle_dispatch.cuh"

int main()
{
  const long long n = (1 << 22) + 12345;
  std::vector<KeyT> h((size_t) n);
  std::mt19937 rng(9);
  std::uniform_int_distribution<int> seg(1, 8), kd(0, 1000000);
  long long i           = 0;
  int prev              = -1;
  long long expect_runs = 0;
  while (i < n)
  {
    int v = kd(rng);
    if (v == prev)
    {
      v = (v + 1) % 1000001;
    }
    prev = v;
    ++expect_runs;
    long long e = std::min<long long>(i + seg(rng), n);
    for (; i < e; ++i)
    {
      h[(size_t) i] = KeyT(v);
    }
  }
  KeyT *dk, *du;
  LenT* dc;
  NumRunsT* dn;
  void* dtemp;
  size_t tempb = 0;
  cudaMalloc(&dk, sizeof(KeyT) * n);
  cudaMemcpy(dk, h.data(), sizeof(KeyT) * n, cudaMemcpyHostToDevice);
  cudaMalloc(&du, sizeof(KeyT) * n);
  cudaMalloc(&dc, sizeof(LenT) * n);
  cudaMalloc(&dn, sizeof(NumRunsT));
  persistent_rle_encode(nullptr, tempb, dk, du, dc, dn, (OffT) n);
  cudaMalloc(&dtemp, tempb);
  cudaMemset(dtemp, 0xAB, tempb);

  cudaStream_t s;
  cudaStreamCreate(&s);
  cudaGraph_t graph;
  cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
  persistent_rle_encode(dtemp, tempb, dk, du, dc, dn, (OffT) n, s);
  cudaStreamEndCapture(s, &graph);
  cudaGraphExec_t exec;
  cudaError_t ge = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
  if (ge != cudaSuccess)
  {
    std::printf("GRAPH INSTANTIATE FAILED: %s\n", cudaGetErrorString(ge));
    return 1;
  }
  int fails = 0;
  for (int rep = 0; rep < 4; ++rep)
  {
    cudaMemset(dn, 0xEE, sizeof(NumRunsT));
    cudaGraphLaunch(exec, s);
    cudaStreamSynchronize(s);
    NumRunsT r = -1;
    cudaMemcpy(&r, dn, sizeof(NumRunsT), cudaMemcpyDeviceToHost);
    const bool ok = ((long long) r == expect_runs);
    std::printf("replay %d: runs=%lld expect=%lld %s\n", rep, (long long) r, expect_runs, ok ? "OK" : "FAIL");
    fails += !ok;
  }
  std::printf(fails ? "GRAPH TEST FAILED\n" : "GRAPH TEST PASS\n");
  return fails ? 1 : 0;
}
