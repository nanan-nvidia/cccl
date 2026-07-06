#include <cub/device/device_run_length_encode.cuh>

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "rle_dispatch.cuh"
#include <nvbench/nvbench.cuh>

namespace
{
// random keys with run lengths uniform in [1, max_seg]; consecutive runs never share a key
// (adjacency-distinctness enforced post-cast so narrow key types can't merge segments)
std::vector<KeyT> gen_keys(long long n, int max_seg, unsigned seed)
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

long long cpu_run_count(const std::vector<KeyT>& h)
{
  long long r = 0;
  for (size_t i = 0; i < h.size(); ++i)
  {
    if (i == 0 || !(h[i] == h[i - 1]))
    {
      ++r;
    }
  }
  return r;
}

struct Bufs
{
  KeyT* dk     = nullptr;
  KeyT* du     = nullptr;
  LenT* dc     = nullptr;
  NumRunsT* dn = nullptr;
  void* dtemp  = nullptr; // persistent-RLE temp storage (header + gen-tagged tile states)
  size_t tempb = 0;
  long long n  = 0;
  long long R  = 0; // #runs
};

Bufs setup(long long n, int max_seg)
{
  Bufs b;
  b.n                  = n;
  const size_t pad     = (size_t) ((n + kTileSize - 1) / kTileSize) * kTileSize;
  auto h               = gen_keys(n, max_seg, 1u);
  const long long cpuR = cpu_run_count(h);

  cudaMalloc(&b.dk, sizeof(KeyT) * pad);
  cudaMemset(b.dk, 0, sizeof(KeyT) * pad);
  cudaMalloc(&b.du, sizeof(KeyT) * (size_t) n);
  cudaMalloc(&b.dc, sizeof(LenT) * (size_t) n);
  cudaMalloc(&b.dn, sizeof(NumRunsT));
  persistent_rle_encode(nullptr, b.tempb, b.dk, b.du, b.dc, b.dn, (OffT) n);
  cudaMalloc(&b.dtemp, b.tempb);
  cudaMemset(b.dtemp, 0xAB, b.tempb); // cold start; steady-state calls take the gen-bump path
  cudaMemcpy(b.dk, h.data(), sizeof(KeyT) * (size_t) n, cudaMemcpyHostToDevice);

  // Run CUB once for the authoritative run count + a correctness gate (aborts a fast-but-wrong build).
  void* tmp   = nullptr;
  size_t tbsz = 0;
  cub::DeviceRunLengthEncode::Encode(tmp, tbsz, b.dk, b.du, b.dc, b.dn, n);
  cudaMalloc(&tmp, tbsz);
  cub::DeviceRunLengthEncode::Encode(tmp, tbsz, b.dk, b.du, b.dc, b.dn, n);
  cudaDeviceSynchronize();
  NumRunsT r_w = 0;
  cudaMemcpy(&r_w, b.dn, sizeof(NumRunsT), cudaMemcpyDeviceToHost);
  b.R = (long long) r_w;
  cudaFree(tmp);
  if (b.R != cpuR)
  {
    std::printf("*** CUB CORRECTNESS FAIL: n=%d max_seg=%d cub_R=%d cpu_R=%d ***\n", n, max_seg, b.R, cpuR);
    std::exit(3);
  }
  return b;
}

void teardown(Bufs& b)
{
  cudaFree(b.dk);
  cudaFree(b.du);
  cudaFree(b.dc);
  cudaFree(b.dn);
  cudaFree(b.dtemp);
}

void add_counters(nvbench::state& s, const Bufs& b)
{
  s.add_element_count(b.n);
  s.add_global_memory_reads<KeyT>(b.n, "keys");
  s.add_global_memory_writes<KeyT>(b.R, "unique");
  s.add_global_memory_writes<LenT>(b.R, "counts");
}
} // namespace

static void persistent_rle_bench(nvbench::state& state)
{
  const long long n = state.get_int64("N");
  const int max_seg = (int) state.get_int64("MaxSeg");
  if (n > 0x7fffffffll && sizeof(OffT) < 8)
  {
    state.skip("N needs 64-bit OffT");
    return;
  }
  Bufs b = setup(n, max_seg);
  add_counters(state, b);
  state.exec(nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    persistent_rle_encode(b.dtemp, b.tempb, b.dk, b.du, b.dc, b.dn, (OffT) n, launch.get_stream());
  });
  teardown(b);
}

static void cub_rle_bench(nvbench::state& state)
{
  const long long n = state.get_int64("N");
  const int max_seg = (int) state.get_int64("MaxSeg");
  if (n > 0x7fffffffll && sizeof(OffT) < 8)
  {
    state.skip("N needs 64-bit OffT");
    return;
  }
  Bufs b = setup(n, max_seg);
  add_counters(state, b);
  void* tmp   = nullptr;
  size_t tbsz = 0;
  cub::DeviceRunLengthEncode::Encode(tmp, tbsz, b.dk, b.du, b.dc, b.dn, (OffT) n);
  cudaMalloc(&tmp, tbsz);
  state.exec(nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    cub::DeviceRunLengthEncode::Encode(tmp, tbsz, b.dk, b.du, b.dc, b.dn, (OffT) n, launch.get_stream());
  });
  cudaFree(tmp);
  teardown(b);
}

using axis = std::vector<nvbench::int64_t>;
static axis kSegs{1, 2, 4, 8, 16, 32, 64, 128, 256, 4096, 1048576};
static axis kN{1 << 16, 1 << 20, 1 << 21, 1 << 22, 1 << 23, 1 << 24, 1 << 28};
// huge axis: only meaningful on wide-offset builds (narrow builds skip at runtime)
static axis kNHuge{1ll << 32};
static axis kSegsHuge{2, 16, 256, 1048576};

NVBENCH_BENCH(persistent_rle_bench).set_name("persistent_rle").add_int64_axis("N", kN).add_int64_axis("MaxSeg", kSegs);
NVBENCH_BENCH(persistent_rle_bench)
  .set_name("persistent_rle_huge")
  .add_int64_axis("N", kNHuge)
  .add_int64_axis("MaxSeg", kSegsHuge);
NVBENCH_BENCH(cub_rle_bench)
  .set_name("cub_DeviceRunLengthEncode")
  .add_int64_axis("N", kN)
  .add_int64_axis("MaxSeg", kSegs);
NVBENCH_BENCH(cub_rle_bench).set_name("cub_rle_huge").add_int64_axis("N", kNHuge).add_int64_axis("MaxSeg", kSegsHuge);
NVBENCH_MAIN;
