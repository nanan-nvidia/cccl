#include <cub/device/device_run_length_encode.cuh>

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "persistent_rle.cu"
#include <nvbench/nvbench.cuh>

namespace
{
// random keys with run lengths uniform in [1, max_seg]; consecutive runs never share a key
// (adjacency-distinctness enforced post-cast so narrow key types can't merge segments)
std::vector<KeyT> gen_keys(int n, int max_seg, unsigned seed)
{
  std::vector<KeyT> k(n);
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> seg(1, max_seg), kd(0, 1000000);
  int i     = 0;
  KeyT prev = KeyT(-1);
  while (i < n)
  {
    int run = seg(rng);
    KeyT v  = KeyT(kd(rng));
    for (int tries = 0; v == prev && tries < 8; ++tries)
    {
      v = KeyT(kd(rng));
    }
    prev  = v;
    int e = std::min(i + run, n);
    for (; i < e; ++i)
    {
      k[i] = v;
    }
  }
  return k;
}

int cpu_run_count(const std::vector<KeyT>& h)
{
  int r = 0;
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
  KeyT* dk   = nullptr;
  KeyT* du   = nullptr;
  LenT* dc   = nullptr;
  int* dn    = nullptr;
  u64* dts   = nullptr; // persistent-RLE per-tile state
  int* dctr  = nullptr; // persistent-RLE work-steal counter (unused on CLC path)
  int n      = 0;
  int ntiles = 0;
  int R      = 0; // #runs
};

Bufs setup(int n, int max_seg)
{
  Bufs b;
  b.n              = n;
  b.ntiles         = (n + kTileSize - 1) / kTileSize;
  const size_t pad = (size_t) b.ntiles * kTileSize;
  auto h           = gen_keys(n, max_seg, 1u);
  const int cpuR   = cpu_run_count(h);

  cudaMalloc(&b.dk, sizeof(KeyT) * pad);
  cudaMemset(b.dk, 0, sizeof(KeyT) * pad);
  cudaMalloc(&b.du, sizeof(KeyT) * n);
  cudaMalloc(&b.dc, sizeof(LenT) * n);
  cudaMalloc(&b.dn, sizeof(int));
  cudaMalloc(&b.dts, sizeof(u64) * b.ntiles);
  cudaMemset(b.dts, 0, sizeof(u64) * b.ntiles); // one-time: states are gen-tagged, launches never clear
  cudaMalloc(&b.dctr, sizeof(int));
  cudaMemcpy(b.dk, h.data(), sizeof(KeyT) * n, cudaMemcpyHostToDevice);

  // Run CUB once for the authoritative run count + a correctness gate (aborts a fast-but-wrong build).
  void* tmp   = nullptr;
  size_t tbsz = 0;
  cub::DeviceRunLengthEncode::Encode(tmp, tbsz, b.dk, b.du, b.dc, b.dn, n);
  cudaMalloc(&tmp, tbsz);
  cub::DeviceRunLengthEncode::Encode(tmp, tbsz, b.dk, b.du, b.dc, b.dn, n);
  cudaDeviceSynchronize();
  cudaMemcpy(&b.R, b.dn, sizeof(int), cudaMemcpyDeviceToHost);
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
  cudaFree(b.dts);
  cudaFree(b.dctr);
}

void add_counters(nvbench::state& s, const Bufs& b)
{
  s.add_element_count(b.n);
  s.add_global_memory_reads<KeyT>(b.n, "keys"); // minimal RLE traffic, identical charge for both impls
  s.add_global_memory_writes<KeyT>(b.R, "unique");
  s.add_global_memory_writes<LenT>(b.R, "counts");
}
} // namespace

static void persistent_rle_bench(nvbench::state& state)
{
  const int n = (int) state.get_int64("N"), max_seg = (int) state.get_int64("MaxSeg");
  Bufs b = setup(n, max_seg);
  add_counters(state, b);
  state.exec(nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    persistent_rle_launch(b.dk, b.du, b.dc, b.dn, b.dts, b.dctr, n, b.ntiles, launch.get_stream());
  });
  teardown(b);
}

static void cub_rle_bench(nvbench::state& state)
{
  const int n = (int) state.get_int64("N"), max_seg = (int) state.get_int64("MaxSeg");
  Bufs b = setup(n, max_seg);
  add_counters(state, b);
  void* tmp   = nullptr;
  size_t tbsz = 0;
  cub::DeviceRunLengthEncode::Encode(tmp, tbsz, b.dk, b.du, b.dc, b.dn, n);
  cudaMalloc(&tmp, tbsz);
  state.exec(nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    cub::DeviceRunLengthEncode::Encode(tmp, tbsz, b.dk, b.du, b.dc, b.dn, n, launch.get_stream());
  });
  cudaFree(tmp);
  teardown(b);
}

using axis = std::vector<nvbench::int64_t>;
static axis kSegs{1, 2, 4, 8, 16, 32, 64, 128, 256, 4096, 1048576};
static axis kN{1 << 28};

NVBENCH_BENCH(persistent_rle_bench).set_name("persistent_rle").add_int64_axis("N", kN).add_int64_axis("MaxSeg", kSegs);
NVBENCH_BENCH(cub_rle_bench)
  .set_name("cub_DeviceRunLengthEncode")
  .add_int64_axis("N", kN)
  .add_int64_axis("MaxSeg", kSegs);
NVBENCH_MAIN;
