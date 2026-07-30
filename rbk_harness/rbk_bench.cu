#include <cub/detail/choose_offset.cuh>

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "rbk_dispatch.cuh"
#include <nvbench/nvbench.cuh>

#ifndef K_IPT
#  define K_IPT 0
#endif
#ifndef K_STAGES
#  define K_STAGES 0
#endif

using KeyT   = int;
using ValueT = float;

namespace
{
// random keys with run lengths uniform in [1, max_seg]; consecutive runs never share a key
// (adjacency-distinctness enforced post-cast so narrow key types can't merge segments)
template <class T>
std::vector<T> gen_keys(long long n, int max_seg, unsigned seed)
{
  std::vector<T> k((size_t) n);
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> seg(1, max_seg), kd(0, 1000000);
  long long i = 0;
  T prev      = T(-1);
  while (i < n)
  {
    int run = seg(rng);
    T v     = T(kd(rng));
    for (int tries = 0; v == prev && tries < 8; ++tries)
    {
      v = T(kd(rng));
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
long long cpu_run_count(const std::vector<T>& h)
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
  using NumRunsT = cub::detail::choose_signed_offset_t<int>;

  KeyT* dk     = nullptr;
  ValueT* dv   = nullptr;
  KeyT* du     = nullptr;
  ValueT* da   = nullptr;
  NumRunsT* dn = nullptr;
  void* dtemp  = nullptr;
  size_t tempb = 0;
  long long n  = 0;
  long long R  = 0;
};

Bufs setup(long long n, int max_seg)
{
  using config_t = rbk_impl::winner_config<KeyT, ValueT, K_IPT, K_STAGES>;
  Bufs b;
  b.n                  = n;
  const size_t pad     = (size_t) ((n + config_t::kTileSize - 1) / config_t::kTileSize) * config_t::kTileSize;
  auto h               = gen_keys<KeyT>(n, max_seg, 1u);
  const long long cpuR = cpu_run_count(h);
  std::vector<ValueT> hv((size_t) n);
  std::mt19937 vrng(2u);
  std::uniform_real_distribution<float> vd(0.0f, 1.0f);
  for (auto& x : hv)
  {
    x = vd(vrng);
  }

  cudaMalloc(&b.dk, sizeof(KeyT) * pad);
  cudaMemset(b.dk, 0, sizeof(KeyT) * pad);
  cudaMalloc(&b.dv, sizeof(ValueT) * pad);
  cudaMemset(b.dv, 0, sizeof(ValueT) * pad);
  cudaMalloc(&b.du, sizeof(KeyT) * (size_t) n);
  cudaMalloc(&b.da, sizeof(ValueT) * (size_t) n);
  cudaMalloc(&b.dn, sizeof(Bufs::NumRunsT));
  rbk_impl::persistent_rbk_encode<config_t>(nullptr, b.tempb, b.dk, b.dv, b.du, b.da, b.dn, (int) n);
  cudaMalloc(&b.dtemp, b.tempb);
  cudaMemset(b.dtemp, 0xAB, b.tempb);
  cudaMemcpy(b.dk, h.data(), sizeof(KeyT) * (size_t) n, cudaMemcpyHostToDevice);
  cudaMemcpy(b.dv, hv.data(), sizeof(ValueT) * (size_t) n, cudaMemcpyHostToDevice);

  Bufs::NumRunsT got = 0;
  cudaMemset(b.dn, 0xEE, sizeof(Bufs::NumRunsT));
  rbk_impl::persistent_rbk_encode<config_t>(b.dtemp, b.tempb, b.dk, b.dv, b.du, b.da, b.dn, (int) n);
  cudaDeviceSynchronize();
  cudaMemcpy(&got, b.dn, sizeof(got), cudaMemcpyDeviceToHost);
  if ((long long) got != cpuR)
  {
    std::printf(
      "*** OURS CORRECTNESS FAIL: n=%lld max_seg=%d got_R=%lld cpu_R=%lld ***\n", n, max_seg, (long long) got, cpuR);
    std::exit(3);
  }
  b.R = cpuR;
  return b;
}

void teardown(Bufs& b)
{
  cudaFree(b.dk);
  cudaFree(b.dv);
  cudaFree(b.du);
  cudaFree(b.da);
  cudaFree(b.dn);
  cudaFree(b.dtemp);
}

void add_counters(nvbench::state& s, const Bufs& b)
{
  s.add_element_count(b.n);
  s.add_global_memory_reads<KeyT>(b.n, "keys");
  s.add_global_memory_reads<ValueT>(b.n, "values");
  s.add_global_memory_writes<KeyT>(b.R, "unique");
  s.add_global_memory_writes<ValueT>(b.R, "aggregates");
}
} // namespace

static void persistent_rbk_bench(nvbench::state& state)
{
  using config_t    = rbk_impl::winner_config<KeyT, ValueT, K_IPT, K_STAGES>;
  const long long n = state.get_int64("Elements{io}");
  const int max_seg = (int) state.get_int64("MaxSegSize");
  auto b            = setup(n, max_seg);
  add_counters(state, b);
  state.exec(nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    rbk_impl::persistent_rbk_encode<config_t>(
      b.dtemp, b.tempb, b.dk, b.dv, b.du, b.da, b.dn, (int) n, launch.get_stream());
  });
  teardown(b);
}

static void cub_rbk_bench(nvbench::state& state)
{
  const long long n = state.get_int64("Elements{io}");
  const int max_seg = (int) state.get_int64("MaxSegSize");
  auto b            = setup(n, max_seg);
  add_counters(state, b);
  void* tmp   = nullptr;
  size_t tbsz = 0;
  cub::DeviceReduce::ReduceByKey(tmp, tbsz, b.dk, b.du, b.dv, b.da, b.dn, cuda::std::plus<>{}, (int) n);
  cudaMalloc(&tmp, tbsz);
  state.exec(nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    cub::DeviceReduce::ReduceByKey(
      tmp, tbsz, b.dk, b.du, b.dv, b.da, b.dn, cuda::std::plus<>{}, (int) n, launch.get_stream());
  });
  cudaFree(tmp);
  teardown(b);
}

NVBENCH_BENCH(persistent_rbk_bench)
  .set_name("persistent_rbk_i32k_f32v")
  .add_int64_power_of_two_axis("Elements{io}", {28})
  .add_int64_power_of_two_axis("MaxSegSize", {0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20});
NVBENCH_BENCH(cub_rbk_bench)
  .set_name("cub_DeviceReduce_ReduceByKey_i32k_f32v")
  .add_int64_power_of_two_axis("Elements{io}", {28})
  .add_int64_power_of_two_axis("MaxSegSize", {0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20});
NVBENCH_MAIN;
