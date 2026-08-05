#include <cub/detail/choose_offset.cuh>

#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "rbk_dispatch.cuh"
#include "rbk_keygen.h"
#include <nvbench/nvbench.cuh>

#ifndef K_IPT
#  define K_IPT 0
#endif
#ifndef K_STAGES
#  define K_STAGES 0
#endif

#ifndef K_KEY_T
#  define K_KEY_T int
#endif
#ifndef K_VALUE_T
#  define K_VALUE_T float
#endif
using KeyT   = K_KEY_T;
using ValueT = K_VALUE_T;
#ifdef K_OP_MIN
struct BenchOp
{
  template <class T>
  __host__ __device__ T operator()(T a, T b) const
  {
    return b < a ? b : a;
  }
};
#  define K_OP_NAME "_min"
#else
using BenchOp = cuda::std::plus<>;
#  define K_OP_NAME ""
#endif

namespace
{
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

Bufs setup_keys(long long n, const std::vector<KeyT>& h)
{
  using config_t = rbk_impl::winner_config<KeyT, ValueT, K_IPT, K_STAGES>;
  Bufs b;
  b.n                  = n;
  const size_t pad     = (size_t) ((n + config_t::kTileSize - 1) / config_t::kTileSize) * config_t::kTileSize;
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
  rbk_impl::persistent_rbk_encode<config_t>(nullptr, b.tempb, b.dk, b.dv, b.du, b.da, b.dn, (int) n, 0, BenchOp{});
  cudaMalloc(&b.dtemp, b.tempb);
  cudaMemset(b.dtemp, 0xAB, b.tempb);
  cudaMemcpy(b.dk, h.data(), sizeof(KeyT) * (size_t) n, cudaMemcpyHostToDevice);
  cudaMemcpy(b.dv, hv.data(), sizeof(ValueT) * (size_t) n, cudaMemcpyHostToDevice);

  Bufs::NumRunsT got = 0;
  cudaMemset(b.dn, 0xEE, sizeof(Bufs::NumRunsT));
  rbk_impl::persistent_rbk_encode<config_t>(b.dtemp, b.tempb, b.dk, b.dv, b.du, b.da, b.dn, (int) n, 0, BenchOp{});
  cudaDeviceSynchronize();
  cudaMemcpy(&got, b.dn, sizeof(got), cudaMemcpyDeviceToHost);
  if ((long long) got != cpuR)
  {
    std::printf("*** OURS CORRECTNESS FAIL: n=%lld got_R=%lld cpu_R=%lld ***\n", n, (long long) got, cpuR);
    std::exit(3);
  }
  b.R = cpuR;
  return b;
}

Bufs setup(long long n, int max_seg)
{
  return setup_keys(n, gen_keys<KeyT>(n, max_seg, 1u));
}

// "rN" = worst skew at N runs/warp tile, "eN" = matched-r even control, plus alt/zipf shapes;
// the N sweep pins every router edge: 3|4 span, 63|64 staging gate, 255|256 <32>|<4>,
// 511|512 <4>|<2>, 895|896 <2>|<1>, 1023 near-dense
std::vector<KeyT> gen_pattern_keys(long long n, const std::string& pat)
{
  if (pat == "alt255_256")
  {
    return gen_keys_skew<KeyT>(n, 0, true, 1u);
  }
  if (pat == "zipf")
  {
    return gen_keys_zipf<KeyT>(n, 1u);
  }
  const int r = std::atoi(pat.c_str() + 1);
  return (pat[0] == 'e') ? gen_keys_even<KeyT>(n, r, 1u) : gen_keys_skew<KeyT>(n, r, false, 1u);
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
      b.dtemp, b.tempb, b.dk, b.dv, b.du, b.da, b.dn, (int) n, launch.get_stream(), BenchOp{});
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
  cub::DeviceReduce::ReduceByKey(tmp, tbsz, b.dk, b.du, b.dv, b.da, b.dn, BenchOp{}, (int) n);
  cudaMalloc(&tmp, tbsz);
  state.exec(nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    cub::DeviceReduce::ReduceByKey(tmp, tbsz, b.dk, b.du, b.dv, b.da, b.dn, BenchOp{}, (int) n, launch.get_stream());
  });
  cudaFree(tmp);
  teardown(b);
}

static void persistent_rbk_pattern_bench(nvbench::state& state)
{
  using config_t    = rbk_impl::winner_config<KeyT, ValueT, K_IPT, K_STAGES>;
  const long long n = state.get_int64("Elements{io}");
  auto b            = setup_keys(n, gen_pattern_keys(n, state.get_string("Pattern")));
  add_counters(state, b);
  state.exec(nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    rbk_impl::persistent_rbk_encode<config_t>(
      b.dtemp, b.tempb, b.dk, b.dv, b.du, b.da, b.dn, (int) n, launch.get_stream(), BenchOp{});
  });
  teardown(b);
}

static void cub_rbk_pattern_bench(nvbench::state& state)
{
  const long long n = state.get_int64("Elements{io}");
  auto b            = setup_keys(n, gen_pattern_keys(n, state.get_string("Pattern")));
  add_counters(state, b);
  void* tmp   = nullptr;
  size_t tbsz = 0;
  cub::DeviceReduce::ReduceByKey(tmp, tbsz, b.dk, b.du, b.dv, b.da, b.dn, BenchOp{}, (int) n);
  cudaMalloc(&tmp, tbsz);
  state.exec(nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    cub::DeviceReduce::ReduceByKey(tmp, tbsz, b.dk, b.du, b.dv, b.da, b.dn, BenchOp{}, (int) n, launch.get_stream());
  });
  cudaFree(tmp);
  teardown(b);
}

// clang-format off
#define RBK_PATTERNS                                                                  \
  {"r3", "e3", "r4", "e4", "r63", "e63", "r64", "e64", "r255", "e255", "r256",        \
   "e256", "r511", "e511", "r512", "e512", "r895", "e895", "r896", "e896", "r1023",   \
   "alt255_256", "zipf"}
// clang-format on

NVBENCH_BENCH(persistent_rbk_bench)
  .set_name("persistent_rbk_i32k_f32v" K_OP_NAME)
  .add_int64_power_of_two_axis("Elements{io}", {28})
  .add_int64_power_of_two_axis("MaxSegSize", {0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20});
NVBENCH_BENCH(cub_rbk_bench)
  .set_name("cub_DeviceReduce_ReduceByKey_i32k_f32v" K_OP_NAME)
  .add_int64_power_of_two_axis("Elements{io}", {28})
  .add_int64_power_of_two_axis("MaxSegSize", {0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20});
NVBENCH_BENCH(persistent_rbk_pattern_bench)
  .set_name("persistent_rbk_pattern_i32k_f32v" K_OP_NAME)
  .add_int64_power_of_two_axis("Elements{io}", {28})
  .add_string_axis("Pattern", RBK_PATTERNS);
NVBENCH_BENCH(cub_rbk_pattern_bench)
  .set_name("cub_DeviceReduce_ReduceByKey_pattern_i32k_f32v" K_OP_NAME)
  .add_int64_power_of_two_axis("Elements{io}", {28})
  .add_string_axis("Pattern", RBK_PATTERNS);
NVBENCH_MAIN;
