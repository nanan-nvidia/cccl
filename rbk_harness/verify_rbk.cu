#include <cub/detail/choose_offset.cuh>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "rbk_dispatch.cuh"

#ifndef K_IPT
#  define K_IPT 0
#endif

#ifndef K_STAGES
#  define K_STAGES 0
#endif

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

template <class T>
static std::vector<T> gen_keys(long long n, int max_seg, unsigned seed)
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

template <class ValueT>
static std::vector<ValueT> gen_values(long long n, unsigned seed)
{
  std::vector<ValueT> v((size_t) n);
  std::mt19937 rng(seed ^ 0x9e3779b9u);
  if constexpr (cuda::std::is_integral_v<ValueT>)
  {
    std::uniform_int_distribution<int> vd(0, 3);
    for (auto& x : v)
    {
      x = (ValueT) vd(rng);
    }
  }
  else
  {
    std::uniform_real_distribution<float> vd(0.0f, 1.0f);
    for (auto& x : v)
    {
      x = (ValueT) vd(rng);
    }
  }
  return v;
}

template <class ValueT>
static bool agg_matches(ValueT got, double ref)
{
  if constexpr (cuda::std::is_integral_v<ValueT>)
  {
    return (double) got == ref;
  }
  else
  {
    return std::abs((double) got - ref) <= 1e-4 * std::max(1.0, std::abs(ref));
  }
}

// associative, NON-commutative, exact: composition of affine maps x -> s*x + o over Z/2^16,
// packed (s | o<<16) in a u32. op(a, b) = "a then b": s = sa*sb, o = oa*sb + ob (mod 2^16)
struct AffineComposeOp
{
  __host__ __device__ unsigned operator()(unsigned a, unsigned b) const
  {
    const unsigned sa = a & 0xffffu, oa = a >> 16, sb = b & 0xffffu, ob = b >> 16;
    return ((sa * sb) & 0xffffu) | ((((oa * sb) + ob) & 0xffffu) << 16);
  }
};

struct MinOp
{
  template <class T>
  __host__ __device__ T operator()(T a, T b) const
  {
    return b < a ? b : a;
  }
};

// generic-op verification: exact CPU fold (same op, left-to-right) vs the device line
template <class ValueT, class OpT>
static bool run_op_case(const char* op_name, long long n, int max_seg, unsigned seed, OpT op, int voff = 0)
{
  using KeyT       = int;
  using RbkConfigT = rbk_impl::winner_config<KeyT, ValueT, K_IPT, K_STAGES>;
  auto h           = gen_keys<KeyT>(n, max_seg, seed);
  std::vector<ValueT> hv((size_t) n);
  {
    std::mt19937 rng(seed ^ 0x51ed2701u);
    std::uniform_int_distribution<unsigned> vd(0u, 0xffffffffu);
    for (auto& x : hv)
    {
      x = (ValueT) vd(rng);
    }
  }
  std::vector<KeyT> ref_u;
  std::vector<ValueT> ref_a;
  for (long long i = 0; i < n; ++i)
  {
    if (i == 0 || !(h[i] == h[i - 1]))
    {
      ref_u.push_back(h[i]);
      ref_a.push_back(hv[i]);
    }
    else
    {
      ref_a.back() = op(ref_a.back(), hv[i]);
    }
  }
  const long long refR = (long long) ref_u.size();

  KeyT *dk{}, *du{};
  ValueT *dv{}, *da{};
  int* dn{};
  void* dtemp{};
  CHECK_CUDA(cudaMalloc(&dk, sizeof(KeyT) * (size_t) n));
  CHECK_CUDA(cudaMemcpy(dk, h.data(), sizeof(KeyT) * (size_t) n, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMalloc(&dv, sizeof(ValueT) * ((size_t) n + voff)));
  dv += voff;
  CHECK_CUDA(cudaMemcpy(dv, hv.data(), sizeof(ValueT) * (size_t) n, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMalloc(&du, sizeof(KeyT) * (size_t) n));
  CHECK_CUDA(cudaMalloc(&da, sizeof(ValueT) * (size_t) n));
  CHECK_CUDA(cudaMalloc(&dn, sizeof(int)));
  size_t temp_bytes = 0;
  rbk_impl::persistent_rbk_encode<RbkConfigT>(nullptr, temp_bytes, dk, dv, du, da, dn, (int) n, 0, op);
  CHECK_CUDA(cudaMalloc(&dtemp, temp_bytes));
  CHECK_CUDA(cudaMemset(da, 0xEE, sizeof(ValueT) * (size_t) n));
  CHECK_CUDA(rbk_impl::persistent_rbk_encode<RbkConfigT>(dtemp, temp_bytes, dk, dv, du, da, dn, (int) n, 0, op));
  CHECK_CUDA(cudaDeviceSynchronize());

  int gotR = -1;
  CHECK_CUDA(cudaMemcpy(&gotR, dn, sizeof(int), cudaMemcpyDeviceToHost));
  std::vector<KeyT> gu((size_t) refR);
  std::vector<ValueT> ga((size_t) refR);
  bool ok = ((long long) gotR == refR);
  if (ok)
  {
    CHECK_CUDA(cudaMemcpy(gu.data(), du, sizeof(KeyT) * (size_t) refR, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(ga.data(), da, sizeof(ValueT) * (size_t) refR, cudaMemcpyDeviceToHost));
    for (long long i = 0; i < refR && ok; ++i)
    {
      ok = (gu[(size_t) i] == ref_u[(size_t) i]) && (ga[(size_t) i] == ref_a[(size_t) i]);
      if (!ok)
      {
        std::printf("  op mismatch at run %lld: key %d/%d agg %llx/%llx\n",
                    i,
                    (int) gu[(size_t) i],
                    (int) ref_u[(size_t) i],
                    (unsigned long long) ga[(size_t) i],
                    (unsigned long long) ref_a[(size_t) i]);
      }
    }
  }
  std::printf(
    "%s\t op=%s n=%lld\tmax_seg=%d\truns=%lld\tvoff=%d\n", ok ? "PASS" : "FAIL", op_name, n, max_seg, refR, voff);
  CHECK_CUDA(cudaFree(dk));
  CHECK_CUDA(cudaFree(dv - voff));
  CHECK_CUDA(cudaFree(du));
  CHECK_CUDA(cudaFree(da));
  CHECK_CUDA(cudaFree(dn));
  CHECK_CUDA(cudaFree(dtemp));
  return ok;
}

template <class T, class ValueT, class OffsetT>
static bool run_case(long long n, int max_seg, unsigned seed, bool sampled = false, int koff = 0, int voff = 0)
{
  using RbkConfigT = rbk_impl::winner_config<T, ValueT, K_IPT, K_STAGES>;
  using NumRunsT   = cub::detail::choose_signed_offset_t<OffsetT>;

  const size_t pad = (size_t) n; // EXACT allocation: the bounded-TMA tail must never over-read
  auto h           = gen_keys<T>(n, max_seg, seed);
  auto hv          = gen_values<ValueT>(n, seed);

  // reference: pass 1 counts runs; pass 2 fills the compared window(s), aggregates accumulated in double
  long long refR = 0;
  for (long long i = 0; i < n; ++i)
  {
    refR += (i == 0 || !(h[i] == h[i - 1])) ? 1 : 0;
  }
  // sampled mode (huge dense inputs): compare num_runs exactly + a window at each end -- the tail
  // window exercises run indices past 2^31 without 2x-full-output host mirrors
  const long long win   = sampled ? std::min<long long>(refR, 1 << 20) : refR;
  const long long tail0 = refR - win;
  std::vector<T> ref_u((size_t) win), ref_u2((size_t) (sampled ? win : 0));
  std::vector<double> ref_a((size_t) win), ref_a2((size_t) (sampled ? win : 0));
  {
    long long idx = -1;
    double acc    = 0.0;
    T cur{};
    auto put = [&]() {
      if (idx < 0)
      {
        return;
      }
      if (idx < win)
      {
        ref_u[(size_t) idx] = cur;
        ref_a[(size_t) idx] = acc;
      }
      else if (sampled && idx >= tail0)
      {
        ref_u2[(size_t) (idx - tail0)] = cur;
        ref_a2[(size_t) (idx - tail0)] = acc;
      }
    };
    for (long long i = 0; i < n; ++i)
    {
      if (i == 0 || !(h[i] == h[i - 1]))
      {
        put();
        ++idx;
        cur = h[i];
        acc = 0.0;
      }
      acc += (double) hv[i];
    }
    put();
  }

  T *dk{}, *du{};
  ValueT *dv{}, *da{};
  NumRunsT* dn{};
  void* dtemp{};
  // koff/voff shift the bases off 16B alignment (element units) to exercise the base_skip load path;
  // the shifted base stays exact on the right (allocation = offset + n elements)
  CHECK_CUDA(cudaMalloc(&dk, sizeof(T) * (pad + (size_t) koff)));
  dk += koff;
  CHECK_CUDA(cudaMemcpy(dk, h.data(), sizeof(T) * (size_t) n, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMalloc(&dv, sizeof(ValueT) * (pad + (size_t) voff)));
  dv += voff;
  CHECK_CUDA(cudaMemcpy(dv, hv.data(), sizeof(ValueT) * (size_t) n, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMalloc(&du, sizeof(T) * (size_t) n));
  CHECK_CUDA(cudaMalloc(&da, sizeof(ValueT) * (size_t) n));
  CHECK_CUDA(cudaMalloc(&dn, sizeof(NumRunsT)));
  size_t temp_bytes = 0;
  rbk_impl::persistent_rbk_encode<RbkConfigT>(nullptr, temp_bytes, dk, dv, du, da, dn, (OffsetT) n);
  CHECK_CUDA(cudaMalloc(&dtemp, temp_bytes));
  CHECK_CUDA(cudaMemset(dtemp, 0xAB, temp_bytes)); // garbage: the per-launch clear must handle it

  constexpr int kVerifyRounds = 2;
  bool ok                     = true;
  for (int round = 0; round < kVerifyRounds; ++round)
  {
    CHECK_CUDA(cudaMemset(du, 0xEE, sizeof(T) * (size_t) n));
    CHECK_CUDA(cudaMemset(da, 0xEE, sizeof(ValueT) * (size_t) n));
    CHECK_CUDA(cudaMemset(dn, 0xEE, sizeof(NumRunsT)));
    CHECK_CUDA(rbk_impl::persistent_rbk_encode<RbkConfigT>(dtemp, temp_bytes, dk, dv, du, da, dn, (OffsetT) n, 0));
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
    std::vector<T> got_u((size_t) win), got_u2((size_t) (sampled ? win : 0));
    std::vector<ValueT> got_a((size_t) win), got_a2((size_t) (sampled ? win : 0));
    CHECK_CUDA(cudaMemcpy(got_u.data(), du, sizeof(T) * win, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(got_a.data(), da, sizeof(ValueT) * win, cudaMemcpyDeviceToHost));
    if (sampled)
    {
      CHECK_CUDA(cudaMemcpy(got_u2.data(), du + tail0, sizeof(T) * win, cudaMemcpyDeviceToHost));
      CHECK_CUDA(cudaMemcpy(got_a2.data(), da + tail0, sizeof(ValueT) * win, cudaMemcpyDeviceToHost));
      for (long long i = 0; i < win; ++i)
      {
        if (!(got_u2[i] == ref_u2[i]) || !agg_matches(got_a2[i], ref_a2[i]))
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
      if (!(got_u[i] == ref_u[i]) || !agg_matches(got_a[i], ref_a[i]))
      {
        std::printf(
          "  round %d run %lld: got (u=%g,a=%g) ref (u=%g,a=%g)\n",
          round,
          i,
          (double) got_u[i],
          (double) got_a[i],
          (double) ref_u[i],
          ref_a[i]);
        ok = false;
        ++shown;
      }
    }
  }
  std::printf("%-8s n=%-12lld max_seg=%-8d runs=%-11lld\n", ok ? "PASS" : "FAIL", n, max_seg, refR);

  cudaFree(dk - koff);
  cudaFree(dv - voff);
  cudaFree(du);
  cudaFree(da);
  cudaFree(dn);
  cudaFree(dtemp);
  return ok;
}

template <class T, class ValueT, class OffsetT>
static int run_combo(const char* v_name, const char* off_name, bool huge)
{
  constexpr int kTileSize = rbk_impl::winner_config<T, ValueT, K_IPT, K_STAGES>::kTileSize;
  std::printf("== ValueT=%s OffsetT=%s (tile %d)\n", v_name, off_name, kTileSize);

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
      fails += run_case<T, ValueT, OffsetT>(c.n, c.max_seg, seed) ? 0 : 1;
    }
  }
  // misaligned KEY bases (element offsets off the 16B boundary); misaligned VALUES are
  // unsupported in the prototype (dispatch errors out)
  struct OffCase
  {
    int koff, voff;
  };
  for (const OffCase& o : {OffCase{1, 0}, OffCase{3, 0}})
  {
    fails += run_case<T, ValueT, OffsetT>(3 * kTileSize + 7, 1, 1u, false, o.koff, o.voff) ? 0 : 1;
    fails += run_case<T, ValueT, OffsetT>((1 << 20) + 12345, 2, 42u, false, o.koff, o.voff) ? 0 : 1;
    fails += run_case<T, ValueT, OffsetT>(1 << 22, 4096, 1u, false, o.koff, o.voff) ? 0 : 1;
  }
  if (huge)
  {
    if constexpr (sizeof(OffsetT) < 8)
    {
      std::printf("huge requested but OffsetT is 32-bit -- skipping\n");
    }
    else
    {
      fails += run_case<T, ValueT, OffsetT>((1ll << 32) + 12345, 1000000, 7u) ? 0 : 1; // element offsets past 2^31
      fails += run_case<T, ValueT, OffsetT>((1ll << 32), 1, 7u, /*sampled=*/true) ? 0 : 1; // 4G runs: indices past 2^31
    }
  }
  return fails;
}

int main(int argc, char** argv)
{
  const bool huge = (argc > 1) && argv[1][0] == 'h'; // 2^32-scale cases (wide-offset combos, big GPUs)

  int fails = 0;
  fails += run_combo<int, int, int>("I32", "I32", huge);
  fails += run_combo<int, int, long long>("I32", "I64", huge);
  fails += run_combo<int, float, int>("F32", "I32", huge);
  fails += run_combo<int, float, long long>("F32", "I64", huge);
  // hardening type lines: non-4-byte keys/values exercise every constexpr fallback path
  fails += run_combo<short, int, int>("I16K/I32", "I32", huge);
  fails += run_combo<signed char, int, int>("I8K/I32", "I32", huge);
  // generic associative (non-commutative) ops through the flagged order-preserving path
  std::printf("== generic ops ==\n");
  const long long gn = (1LL << 24) + 12345; // >= 1024 tiles at every tile size in play
  for (int seg : {1, 2, 4, 64, 1024, 65536})
  {
    fails += !run_op_case<unsigned>("affine", gn, seg, 1234u + (unsigned) seg, AffineComposeOp{});
  }
  fails += !run_op_case<int>("min", gn, 64, 4321u, MinOp{});

  std::printf(fails ? "*** %d FAILURES ***\n" : "ALL PASS\n", fails);
  return fails ? 1 : 0;
}
