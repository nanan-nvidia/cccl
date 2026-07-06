// watchdog runner for hang debugging: ./wd_runner <max_seg> [launches=200]
// Loops full-size launches (N=2^28); on a >10s stall OR a launch error, dumps the per-warp
// heartbeat buffer and barrier arrive-counters (RLE_HEARTBEAT=1 builds).
// Site codes: 1=load-empty 2=comp-full 3=comp-posfree 4=poll-seq 5=store-computed 6=store-staged
// 7=store-flushwait 8=store-prefixed 9=bk-computed 10=bk-prefixed 12=load-clc 13=comp-pubfanin
// NOTE: the RLE_HEARTBEAT instrumentation this runner dumps lives in the ARCHIVED debug kernel
// (~/rle_wip/rbpipe_wip/persistent_rle_rbpipe_hb.cu); against the production kernel this runner
// still works as a plain watchdog (hang detector), it just prints an empty heartbeat table.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#include "rle_dispatch.cuh"

static std::vector<KeyT> gen_keys(long long n, int max_seg, unsigned seed)
{
  std::vector<KeyT> k((size_t) n);
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> seg(1, max_seg), kd(0, 1000000);
  long long i = 0;
  int prev    = -1;
  while (i < n)
  {
    int run = seg(rng), v = kd(rng);
    if (v == prev)
    {
      v = (v + 1) % 1000001;
    }
    prev        = v;
    long long e = std::min<long long>(i + run, n);
    for (; i < e; ++i)
    {
      k[i] = KeyT(v);
    }
  }
  return k;
}

static const char* site_name(unsigned s)
{
  switch (s)
  {
    case 1:
      return "load-empty";
    case 2:
      return "comp-full";
    case 3:
      return "comp-posfree";
    case 4:
      return "poll-seq";
    case 5:
      return "store-computed";
    case 6:
      return "store-staged";
    case 7:
      return "store-FLUSHWAIT";
    case 8:
      return "store-prefixed";
    case 9:
      return "bk-computed";
    case 10:
      return "bk-prefixed";
    case 12:
      return "load-clc";
    case 13:
      return "comp-pubfanin";
    default:
      return "?";
  }
}

static void dump_state(volatile unsigned* hb_h, volatile unsigned* hbc_h)
{
  unsigned min_gen[16], max_gen[16], cnt[16];
  for (int s = 0; s < 16; ++s)
  {
    min_gen[s] = ~0u;
    max_gen[s] = 0;
    cnt[s]     = 0;
  }
  unsigned global_min_gen = ~0u;
  int min_blk             = -1;
  for (int b = 0; b < 2048; ++b)
  {
    for (int w = 0; w < 32; ++w)
    {
      const unsigned v = hb_h[b * 32 + w];
      if (!v)
      {
        continue;
      }
      const unsigned s = v & 15u, g = v >> 4;
      cnt[s]++;
      if (g < min_gen[s])
      {
        min_gen[s] = g;
      }
      if (g > max_gen[s])
      {
        max_gen[s] = g;
      }
      if (g < global_min_gen)
      {
        global_min_gen = g;
        min_blk        = b;
      }
    }
  }
  for (int s = 0; s < 16; ++s)
  {
    if (cnt[s])
    {
      std::printf("  site %-16s: %6u warps, gen [%u..%u]\n", site_name(s), cnt[s], min_gen[s], max_gen[s]);
    }
  }
  if (min_blk < 0)
  {
    return;
  }
  std::printf("  most-stuck block %d (lowest gen %u), all its warps:\n", min_blk, global_min_gen);
  for (int w = 0; w < 32; ++w)
  {
    const unsigned v = hb_h[min_blk * 32 + w];
    if (v)
    {
      std::printf("    warp %2d: gen %6u site %s\n", w, v >> 4, site_name(v & 15u));
    }
  }
  static const char* kind_name[5] = {"computed", "prefixed", "empty", "posfree", "stagedwt"};
  std::printf("  arrive counters for block %d (kind[slot]=count):\n", min_blk);
  for (int k = 0; k < 5; ++k)
  {
    std::printf("    %-9s:", kind_name[k]);
    for (int s = 0; s < 5; ++s)
    {
      std::printf(" [%d]=%u", s, hbc_h[min_blk * 64 + k * 5 + s]);
    }
    std::printf("\n");
  }
  static const char* ev_name[8] = {"-", "bufearly", "tail", "sentinel", "noearly", "bklate", "bkearly", "bksent"};
  std::printf("  per-warp [computed-pass gen | last empty-arrive gen@site]:\n");
  for (int w = 10; w < 19; ++w)
  {
    const int cp     = (int) hbc_h[min_blk * 64 + 40 + (w - 10) * 2] - 1;
    const unsigned e = hbc_h[min_blk * 64 + 41 + (w - 10) * 2];
    std::printf("    w%-2d: computed-pass %3d | arrive %3d @%s (launch tag %u)\n",
                w,
                cp,
                (int) ((e >> 4) & 0x3ffffff) - 1,
                ev_name[e & 7u],
                e >> 30);
  }
  std::printf("  store-sentinel: tile_id=%d at gen %d | load-sentinel: tile_id=%d at gen %d\n",
              (int) hbc_h[min_blk * 64 + 60],
              (int) hbc_h[min_blk * 64 + 61] - 1,
              (int) hbc_h[min_blk * 64 + 62],
              (int) hbc_h[min_blk * 64 + 63] - 1);
  std::fflush(stdout);
}

int main(int argc, char** argv)
{
  const int max_seg  = (argc > 1) ? atoi(argv[1]) : 64;
  const int launches = (argc > 2) ? atoi(argv[2]) : 200;
  const long long n  = 1 << 28;
  auto h             = gen_keys(n, max_seg, 1u);
  KeyT *dk, *du;
  LenT* dc;
  NumRunsT* dn;
  void* dtemp;
  size_t tempb = 0;
  cudaMalloc(&dk, sizeof(KeyT) * (size_t) n);
  cudaMemcpy(dk, h.data(), sizeof(KeyT) * (size_t) n, cudaMemcpyHostToDevice);
  cudaMalloc(&du, sizeof(KeyT) * (size_t) n);
  cudaMalloc(&dc, sizeof(LenT) * (size_t) n);
  cudaMalloc(&dn, sizeof(NumRunsT));
  persistent_rle_encode(nullptr, tempb, dk, du, dc, dn, (OffT) n);
  cudaMalloc(&dtemp, tempb);
  cudaMemset(dtemp, 0xAB, tempb);

  volatile unsigned *hb_h, *hbc_h;
  unsigned *hb_d, *hbc_d;
  cudaHostAlloc((void**) &hb_h, 2048 * 32 * sizeof(unsigned), cudaHostAllocMapped);
  cudaHostGetDevicePointer((void**) &hb_d, (void*) hb_h, 0);
  cudaHostAlloc((void**) &hbc_h, 2048 * 64 * sizeof(unsigned), cudaHostAllocMapped);
  cudaHostGetDevicePointer((void**) &hbc_d, (void*) hbc_h, 0);
#if RLE_HEARTBEAT
  cudaMemcpyToSymbol(g_hb, &hb_d, sizeof(hb_d));
  cudaMemcpyToSymbol(g_hbc, &hbc_d, sizeof(hbc_d));
#endif

  cudaEvent_t done;
  cudaEventCreateWithFlags(&done, cudaEventDisableTiming);
  int R = -1;
  for (int i = 0; i < launches; ++i)
  {
    memset((void*) hb_h, 0, 2048 * 32 * sizeof(unsigned));
    memset((void*) hbc_h, 0, 2048 * 64 * sizeof(unsigned));
    persistent_rle_encode(dtemp, tempb, dk, du, dc, dn, (OffT) n, 0);
    cudaEventRecord(done, 0);
    auto t0 = std::chrono::steady_clock::now();
    while (cudaEventQuery(done) == cudaErrorNotReady)
    {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > 10.0)
      {
        std::printf("*** HANG at launch %d (seg=%d) ***\n", i, max_seg);
        dump_state(hb_h, hbc_h);
        std::_Exit(3);
      }
    }
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
    {
      std::printf("*** ERROR at launch %d: %s ***\n", i, cudaGetErrorString(e));
      dump_state(hb_h, hbc_h);
      std::_Exit(4);
    }
    if (i % 25 == 0)
    {
      cudaMemcpy(&R, dn, sizeof(NumRunsT), cudaMemcpyDeviceToHost);
      std::printf("launch %d ok (R=%d)\n", i, R);
      std::fflush(stdout);
    }
  }
  std::printf("ALL %d LAUNCHES OK seg=%d\n", launches, max_seg);
  return 0;
}
