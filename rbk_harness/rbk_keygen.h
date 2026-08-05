#pragma once

#include <algorithm>
#include <random>
#include <vector>

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

// adjacency-distinct key stream shared by the constructed patterns below
template <class T>
struct KeyDrawer
{
  std::mt19937 rng;
  std::uniform_int_distribution<int> kd{0, 1000000};
  T prev{T(-1)};

  explicit KeyDrawer(unsigned seed)
      : rng(seed)
  {}

  T next()
  {
    T v = T(kd(rng));
    for (int tries = 0; v == prev && tries < 8; ++tries)
    {
      v = T(kd(rng));
    }
    prev = v;
    return v;
  }
};

// worst within-warp-tile skew: each 1024-element warp tile is one long run in front plus
// (runs_per_tile - 1) singleton runs packed at the tail -- heads land in the highest lanes
// (max stitch distance, max store scatter, latest lead export). alternate=true ignores
// runs_per_tile and alternates r=255/256 across warp tiles (both mid bands hot per block tile)
template <class T>
std::vector<T> gen_keys_skew(long long n, int runs_per_tile, bool alternate, unsigned seed)
{
  std::vector<T> k((size_t) n);
  KeyDrawer<T> key(seed);
  constexpr long long wt = 1024;
  for (long long base = 0; base < n; base += wt)
  {
    const long long len     = std::min<long long>(wt, n - base);
    const int r             = alternate ? ((((base / wt) & 1) != 0) ? 256 : 255) : runs_per_tile;
    const long long singles = std::min<long long>(r - 1, len - 1);
    const T front           = key.next();
    for (long long i = 0; i < len - singles; ++i)
    {
      k[base + i] = front;
    }
    for (long long s = len - singles; s < len; ++s)
    {
      k[base + s] = key.next();
    }
  }
  return k;
}

// matched-r control for the skew cells: the same runs_per_tile spread evenly through the tile
template <class T>
std::vector<T> gen_keys_even(long long n, int runs_per_tile, unsigned seed)
{
  std::vector<T> k((size_t) n);
  KeyDrawer<T> key(seed);
  constexpr long long wt = 1024;
  for (long long base = 0; base < n; base += wt)
  {
    const long long len = std::min<long long>(wt, n - base);
    const long long r   = std::min<long long>(runs_per_tile, len);
    for (long long j = 0; j < r; ++j)
    {
      const long long lo = base + j * len / r;
      const long long hi = base + (j + 1) * len / r;
      const T v          = key.next();
      for (long long i = lo; i < hi; ++i)
      {
        k[i] = v;
      }
    }
  }
  return k;
}

// heavy-tailed run lengths: P(run >= k) = 1/k (Zipf alpha=1 tail), capped at 2^20 --
// naturally mixes near-dense stretches with giant runs (the across-tile bimodal shape)
template <class T>
std::vector<T> gen_keys_zipf(long long n, unsigned seed)
{
  std::vector<T> k((size_t) n);
  KeyDrawer<T> key(seed);
  std::mt19937 rng(seed ^ 0x9e3779b9u);
  std::uniform_real_distribution<double> ud(1e-9, 1.0);
  long long i = 0;
  while (i < n)
  {
    const long long run = std::min<long long>((long long) (1.0 / ud(rng)), 1 << 20);
    const T v           = key.next();
    const long long e   = std::min<long long>(i + run, n);
    for (; i < e; ++i)
    {
      k[i] = v;
    }
  }
  return k;
}
