// Bench driver for the seekdb <-> hipVS CAGRA bridge (standalone, no seekdb needed).
// Times index build + nq single-row searches and reports Recall@k vs CPU brute force.
//   hipcc -O2 -fopenmp -I. bench.cpp ./libseekdb_cuvs_bridge.so -Wl,-rpath,'$ORIGIN' -o bench
//   ./bench n dim nq topk recall_n
#include "seekdb_cuvs_bridge.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point t) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

int main(int argc, char** argv) {
  const long n = argc > 1 ? atol(argv[1]) : 250000;
  const long dim = argc > 2 ? atol(argv[2]) : 128;
  const long nq = argc > 3 ? atol(argv[3]) : 4096;
  const long k = argc > 4 ? atol(argv[4]) : 10;
  const long rn = argc > 5 ? atol(argv[5]) : 256;
  const long threads =
#ifdef _OPENMP
      omp_get_max_threads();
#else
      1;
#endif

  std::mt19937 rng(42);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> base((size_t)n * dim), q((size_t)nq * dim);
  for (auto& v : base) v = nd(rng);
  for (auto& v : q) v = nd(rng);

  auto t0 = Clock::now();
  void* h = seekdb_cuvs_build(base.data(), n, dim);
  double build_ms = ms(t0);
  if (!h) { printf("BUILD_FAILED\n"); return 1; }

  std::vector<unsigned> ids((size_t)nq * k);
  std::vector<float> d(k);
  t0 = Clock::now();
  if (seekdb_cuvs_search(h, q.data(), 1, k, ids.data(), d.data())) { printf("SEARCH_FAILED\n"); return 2; }
  double first_ms = ms(t0);

  printf("CFG n=%ld dim=%ld nq=%ld topk=%ld threads=%ld\n", n, dim, nq, k, threads);
  printf("BUILD_MS=%.1f FIRST_QUERY_MS=%.2f\n", build_ms, first_ms);

  std::vector<double> lat((size_t)nq);
  t0 = Clock::now();
  for (long i = 0; i < nq; ++i) {
    auto s = Clock::now();
    if (seekdb_cuvs_search(h, q.data() + i * dim, 1, k, ids.data() + i * k, d.data())) {
      printf("SEARCH_FAILED i=%ld\n", i);
      return 3;
    }
    lat[i] = ms(s);
  }
  double total = ms(t0);
  std::sort(lat.begin(), lat.end());
  printf("QPS=%.1f TOTAL_MS=%.0f P50_MS=%.3f P95_MS=%.3f\n", nq * 1000.0 / total, total,
         lat[nq / 2], lat[(size_t)(nq * 0.95)]);
  fflush(stdout);

  long hit = 0;
  long checked = rn < nq ? rn : nq;
  std::vector<std::pair<float, unsigned>> d2((size_t)n);
  for (long i = 0; i < checked; ++i) {
    const float* qv = q.data() + i * dim;
    long j;
#pragma omp parallel for schedule(static)
    for (j = 0; j < n; ++j) {
      const float* bv = base.data() + j * dim;
      float a = 0.f;
      for (long t = 0; t < dim; ++t) { float x = bv[t] - qv[t]; a += x * x; }
      d2[(size_t)j] = {a, (unsigned)j};
    }
    std::partial_sort(d2.begin(), d2.begin() + k, d2.end());
    for (long a = 0; a < k; ++a) {
      for (long b = 0; b < k; ++b) {
        if (ids[(size_t)i * k + b] == d2[(size_t)a].second) { ++hit; break; }
      }
    }
  }
  printf("RECALL@%ld=%.4f (nq=%ld)\n", k, hit / double(checked * k), checked);
  seekdb_cuvs_free(h);
  return 0;
}
