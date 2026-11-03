// siftbench.cpp -- SIFT (fvecs) CAGRA benchmark: build time / query latency / exact recall.
// Build:  g++ -O2 -fopenmp -I . -o siftbench siftbench.cpp ./libseekdb_cuvs_bridge.so -Wl,-rpath,$PWD
// Run:    ./siftbench sift_base.fvecs sift_query.fvecs <nbase> <nquery> <topk>
// Env:    SEEKDB_CUVS_{TRACE,NITER,GRAPH_DEGREE,INTERMEDIATE_DEGREE,ITOPK,WIDTH}
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <chrono>
#include <utility>
extern "C" {
#include "seekdb_cuvs_bridge.h"
}

static std::vector<float> load_fvecs(const char* p, long n, long dim) {
  FILE* f = fopen(p, "rb");
  if (!f) { perror(p); exit(1); }
  std::vector<float> out((size_t)n * dim);
  int hdr = 0;
  for (long i = 0; i < n; i++) {
    if (fread(&hdr, 4, 1, f) != 1) { fprintf(stderr, "short read %s at %ld\n", p, i); exit(1); }
    if (hdr != (int)dim) { fprintf(stderr, "dim mismatch %d\n", hdr); exit(1); }
    if (fread(out.data() + (size_t)i * dim, 4, dim, f) != (size_t)dim) { fprintf(stderr, "short vec\n"); exit(1); }
  }
  fclose(f);
  return out;
}

static double now_ms() {
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
  if (argc < 6) { fprintf(stderr, "usage: %s base.fvecs query.fvecs n nq topk\n", argv[0]); return 2; }
  const char* bp = argv[1];
  const char* qp = argv[2];
  long n = atol(argv[3]), nq = atol(argv[4]), k = atol(argv[5]);
  long dim = 128;

  std::vector<float> base = load_fvecs(bp, n, dim);
  std::vector<float> qs = load_fvecs(qp, nq, dim);

  double t0 = now_ms();
  void* h = seekdb_cuvs_build(base.data(), n, dim);
  double build = now_ms() - t0;
  if (!h) { fprintf(stderr, "BUILD_FAILED\n"); return 3; }

  std::vector<unsigned> ids((size_t)nq * k);
  std::vector<float> dst((size_t)nq * k);
  int rc = seekdb_cuvs_search(h, qs.data(), 1, k, ids.data(), dst.data());   // warm-up
  if (rc) { fprintf(stderr, "first search rc=%d\n", rc); return 4; }

  std::vector<double> lat(nq);
  for (long i = 0; i < nq; i++) {
    double a = now_ms();
    rc = seekdb_cuvs_search(h, qs.data() + (size_t)i * dim, 1, k,
                            ids.data() + (size_t)i * k, dst.data() + (size_t)i * k);
    if (rc) { fprintf(stderr, "search rc=%d at %ld\n", rc, i); return 4; }
    lat[i] = now_ms() - a;
  }
  double tot = 0;
  for (double v : lat) tot += v;
  std::vector<double> s = lat;
  std::sort(s.begin(), s.end());

  // Exact ground truth by brute force (squared L2), so no ground-truth file is needed.
  long hit = 0;
#pragma omp parallel for schedule(dynamic) reduction(+ : hit)
  for (long qi = 0; qi < nq; qi++) {
    const float* q = qs.data() + (size_t)qi * dim;
    std::vector<std::pair<float, unsigned>> best(n);
    for (long j = 0; j < n; j++) {
      const float* v = base.data() + (size_t)j * dim;
      float acc = 0;
      for (long t = 0; t < dim; t++) { float df = q[t] - v[t]; acc += df * df; }
      best[j] = std::make_pair(acc, (unsigned)j);
    }
    std::partial_sort(best.begin(), best.begin() + k, best.end());
    for (long r = 0; r < k; r++) {
      for (long c = 0; c < k; c++) {
        if (ids[(size_t)qi * k + c] == best[r].second) { hit++; break; }
      }
    }
  }

  printf("CFG n=%ld dim=%ld nq=%ld topk=%ld\n", n, dim, nq, k);
  printf("BUILD_MS=%.1f\n", build);
  printf("QPS=%.1f TOTAL_MS=%.1f P50_MS=%.4f P95_MS=%.4f\n",
         nq * 1000.0 / tot, tot, s[nq / 2], s[(size_t)(nq * 0.95)]);
  printf("RECALL_AT_%ld=%.4f nq=%ld\n", k, (double)hit / (double)(nq * k), nq);
  seekdb_cuvs_free(h);
  return 0;
}
