// Self-test / benchmark for the seekdb <-> hipVS CAGRA bridge (no seekdb needed).
//   hipcc -O2 -fopenmp -I. bridge_selftest.cpp ./libseekdb_cuvs_bridge.so \
//         -Wl,-rpath,'$ORIGIN' -o bridge_selftest
//   ./bridge_selftest base.fvecs [n] [dim] [nq] [nq_recall]
//
// Reports: build seconds, VRAM peak during build (rocm-smi sampling), first-query
// latency, batched QPS for top-10 and top-100, recall@10/@100 vs exact CPU kNN,
// and the same numbers again after a serialize -> free -> deserialize round trip.
#include "seekdb_cuvs_bridge.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

// rocm-smi writes its table to stderr, so merge the streams or nothing is parsed.
static long vram_used_mb() {
  FILE* p = popen("rocm-smi --showmeminfo vram --csv 2>&1 | grep -m1 '^card0' | cut -d, -f3", "r");
  if (!p) return -1;
  char buf[64] = {0};
  if (!fgets(buf, sizeof(buf), p)) { pclose(p); return -1; }
  pclose(p);
  return atol(buf);
}

struct VramWatch {
  std::atomic<long> peak{-1};
  std::atomic<bool> stop{false};
  std::thread th;
  void start() {
    th = std::thread([this] {
      while (!stop.load()) {
        long v = vram_used_mb();
        if (v > peak.load()) peak.store(v);
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
      }
    });
  }
  long finish() { stop.store(true); if (th.joinable()) th.join(); return peak.load(); }
};

static bool read_fvecs(const char* path, long want_n, long want_dim, std::vector<float>& out,
                       long& n, long& dim) {
  FILE* f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", path); return false; }
  long cnt = 0;
  int d = 0;
  while (fread(&d, 4, 1, f) == 1) {
    if (d <= 0 || d > 8192) { fprintf(stderr, "bad dim %d at row %ld\n", d, cnt); fclose(f); return false; }
    if (cnt == 0) {
      dim = d;
      out.resize((size_t)(want_n > 0 ? want_n : 1) * dim);
    } else if (d != dim) {
      fprintf(stderr, "ragged fvecs at row %ld: %d vs %ld\n", cnt, d, dim);
      fclose(f); return false;
    }
    if (want_n > 0 && cnt >= want_n) break;
    if ((size_t)(cnt + 1) * dim > out.size()) out.resize((size_t)(cnt + 1) * dim);
    if (fread(&out[(size_t)cnt * dim], 4, dim, f) != (size_t)dim) { fclose(f); return false; }
    cnt++;
  }
  fclose(f);
  if (want_dim > 0 && dim != want_dim) { fprintf(stderr, "file dim %ld != %ld\n", dim, want_dim); return false; }
  n = cnt;
  return n > 0;
}

// exact kNN on the CPU, OpenMP over the base rows; returns recall over `rows` queries
static double recall_cpu(const std::vector<float>& base, long n, long dim,
                         const std::vector<float>& q, long nq, long topk,
                         const std::vector<unsigned>& ids, long* hit_out, long* slots_out) {
  long hit = 0, slots = 0;
  std::vector<std::pair<float, unsigned>> d2((size_t)n);
  std::vector<unsigned> gt((size_t)topk);
  for (long i = 0; i < nq; ++i) {
    const float* qv = &q[(size_t)i * dim];
#pragma omp parallel for schedule(static)
    for (long j = 0; j < n; ++j) {
      const float* bv = &base[(size_t)j * dim];
      float a = 0.f;
      for (long t = 0; t < dim; ++t) { float x = bv[t] - qv[t]; a += x * x; }
      d2[(size_t)j] = {a, (unsigned)j};
    }
    std::partial_sort(d2.begin(), d2.begin() + topk, d2.end());
    for (long t = 0; t < topk; ++t) gt[(size_t)t] = d2[(size_t)t].second;
    for (long a = 0; a < topk; ++a) {
      for (long b = 0; b < topk; ++b) {
        if (ids[(size_t)i * topk + b] == gt[(size_t)a]) { ++hit; break; }
      }
      ++slots;
    }
  }
  if (hit_out) *hit_out = hit;
  if (slots_out) *slots_out = slots;
  return slots ? hit / double(slots) : 0.0;
}

static void report_search(const char* tag, void* h, const std::vector<float>& q, long nq, long dim,
                          long topk, double* qps_out, std::vector<unsigned>* ids_out) {
  std::vector<unsigned> ids((size_t)nq * topk);
  std::vector<float> dist((size_t)nq * topk);
  auto t0 = Clock::now();
  if (seekdb_cuvs_search(h, q.data(), 1, topk, ids.data(), dist.data())) {
    printf("%s SEARCH_FAILED\n", tag);
    return;
  }
  double first = ms_since(t0);
  t0 = Clock::now();
  if (seekdb_cuvs_search(h, q.data(), nq, topk, ids.data(), dist.data())) {
    printf("%s SEARCH_FAILED(batch)\n", tag);
    return;
  }
  double total = ms_since(t0);
  double qps = nq * 1000.0 / total;
  printf("%s topk=%ld first_ms=%.2f batch_ms=%.0f QPS=%.1f\n", tag, topk, first, total, qps);
  if (qps_out) *qps_out = qps;
  if (ids_out) ids_out->swap(ids);
  fflush(stdout);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("usage: %s base.fvecs [n] [dim] [nq] [nq_recall]\n", argv[0]);
    return 2;
  }
  long n = argc > 2 ? atol(argv[2]) : 250000;
  long dim = argc > 3 ? atol(argv[3]) : 128;
  long nq = argc > 4 ? atol(argv[4]) : 2048;
  long nqr = argc > 5 ? atol(argv[5]) : 100;
  const char* qpath = argc > 6 ? argv[6] : NULL;

  std::vector<float> raw;
  long have = 0;
  if (!read_fvecs(argv[1], n + nq, dim, raw, have, dim)) return 3;
  if (have < n) { fprintf(stderr, "file has %ld rows, need %ld\n", have, n); return 3; }
  std::vector<float> base(raw.begin(), raw.begin() + (size_t)n * dim);

  // Queries: an explicit query file wins; otherwise the held-out tail; otherwise
  // perturbed rows.  A query file with a different width is padded with zeros.
  std::vector<float> q((size_t)nq * dim);
  if (qpath) {
    std::vector<float> qs;
    long qn = 0, qd = 0;
    if (!read_fvecs(qpath, nq, 0, qs, qn, qd)) return 3;
    if (qn < nq) { fprintf(stderr, "query file has %ld rows, need %ld\n", qn, nq); return 3; }
    long copy_d = qd < dim ? qd : dim;
    for (long i = 0; i < nq; ++i)
      memcpy(&q[(size_t)i * dim], &qs[(size_t)i * qd], sizeof(float) * copy_d);
  } else if (have - n >= nq) {
    memcpy(q.data(), &raw[(size_t)n * dim], sizeof(float) * (size_t)nq * dim);
  } else {
    for (long i = 0; i < nq; ++i) {
      long src = (i * 7919) % n;
      memcpy(&q[(size_t)i * dim], &base[(size_t)src * dim], sizeof(float) * dim);
      q[(size_t)i * dim] += 1e-3f;
    }
  }

  long base_mb = vram_used_mb();
  printf("CFG n=%ld dim=%ld nq=%ld nq_recall=%ld vram_idle_mb=%ld\n", n, dim, nq, nqr, base_mb);

  VramWatch w;
  w.start();
  auto t0 = Clock::now();
  void* h = seekdb_cuvs_build(base.data(), n, dim);
  double build_ms = ms_since(t0);
  long peak = w.finish();
  if (!h) { printf("BUILD_FAILED\n"); return 4; }
  printf("BUILD_MS=%.0f VRAM_PEAK_MB=%ld VRAM_DELTA_MB=%ld\n", build_ms, peak, peak - base_mb);
  fflush(stdout);

  double qps10 = 0, qps100 = 0;
  std::vector<unsigned> ids10, ids100;
  report_search("SEARCH10", h, q, nq, dim, 10, &qps10, &ids10);
  report_search("SEARCH100", h, q, nq, dim, 100, &qps100, &ids100);

  {
    std::vector<float> qs(q.begin(), q.begin() + (size_t)nqr * dim);
    std::vector<unsigned> sub(ids10.begin(), ids10.begin() + (size_t)nqr * 10);
    long hit = 0, slots = 0;
    double r10 = recall_cpu(base, n, dim, qs, nqr, 10, sub, &hit, &slots);
    printf("RECALL@10=%.4f (%ld/%ld)\n", r10, hit, slots);
    std::vector<unsigned> sub100(ids100.begin(), ids100.begin() + (size_t)nqr * 100);
    double r100 = recall_cpu(base, n, dim, qs, nqr, 100, sub100, &hit, &slots);
    printf("RECALL@100=%.4f (%ld/%ld)\n", r100, hit, slots);
    fflush(stdout);
  }

  // persistence round trip
  const char* snap = "/tmp/bridge_selftest.idx";
  if (seekdb_cuvs_serialize(h, snap) == 0) {
    seekdb_cuvs_free(h);
    h = NULL;
    auto t1 = Clock::now();
    h = seekdb_cuvs_deserialize(snap, n, dim);
    printf("DESERIALIZE_MS=%.0f\n", ms_since(t1));
    if (h) {
      double q2 = 0;
      std::vector<unsigned> ids2;
      report_search("SEARCH10_AFTER_LOAD", h, q, nq, dim, 10, &q2, &ids2);
      long same = 0, tot = 0;
      for (size_t i = 0; i < ids2.size() && i < ids10.size(); ++i) { same += (ids2[i] == ids10[i]); ++tot; }
      printf("IDS_MATCH_AFTER_LOAD=%.4f (%ld/%ld)\n", tot ? same / double(tot) : 0.0, same, tot);
    } else {
      printf("DESERIALIZE_FAILED\n");
    }
  } else {
    printf("SERIALIZE_FAILED\n");
  }
  if (h) seekdb_cuvs_free(h);
  printf("DONE\n");
  return 0;
}
