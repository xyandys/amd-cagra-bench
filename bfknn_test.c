/* bfknn_test.c — handoff §6 step 2
 *
 * GPU brute-force exact kNN via cuvsBruteForceSearch (k+1 neighbors, self
 * dropped on host) -> cuvsCagraIndexFromArgs -> official cuvsCagraSearch.
 * Measures: kNN wall time, recall@10 / recall@100, search QPS at nq=4096.
 *
 * Usage: ./bfknn_test [n] [dim] [degree] [nq] [topk] [itopk]
 * Env:   BF_SELFTEST=1  verify bf neighbors against CPU exact (rows sampled)
 *        BF_QPS=1       timed search loops, nq=4096, topk 10 and 100
 */
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/cagra.h>
#include <cuvs/neighbors/brute_force.h>
#include <dlpack/dlpack.h>
#ifdef __HIP_PLATFORM_AMD__
#include <cuvs/cuda_runtime.h>
#else
#include <cuda_runtime.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAXK 128

static double now_sec(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static unsigned rng_state = 20261110u;
static float frand(void) {
  rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17; rng_state ^= rng_state << 5;
  return (float)(rng_state % 1000000u) / 1000000.0f;
}

static void topk_push(float* bd, long* bi, long k, float d2, long id) {
  if (d2 < bd[k - 1]) {
    long p = k - 1;
    while (p > 0 && bd[p - 1] > d2) { bd[p] = bd[p - 1]; bi[p] = bi[p - 1]; p--; }
    bd[p] = d2; bi[p] = id;
  }
}

/* exact ground truth rows (squared L2) for query block [q0, q0+nq) taken from qs */
static void exact_gt_rows(const float* base, long n, long dim, const float* qs, long q0,
                          long nq, long k, long* gt) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 4)
#endif
  for (long qi = 0; qi < nq; qi++) {
    float bd[MAXK]; long bi[MAXK];
    const float* q = qs + (size_t)qi * dim;
    for (long t = 0; t < k; t++) { bd[t] = 3.4e38f; bi[t] = -1; }
    for (long j = 0; j < n; j++) {
      const float* v = base + (size_t)j * dim;
      float acc = 0.0f;
      for (long t = 0; t < dim; t++) { float df = q[t] - v[t]; acc += df * df; }
      topk_push(bd, bi, k, acc, j);
    }
    for (long t = 0; t < k; t++) gt[(size_t)qi * k + t] = bi[t];
    (void)q0;
  }
}

static void fill_tensor(DLManagedTensor* t, void* data, int device, long rows, long cols,
                        int uint32_flag) {
  memset(t, 0, sizeof(*t));
  t->dl_tensor.data = data;
  t->dl_tensor.device.device_type = device ? kDLCUDA : kDLCPU;
  t->dl_tensor.device.device_id = 0;
  t->dl_tensor.ndim = 2;
  t->dl_tensor.dtype.code = uint32_flag ? kDLUInt : kDLFloat;
  t->dl_tensor.dtype.bits = 32;
  t->dl_tensor.dtype.lanes = 1;
  t->dl_tensor.shape = (int64_t*)malloc(sizeof(int64_t) * 2);
  t->dl_tensor.shape[0] = rows; t->dl_tensor.shape[1] = cols;
  t->dl_tensor.strides = 0;
  t->dl_tensor.byte_offset = 0;
}

/* cuvsBruteForceSearch demands int64 neighbors despite what brute_force.h
 * docs claim (uint32); hipVS build checks dtype.code == kDLInt, bits == 64. */
static void fill_tensor_i64(DLManagedTensor* t, void* data, long rows, long cols) {
  memset(t, 0, sizeof(*t));
  t->dl_tensor.data = data;
  t->dl_tensor.device.device_type = kDLCUDA;
  t->dl_tensor.device.device_id = 0;
  t->dl_tensor.ndim = 2;
  t->dl_tensor.dtype.code = kDLInt;
  t->dl_tensor.dtype.bits = 64;
  t->dl_tensor.dtype.lanes = 1;
  t->dl_tensor.shape = (int64_t*)malloc(sizeof(int64_t) * 2);
  t->dl_tensor.shape[0] = rows; t->dl_tensor.shape[1] = cols;
  t->dl_tensor.strides = 0;
  t->dl_tensor.byte_offset = 0;
}

/* one CAGRA search batch; mirrors the bridge's search path */
static int cagra_search_batch(cuvsResources_t res, cuvsCagraIndex_t idx, long dim,
                              const float* qs, long nq, long topk, unsigned* out_ids,
                              long itopk) {
  float* qd = 0; unsigned* nd = 0; float* dd = 0; cuvsCagraSearchParams_t sp = 0;
  int rc = 0;
  if (cuvsRMMAlloc(res, (void**)&qd, sizeof(float) * (size_t)nq * dim) != CUVS_SUCCESS) return 10;
  if (cuvsRMMAlloc(res, (void**)&nd, sizeof(unsigned) * (size_t)nq * topk) != CUVS_SUCCESS) { rc = 11; goto done; }
  if (cuvsRMMAlloc(res, (void**)&dd, sizeof(float) * (size_t)nq * topk) != CUVS_SUCCESS) { rc = 12; goto done; }
  cudaMemcpy(qd, qs, sizeof(float) * (size_t)nq * dim, cudaMemcpyDefault);
  {
    DLManagedTensor qt, nt, dt;
    fill_tensor(&qt, qd, 1, nq, dim, 0);
    fill_tensor(&nt, nd, 1, nq, topk, 1);
    fill_tensor(&dt, dd, 1, nq, topk, 0);
    if (cuvsCagraSearchParamsCreate(&sp) != CUVS_SUCCESS) { rc = 13; goto done; }
    size_t need = (size_t)topk * 4;
    if (need < 512) need = 512;
    if ((long)need < itopk) need = (size_t)itopk;
    sp->itopk_size = need;
    sp->search_width = 4;
    cuvsFilter f; f.type = NO_FILTER; f.addr = (uintptr_t)0;
    if (cuvsCagraSearch(res, sp, idx, &qt, &nt, &dt, f) != CUVS_SUCCESS) { rc = 14; goto done; }
    if (cudaMemcpy(out_ids, nd, sizeof(unsigned) * (size_t)nq * topk, cudaMemcpyDefault) != cudaSuccess) rc = 15;
  }
done:
  if (sp) cuvsCagraSearchParamsDestroy(sp);
  if (dd) cuvsRMMFree(res, dd, sizeof(float) * (size_t)nq * topk);
  if (nd) cuvsRMMFree(res, nd, sizeof(unsigned) * (size_t)nq * topk);
  if (qd) cuvsRMMFree(res, qd, sizeof(float) * (size_t)nq * dim);
  return rc;
}

static double recall_of(const unsigned* got, const long* gt, long nq, long k) {
  long hits = 0;
  for (long qi = 0; qi < nq; qi++)
    for (long r = 0; r < k; r++) {
      long gr = gt[(size_t)qi * k + r];
      if (gr < 0) { hits++; continue; }
      for (long c = 0; c < k; c++)
        if (got[(size_t)qi * k + c] == (unsigned)gr) { hits++; break; }
    }
  return (double)hits / (double)(nq * k);
}

int main(int argc, char** argv) {
  long n      = argc > 1 ? atol(argv[1]) : 20000;
  long dim    = argc > 2 ? atol(argv[2]) : 32;
  long degree = argc > 3 ? atol(argv[3]) : 32;
  long nq     = argc > 4 ? atol(argv[4]) : 300;
  long topk   = argc > 5 ? atol(argv[5]) : 10;
  long itopk  = argc > 6 ? atol(argv[6]) : 512;
  if (degree > MAXK - 1) degree = MAXK - 1;
  int selftest = getenv("BF_SELFTEST") && atoi(getenv("BF_SELFTEST")) == 1;
  int do_qps   = getenv("BF_QPS") && atoi(getenv("BF_QPS")) == 1;
  long kp1 = degree + 1;

  float* base = (float*)malloc(sizeof(float) * (size_t)n * dim);
  float* qs   = (float*)malloc(sizeof(float) * (size_t)nq * dim);
  long long* nb = (long long*)malloc(sizeof(long long) * (size_t)n * kp1);
  unsigned* graph = (unsigned*)malloc(sizeof(unsigned) * (size_t)n * degree);
  long* gt10 = (long*)malloc(sizeof(long) * (size_t)nq * 10);
  long* gt100 = (long*)malloc(sizeof(long) * (size_t)nq * 100);
  unsigned* got = (unsigned*)malloc(sizeof(unsigned) * (size_t)nq * 100);
  if (!base || !qs || !nb || !graph || !gt10 || !gt100 || !got) { fprintf(stderr, "OOM(host)\n"); return 2; }

  for (long i = 0; i < n * dim; i++) base[i] = frand();
  for (long i = 0; i < nq * dim; i++) qs[i] = frand();

  printf("[bfknn] n=%ld dim=%ld degree=%ld nq=%ld itopk=%ld\n", n, dim, degree, nq, itopk);

  cuvsResources_t res = 0;
  if (cuvsResourcesCreate(&res) != CUVS_SUCCESS) { fprintf(stderr, "res: %s\n", cuvsGetLastErrorText()); return 3; }

  /* ---- upload dataset ---- */
  float* ds_dev = 0; unsigned* nb_dev = 0; float* dist_dev = 0; unsigned* graph_dev = 0;
  double t0 = now_sec();
  if (cuvsRMMAlloc(res, (void**)&ds_dev, sizeof(float) * (size_t)n * dim) != CUVS_SUCCESS) { fprintf(stderr, "alloc ds\n"); return 4; }
  cudaMemcpy(ds_dev, base, sizeof(float) * (size_t)n * dim, cudaMemcpyDefault);
  double t1 = now_sec();

  DLManagedTensor ds_t;
  fill_tensor(&ds_t, ds_dev, 1, n, dim, 0);

  /* ---- brute-force build (trivial) + full self-search: THE graph builder ---- */
  cuvsBruteForceIndex_t bf = 0;
  if (cuvsBruteForceIndexCreate(&bf) != CUVS_SUCCESS) { fprintf(stderr, "bf create: %s\n", cuvsGetLastErrorText()); return 5; }
  double t2 = now_sec();
  if (cuvsBruteForceBuild(res, &ds_t, (cuvsDistanceType)0, 0.f, bf) != CUVS_SUCCESS) {
    fprintf(stderr, "bf build: %s\n", cuvsGetLastErrorText()); return 5;
  }
  double t3 = now_sec();

  if (cuvsRMMAlloc(res, (void**)&nb_dev, sizeof(unsigned) * (size_t)n * kp1) != CUVS_SUCCESS) { fprintf(stderr, "alloc nb\n"); return 4; }
  if (cuvsRMMAlloc(res, (void**)&dist_dev, sizeof(float) * (size_t)n * kp1) != CUVS_SUCCESS) { fprintf(stderr, "alloc dd\n"); return 4; }

  DLManagedTensor qt, nbt, dt;
  fill_tensor(&qt, ds_dev, 1, n, dim, 0);           /* queries = dataset itself */
  fill_tensor_i64(&nbt, nb_dev, n, kp1);
  fill_tensor(&dt, dist_dev, 1, n, kp1, 0);
  cuvsFilter f; f.type = NO_FILTER; f.addr = (uintptr_t)0;

  double t4 = now_sec();
  if (cuvsBruteForceSearch(res, bf, &qt, &nbt, &dt, f) != CUVS_SUCCESS) {
    fprintf(stderr, "bf search: %s\n", cuvsGetLastErrorText()); return 6;
  }
  double t5 = now_sec();

  cudaMemcpy(nb, nb_dev, sizeof(unsigned) * (size_t)n * kp1, cudaMemcpyDefault);
  double t6 = now_sec();

  /* ---- drop self, build graph ---- */
  for (long i = 0; i < n; i++) {
    const long long* row = nb + (size_t)i * kp1;
    unsigned* out = graph + (size_t)i * degree;
    long w = 0;
    for (long c = 0; c < kp1 && w < degree; c++)
      if (row[c] != (long long)i) out[w++] = (unsigned)row[c];
    while (w < degree) out[w++] = (unsigned)row[kp1 - 1];
  }
  double t7 = now_sec();

  if (cuvsRMMAlloc(res, (void**)&graph_dev, sizeof(unsigned) * (size_t)n * degree) != CUVS_SUCCESS) { fprintf(stderr, "alloc g\n"); return 4; }
  cudaMemcpy(graph_dev, graph, sizeof(unsigned) * (size_t)n * degree, cudaMemcpyDefault);
  double t8 = now_sec();

  printf("[bfknn] upload %.3fs | bfBuild %.3fs | bfSearch %.3fs | d2h %.3fs | selfdrop %.3fs | g2d %.3fs\n",
         t1 - t0, t3 - t2, t5 - t4, t6 - t5, t7 - t6, t8 - t7);
  printf("[bfknn] GRAPH_TOTAL (upload+build+search+post+upload) = %.3f s\n", t8 - t0);

  if (selftest) {
    long rows = n < 20 ? n : 20;
    long bad = 0;
    for (long ri = 0; ri < rows; ri++) {
      long i = (long)((double)ri * (double)(n - 1) / (double)(rows > 1 ? rows - 1 : 1));
      float bd[MAXK]; long bi[MAXK];
      const float* q = base + (size_t)i * dim;
      for (long t = 0; t < degree; t++) { bd[t] = 3.4e38f; bi[t] = -1; }
      for (long j = 0; j < n; j++) {
        if (j == i) continue;
        const float* v = base + (size_t)j * dim;
        float acc = 0.0f;
        for (long t = 0; t < dim; t++) { float df = q[t] - v[t]; acc += df * df; }
        topk_push(bd, bi, degree, acc, j);
      }
      const unsigned* g = graph + (size_t)i * degree;
      for (long t = 0; t < degree; t++) {
        int found = 0;
        for (long c = 0; c < degree; c++) if (g[c] == (unsigned)bi[t]) { found = 1; break; }
        if (!found) bad++;
      }
    }
    printf("[bfknn] selftest: %ld/%ld exact neighbors missing\n", bad, rows * degree);
    printf(bad == 0 ? "SELFTEST PASS\n" : "SELFTEST FAIL\n");
  }

  cuvsBruteForceIndexDestroy(bf);
  cuvsRMMFree(res, nb_dev, sizeof(unsigned) * (size_t)n * kp1);
  cuvsRMMFree(res, dist_dev, sizeof(float) * (size_t)n * kp1);

  /* ---- FromArgs -> CAGRA ---- */
  cuvsCagraIndex_t idx = 0;
  if (cuvsCagraIndexCreate(&idx) != CUVS_SUCCESS) { fprintf(stderr, "idx: %s\n", cuvsGetLastErrorText()); return 7; }
  DLManagedTensor gt_t;
  fill_tensor(&gt_t, graph_dev, 1, n, degree, 1);
  double t9 = now_sec();
  cuvsError_t rc = cuvsCagraIndexFromArgs(res, (cuvsDistanceType)0, &gt_t, &ds_t, idx);
  double t10 = now_sec();
  printf("[bfknn] FromArgs rc=%d (%.3fs): %s\n", (int)rc, t10 - t9,
         rc == CUVS_SUCCESS ? "ok" : cuvsGetLastErrorText());
  if (rc != CUVS_SUCCESS) { printf("RESULT FAIL fromargs\n"); return 8; }

  int64_t sn = 0, sd = 0;
  cuvsCagraIndexGetSize(idx, &sn);
  cuvsCagraIndexGetDims(idx, &sd);
  printf("[bfknn] cagra index size=%ld dims=%ld\n", (long)sn, (long)sd);

  /* recall@10 and recall@100 vs CPU exact GT */
  exact_gt_rows(base, n, dim, qs, 0, nq, 10, gt10);
  if (cagra_search_batch(res, idx, dim, qs, nq, 10, got, itopk) != 0) { printf("RESULT FAIL search10\n"); return 9; }
  printf("[bfknn] recall@10 = %.4f\n", recall_of(got, gt10, nq, 10));

  if (n >= 100) {
    exact_gt_rows(base, n, dim, qs, 0, nq, 100, gt100);
    if (cagra_search_batch(res, idx, dim, qs, nq, 100, got, itopk) != 0) { printf("RESULT FAIL search100\n"); return 9; }
    printf("[bfknn] recall@100 = %.4f\n", recall_of(got, gt100, nq, 100));
  }

  if (do_qps) {
    long qn = 4096;
    float* qbig = (float*)malloc(sizeof(float) * (size_t)qn * dim);
    unsigned* obig = (unsigned*)malloc(sizeof(unsigned) * (size_t)qn * 100);
    for (long i = 0; i < qn * dim; i++) qbig[i] = frand();
    for (int tk = 10; tk <= 100; tk += 90) {
      cagra_search_batch(res, idx, dim, qbig, qn, tk, obig, itopk); /* warmup */
      double a = now_sec();
      int iters = 50;
      for (int it = 0; it < iters; it++)
        cagra_search_batch(res, idx, dim, qbig, qn, tk, obig, itopk);
      double b = now_sec();
      double qps = (double)iters * (double)qn / (b - a);
      printf("[bfknn] QPS topk=%d: %.1f (nq=%ld, %d iters, %.3f s total)\n", tk, qps, qn, iters, b - a);
    }
    free(qbig); free(obig);
  }

  printf("RESULT PASS\n");

  cuvsCagraIndexDestroy(idx);
  cuvsRMMFree(res, graph_dev, sizeof(unsigned) * (size_t)n * degree);
  cuvsRMMFree(res, ds_dev, sizeof(float) * (size_t)n * dim);
  cuvsResourcesDestroy(res);
  free(base); free(qs); free(nb); free(graph); free(gt10); free(gt100); free(got);
  return 0;
}
