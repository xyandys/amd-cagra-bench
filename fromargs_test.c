/* fromargs_test.c — handoff §6 step 1
 *
 * Question: can cuvsCagraIndexFromArgs() accept a self-built exact kNN graph
 * and still serve the official cuvsCagraSearch() with full recall?
 * Small scale on purpose: CPU brute-force exact kNN, default 5000 x 32.
 *
 * Usage: ./fromargs_test [n] [dim] [degree] [nq] [topk] [itopk]
 * Env:   CUVS_METRIC (cuvsDistanceType numeric value, default 0)
 *        GRAPH_ON_DEVICE=1 to hand the graph over as a device tensor.
 *
 * Pass criteria: prints "RESULT PASS" with recall@topk >= 0.99.
 */
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/cagra.h>
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
static float frand(void) { /* xorshift -> [0,1) */
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

/* exact kNN graph rows (squared L2, self excluded) */
static void exact_graph(const float* base, long n, long dim, long degree, unsigned* graph) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 64)
#endif
  for (long i = 0; i < n; i++) {
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
    for (long t = 0; t < degree; t++)
      graph[(size_t)i * degree + t] = (unsigned)(bi[t] >= 0 ? bi[t] : 0);
  }
}

/* exact ground truth for query row qi */
static void exact_gt(const float* base, long n, long dim, const float* q, long qi, long k, long* gt) {
  float bd[MAXK]; long bi[MAXK];
  const float* qq = q + (size_t)qi * dim;
  for (long t = 0; t < k; t++) { bd[t] = 3.4e38f; bi[t] = -1; }
  for (long j = 0; j < n; j++) {
    const float* v = base + (size_t)j * dim;
    float acc = 0.0f;
    for (long t = 0; t < dim; t++) { float df = qq[t] - v[t]; acc += df * df; }
    topk_push(bd, bi, k, acc, j);
  }
  for (long t = 0; t < k; t++) gt[(size_t)qi * k + t] = bi[t];
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

int main(int argc, char** argv) {
  long n      = argc > 1 ? atol(argv[1]) : 5000;
  long dim    = argc > 2 ? atol(argv[2]) : 32;
  long degree = argc > 3 ? atol(argv[3]) : 32;
  long nq     = argc > 4 ? atol(argv[4]) : 500;
  long topk   = argc > 5 ? atol(argv[5]) : 10;
  long itopk  = argc > 6 ? atol(argv[6]) : 512;
  if (degree > MAXK) degree = MAXK;
  if (topk > MAXK) topk = MAXK;

  const char* me = getenv("CUVS_METRIC");
  long metric_v = (me && *me) ? strtol(me, NULL, 10) : 0;
  int graph_on_device = getenv("GRAPH_ON_DEVICE") && atoi(getenv("GRAPH_ON_DEVICE")) == 1;

  float* base = (float*)malloc(sizeof(float) * (size_t)n * dim);
  float* qs   = (float*)malloc(sizeof(float) * (size_t)nq * dim);
  unsigned* graph = (unsigned*)malloc(sizeof(unsigned) * (size_t)n * degree);
  long* gt = (long*)malloc(sizeof(long) * (size_t)nq * topk);
  unsigned* got = (unsigned*)malloc(sizeof(unsigned) * (size_t)nq * topk);
  if (!base || !qs || !graph || !gt || !got) { fprintf(stderr, "OOM(host)\n"); return 2; }

  for (long i = 0; i < n * dim; i++) base[i] = frand();
  for (long i = 0; i < nq * dim; i++) qs[i] = frand();

  double t0 = now_sec();
  exact_graph(base, n, dim, degree, graph);
  double t1 = now_sec();

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (long qi = 0; qi < nq; qi++) exact_gt(base, n, dim, qs, qi, topk, gt);
  double t2 = now_sec();
  printf("[fromargs] n=%ld dim=%ld degree=%ld nq=%ld topk=%ld itopk=%ld metric=%ld device_graph=%d\n",
         n, dim, degree, nq, topk, itopk, metric_v, graph_on_device);
  printf("[fromargs] exact graph %.3fs, query GT %.3fs (CPU)\n", t1 - t0, t2 - t1);

  cuvsResources_t res = 0; cuvsCagraIndex_t idx = 0;
  if (cuvsResourcesCreate(&res) != CUVS_SUCCESS) { fprintf(stderr, "res create: %s\n", cuvsGetLastErrorText()); return 3; }
  if (cuvsCagraIndexCreate(&idx) != CUVS_SUCCESS) { fprintf(stderr, "idx create: %s\n", cuvsGetLastErrorText()); return 3; }

  void* g_dev = 0; void* d_dev = 0;
  if (graph_on_device) {
    if (cudaMalloc(&g_dev, sizeof(unsigned) * (size_t)n * degree) != cudaSuccess ||
        cudaMalloc(&d_dev, sizeof(float) * (size_t)n * dim) != cudaSuccess) {
      fprintf(stderr, "cudaMalloc failed\n"); return 4;
    }
    cudaMemcpy(g_dev, graph, sizeof(unsigned) * (size_t)n * degree, cudaMemcpyDefault);
    cudaMemcpy(d_dev, base, sizeof(float) * (size_t)n * dim, cudaMemcpyDefault);
  }

  DLManagedTensor gt_t, ds_t;
  fill_tensor(&gt_t, graph_on_device ? g_dev : (void*)graph, graph_on_device, n, degree, 1);
  fill_tensor(&ds_t, graph_on_device ? d_dev : (void*)base, graph_on_device, n, dim, 0);

  double t3 = now_sec();
  cuvsError_t rc = cuvsCagraIndexFromArgs(res, (cuvsDistanceType)metric_v, &gt_t, &ds_t, idx);
  double t4 = now_sec();
  printf("[fromargs] cuvsCagraIndexFromArgs rc=%d (%.3fs): %s\n", (int)rc, t4 - t3,
         rc == CUVS_SUCCESS ? "ok" : cuvsGetLastErrorText());
  if (rc != CUVS_SUCCESS) { printf("RESULT FAIL fromargs\n"); return 5; }

  int64_t sn = 0, sd = 0;
  cuvsCagraIndexGetSize(idx, &sn);
  cuvsCagraIndexGetDims(idx, &sd);
  printf("[fromargs] index size=%ld dims=%ld\n", (long)sn, (long)sd);

  /* search path mirrors tools/hipvs_bridge/seekdb_cuvs_bridge.c */
  float* qd = 0; unsigned* nd = 0; float* dd = 0; cuvsCagraSearchParams_t sp = 0;
  if (cuvsRMMAlloc(res, (void**)&qd, sizeof(float) * (size_t)nq * dim) != CUVS_SUCCESS) { printf("RESULT FAIL alloc\n"); return 6; }
  if (cuvsRMMAlloc(res, (void**)&nd, sizeof(unsigned) * (size_t)nq * topk) != CUVS_SUCCESS) { printf("RESULT FAIL alloc\n"); return 6; }
  if (cuvsRMMAlloc(res, (void**)&dd, sizeof(float) * (size_t)nq * topk) != CUVS_SUCCESS) { printf("RESULT FAIL alloc\n"); return 6; }
  cudaMemcpy(qd, qs, sizeof(float) * (size_t)nq * dim, cudaMemcpyDefault);

  DLManagedTensor qt, nt, dt;
  fill_tensor(&qt, qd, 1, nq, dim, 0);
  fill_tensor(&nt, nd, 1, nq, topk, 1);
  fill_tensor(&dt, dd, 1, nq, topk, 0);

  if (cuvsCagraSearchParamsCreate(&sp) != CUVS_SUCCESS) { printf("RESULT FAIL sp: %s\n", cuvsGetLastErrorText()); return 6; }
  sp->itopk_size = (size_t)itopk;
  sp->search_width = 4;
  cuvsFilter f; f.type = NO_FILTER; f.addr = (uintptr_t)0;

  double t5 = now_sec();
  cuvsError_t src = cuvsCagraSearch(res, sp, idx, &qt, &nt, &dt, f);
  double t6 = now_sec();
  printf("[fromargs] cuvsCagraSearch rc=%d (%.3fs): %s\n", (int)src, t6 - t5,
         src == CUVS_SUCCESS ? "ok" : cuvsGetLastErrorText());
  if (src != CUVS_SUCCESS) { printf("RESULT FAIL search\n"); return 7; }

  cudaMemcpy(got, nd, sizeof(unsigned) * (size_t)nq * topk, cudaMemcpyDefault);

  long hits = 0;
  for (long qi = 0; qi < nq; qi++)
    for (long r = 0; r < topk; r++)
      for (long c = 0; c < topk; c++)
        if (got[(size_t)qi * topk + c] == (unsigned)gt[(size_t)qi * topk + r]) { hits++; break; }
  double recall = (double)hits / (double)(nq * topk);
  printf("[fromargs] recall@%ld = %.4f\n", topk, recall);
  printf(recall >= 0.99 ? "RESULT PASS\n" : "RESULT FAIL recall\n");

  cuvsCagraSearchParamsDestroy(sp);
  cuvsRMMFree(res, dd, sizeof(float) * (size_t)nq * topk);
  cuvsRMMFree(res, nd, sizeof(unsigned) * (size_t)nq * topk);
  cuvsRMMFree(res, qd, sizeof(float) * (size_t)nq * dim);
  cuvsCagraIndexDestroy(idx);
  cuvsResourcesDestroy(res);
  if (g_dev) cudaFree(g_dev);
  if (d_dev) cudaFree(d_dev);
  return recall >= 0.99 ? 0 : 1;
}
