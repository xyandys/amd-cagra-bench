/* bfknn2.cu — handoff §6 step 2, custom route
 *
 * cuvsBruteForceSearch is broken in this hipVS build (illegal address +
 * rocBLASLt Tensile init failure), so we build the exact kNN graph with our
 * own fused tiled-distance + top-k HIP kernel. No rocBLAS anywhere.
 *
 * Design:
 *   kernel A: grid (n/1024 row-blocks, n/32768 col-bands). Each thread owns
 *   4 rows; column tiles of 16 x dim-chunks of 16 accumulate squared L2 in
 *   registers; a 64-slot unsorted candidate list per row lives in scratch
 *   (compare against a register-resident max, insert by replace+rescan —
 *   inserts are rare, O(k ln(n/k)) per row). Self excluded inline. Per-band
 *   partial lists go to global memory.
 *   kernel B: one thread per row merges the band partials into the final
 *   (n, 64) uint32 graph.
 *
 * Then: cuvsCagraIndexFromArgs -> official cuvsCagraSearch, recall@10/@100,
 * optional QPS loops at nq=4096.
 *
 * Usage: ./bfknn2 [n] [dim] [ignored_degree] [nq] [topk] [itopk]
 * Env:   BF_SELFTEST=1, BF_QPS=1
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

#define KDEG 64
#define CT 16
#define DC 16
#define RPT 4
#define NTHREADS 256
#define RPB (NTHREADS * RPT)
#define CBAND 32768L

#define CUCK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA err %s at line %d\n", cudaGetErrorString(e_), __LINE__); return 9; } } while (0)

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

__global__ void knn_dist_topk(const float* __restrict__ base, long n, long dim,
                              unsigned* __restrict__ pid, float* __restrict__ pval)
{
  __shared__ float bt[CT][DC];
  const long m = n;
  const long band = blockIdx.y;
  const long band0 = (long)band * CBAND;
  const long bandn = band0 + CBAND < n ? band0 + CBAND : n;
  const long rowq0 = (long)blockIdx.x * RPB;
  const int tx = threadIdx.x;

  float lv[RPT][KDEG];
  unsigned li[RPT][KDEG];
  float cmax[RPT]; int cpos[RPT];

#pragma unroll
  for (int r = 0; r < RPT; r++) {
#pragma unroll
    for (int t = 0; t < KDEG; t++) { lv[r][t] = 3.4e38f; li[r][t] = 0xFFFFFFFFu; }
    cmax[r] = 3.4e38f; cpos[r] = 0;
  }

  for (long tile = band0; tile < bandn; tile += CT) {
    float acc[RPT][CT];
#pragma unroll
    for (int r = 0; r < RPT; r++)
#pragma unroll
      for (int c = 0; c < CT; c++) acc[r][c] = 0.f;

    for (long d0 = 0; d0 < dim; d0 += DC) {
      for (int i = tx; i < CT * DC; i += NTHREADS) {
        int c = i / DC, d = i % DC;
        long col = tile + c;
        long dd = d0 + d;
        bt[c][d] = (col < n && dd < dim) ? base[col * dim + dd] : 0.f;
      }
      __syncthreads();
      float qv[RPT][DC];
#pragma unroll
      for (int r = 0; r < RPT; r++) {
        long row = rowq0 + r * NTHREADS + tx;
#pragma unroll
        for (int d = 0; d < DC; d++)
          qv[r][d] = (row < m && d0 + d < dim) ? base[row * dim + d0 + d] : 0.f;
      }
#pragma unroll
      for (int c = 0; c < CT; c++)
#pragma unroll
        for (int d = 0; d < DC; d++) {
          float b = bt[c][d];
#pragma unroll
          for (int r = 0; r < RPT; r++) {
            float df = qv[r][d] - b;
            acc[r][c] += df * df;
          }
        }
      __syncthreads();
    }

#pragma unroll
    for (int c = 0; c < CT; c++) {
      long col = tile + c;
      if (col >= n) continue;
#pragma unroll
      for (int r = 0; r < RPT; r++) {
        long row = rowq0 + r * NTHREADS + tx;
        if (row >= m || col == row) continue;
        float dv = acc[r][c];
        if (dv < cmax[r]) {
          lv[r][cpos[r]] = dv; li[r][cpos[r]] = (unsigned)col;
          float bm = -1.f; int bp = 0;
#pragma unroll
          for (int t = 0; t < KDEG; t++)
            if (lv[r][t] > bm) { bm = lv[r][t]; bp = t; }
          cmax[r] = bm; cpos[r] = bp;
        }
      }
    }
  }

#pragma unroll
  for (int r = 0; r < RPT; r++) {
    long row = rowq0 + r * NTHREADS + tx;
    if (row >= m) continue;
    unsigned* pib = pid + ((size_t)band * m + row) * KDEG;
    float* pvb = pval + ((size_t)band * m + row) * KDEG;
#pragma unroll
    for (int t = 0; t < KDEG; t++) { pib[t] = li[r][t]; pvb[t] = lv[r][t]; }
  }
}

__global__ void knn_merge(const unsigned* __restrict__ pid, const float* __restrict__ pval,
                          long m, long nbands, unsigned* __restrict__ graph, long n)
{
  long row = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m) return;
  float lv[KDEG]; unsigned lgi[KDEG];
  float cmax = 3.4e38f; int cpos = 0;
#pragma unroll
  for (int t = 0; t < KDEG; t++) { lv[t] = 3.4e38f; lgi[t] = 0xFFFFFFFFu; }
  for (long band = 0; band < nbands; band++) {
    const unsigned* pib = pid + ((size_t)band * m + row) * KDEG;
    const float* pvb = pval + ((size_t)band * m + row) * KDEG;
#pragma unroll
    for (int t = 0; t < KDEG; t++) {
      float dv = pvb[t]; unsigned c = pib[t];
      if (c < (unsigned)n && c != (unsigned)row && dv < cmax) {
        lv[cpos] = dv; lgi[cpos] = c;
        float bm = -1.f; int bp = 0;
#pragma unroll
        for (int u = 0; u < KDEG; u++)
          if (lv[u] > bm) { bm = lv[u]; bp = u; }
        cmax = bm; cpos = bp;
      }
    }
  }
  unsigned* g = graph + (size_t)row * KDEG;
#pragma unroll
  for (int t = 0; t < KDEG; t++)
    g[t] = (lgi[t] < (unsigned)n) ? lgi[t] : (unsigned)((row + 1 + t) % n);
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
  t->dl_tensor.strides = (int64_t*)malloc(sizeof(int64_t) * 2);
  t->dl_tensor.strides[0] = cols; t->dl_tensor.strides[1] = 1;
  t->dl_tensor.byte_offset = 0;
}

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

static void exact_gt_rows(const float* base, long n, long dim, const float* qs,
                          long nq, long k, long* gt) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 4)
#endif
  for (long qi = 0; qi < nq; qi++) {
    float bd[128]; long bi[128];
    const float* q = qs + (size_t)qi * dim;
    for (long t = 0; t < k; t++) { bd[t] = 3.4e38f; bi[t] = -1; }
    for (long j = 0; j < n; j++) {
      const float* v = base + (size_t)j * dim;
      float acc = 0.0f;
      for (long t = 0; t < dim; t++) { float df = q[t] - v[t]; acc += df * df; }
      topk_push(bd, bi, k, acc, j);
    }
    for (long t = 0; t < k; t++) gt[(size_t)qi * k + t] = bi[t];
  }
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
  long nq     = argc > 4 ? atol(argv[4]) : 300;
  long topk   = argc > 5 ? atol(argv[5]) : 10;
  long itopk  = argc > 6 ? atol(argv[6]) : 512;
  (void)argc; (void)argv;
  int selftest = getenv("BF_SELFTEST") && atoi(getenv("BF_SELFTEST")) == 1;
  int do_qps   = getenv("BF_QPS") && atoi(getenv("BF_QPS")) == 1;

  float* base = (float*)malloc(sizeof(float) * (size_t)n * dim);
  float* qs   = (float*)malloc(sizeof(float) * (size_t)nq * dim);
  unsigned* graph = (unsigned*)malloc(sizeof(unsigned) * (size_t)n * KDEG);
  long* gt10 = (long*)malloc(sizeof(long) * (size_t)nq * 10);
  long* gt100 = (long*)malloc(sizeof(long) * (size_t)nq * 100);
  unsigned* got = (unsigned*)malloc(sizeof(unsigned) * (size_t)nq * 100);
  if (!base || !qs || !graph || !gt10 || !gt100 || !got) { fprintf(stderr, "OOM(host)\n"); return 2; }

  for (long i = 0; i < n * dim; i++) base[i] = frand();
  for (long i = 0; i < nq * dim; i++) qs[i] = frand();
  printf("[bfknn2] n=%ld dim=%ld degree=%d nq=%ld itopk=%ld\n", n, dim, KDEG, nq, itopk);

  long nbands = (n + CBAND - 1) / CBAND;
  float* ds_dev = 0; unsigned* pid = 0; float* pval = 0; unsigned* graph_dev = 0;
  CUCK(cudaMalloc(&ds_dev, sizeof(float) * (size_t)n * dim));
  CUCK(cudaMalloc(&pid, sizeof(unsigned) * (size_t)nbands * n * KDEG));
  CUCK(cudaMalloc(&pval, sizeof(float) * (size_t)nbands * n * KDEG));
  CUCK(cudaMalloc(&graph_dev, sizeof(unsigned) * (size_t)n * KDEG));

  double t0 = now_sec();
  CUCK(cudaMemcpy(ds_dev, base, sizeof(float) * (size_t)n * dim, cudaMemcpyDefault));
  double t1 = now_sec();

  dim3 grid((unsigned)((n + RPB - 1) / RPB), (unsigned)nbands);
  double t2 = now_sec();
  knn_dist_topk<<<grid, NTHREADS>>>(ds_dev, n, dim, pid, pval);
  CUCK(cudaGetLastError());
  CUCK(cudaDeviceSynchronize());
  double t3 = now_sec();
  knn_merge<<<(unsigned)((n + 255) / 256), 256>>>(pid, pval, n, nbands, graph_dev, n);
  CUCK(cudaGetLastError());
  CUCK(cudaDeviceSynchronize());
  double t4 = now_sec();
  CUCK(cudaMemcpy(graph, graph_dev, sizeof(unsigned) * (size_t)n * KDEG, cudaMemcpyDefault));
  double t5 = now_sec();

  printf("[bfknn2] upload %.3fs | dist_topk %.3fs | merge %.3fs | d2h %.3fs\n",
         t1 - t0, t3 - t2, t4 - t3, t5 - t4);
  printf("[bfknn2] GRAPH_TOTAL = %.3f s\n", t5 - t0);

  if (selftest) {
    long rows = n < 20 ? n : 20;
    long bad = 0;
    for (long ri = 0; ri < rows; ri++) {
      long i = (long)((double)ri * (double)(n - 1) / (double)(rows > 1 ? rows - 1 : 1));
      float bd[KDEG]; long bi[KDEG];
      const float* q = base + (size_t)i * dim;
      for (long t = 0; t < KDEG; t++) { bd[t] = 3.4e38f; bi[t] = -1; }
      for (long j = 0; j < n; j++) {
        if (j == i) continue;
        const float* v = base + (size_t)j * dim;
        float acc = 0.0f;
        for (long t = 0; t < dim; t++) { float df = q[t] - v[t]; acc += df * df; }
        topk_push(bd, bi, KDEG, acc, j);
      }
      const unsigned* g = graph + (size_t)i * KDEG;
      for (long t = 0; t < KDEG; t++) {
        int found = 0;
        for (long c = 0; c < KDEG; c++) if (g[c] == (unsigned)bi[t]) { found = 1; break; }
        if (!found) bad++;
      }
    }
    printf("[bfknn2] selftest: %ld/%ld exact neighbors missing\n", bad, rows * KDEG);
    printf(bad == 0 ? "SELFTEST PASS\n" : "SELFTEST FAIL\n");
  }

  cuvsResources_t res = 0;
  if (cuvsResourcesCreate(&res) != CUVS_SUCCESS) { fprintf(stderr, "res: %s\n", cuvsGetLastErrorText()); return 3; }

  DLManagedTensor ds_t, gt_t;
  fill_tensor(&ds_t, ds_dev, 1, n, dim, 0);
  fill_tensor(&gt_t, graph_dev, 1, n, KDEG, 1);
  cuvsCagraIndex_t idx = 0;
  if (cuvsCagraIndexCreate(&idx) != CUVS_SUCCESS) { fprintf(stderr, "idx: %s\n", cuvsGetLastErrorText()); return 7; }
  double t6 = now_sec();
  cuvsError_t rc = cuvsCagraIndexFromArgs(res, (cuvsDistanceType)0, &gt_t, &ds_t, idx);
  double t7 = now_sec();
  printf("[bfknn2] FromArgs rc=%d (%.3fs): %s\n", (int)rc, t7 - t6,
         rc == CUVS_SUCCESS ? "ok" : cuvsGetLastErrorText());
  if (rc != CUVS_SUCCESS) { printf("RESULT FAIL fromargs\n"); return 8; }

  int64_t sn = 0, sd = 0;
  cuvsCagraIndexGetSize(idx, &sn);
  cuvsCagraIndexGetDims(idx, &sd);
  printf("[bfknn2] cagra index size=%ld dims=%ld\n", (long)sn, (long)sd);

  exact_gt_rows(base, n, dim, qs, nq, 10, gt10);
  if (cagra_search_batch(res, idx, dim, qs, nq, 10, got, itopk) != 0) { printf("RESULT FAIL search10\n"); return 9; }
  printf("[bfknn2] recall@10 = %.4f\n", recall_of(got, gt10, nq, 10));

  if (n >= 100) {
    exact_gt_rows(base, n, dim, qs, nq, 100, gt100);
    if (cagra_search_batch(res, idx, dim, qs, nq, 100, got, itopk) != 0) { printf("RESULT FAIL search100\n"); return 9; }
    printf("[bfknn2] recall@100 = %.4f\n", recall_of(got, gt100, nq, 100));
  }

  if (do_qps) {
    long qn = 4096;
    float* qbig = (float*)malloc(sizeof(float) * (size_t)qn * dim);
    unsigned* obig = (unsigned*)malloc(sizeof(unsigned) * (size_t)qn * 100);
    for (long i = 0; i < qn * dim; i++) qbig[i] = frand();
    for (int tk = 10; tk <= 100; tk += 90) {
      cagra_search_batch(res, idx, dim, qbig, qn, tk, obig, itopk);
      double a = now_sec();
      int iters = 50;
      for (int it = 0; it < iters; it++)
        cagra_search_batch(res, idx, dim, qbig, qn, tk, obig, itopk);
      double b = now_sec();
      printf("[bfknn2] QPS topk=%d: %.1f (nq=%ld, %d iters, %.3f s)\n", tk,
             (double)iters * (double)qn / (b - a), qn, iters, b - a);
    }
    free(qbig); free(obig);
  }

  printf("RESULT PASS\n");
  cuvsCagraIndexDestroy(idx);
  cuvsResourcesDestroy(res);
  cudaFree(pid); cudaFree(pval); cudaFree(graph_dev); cudaFree(ds_dev);
  free(base); free(qs); free(graph); free(gt10); free(gt100); free(got);
  return 0;
}
