/* seekdb <-> hipVS bridge: plain-C CAGRA kNN; all cuVS/ROCm deps live here.
 *
 * Optimized variant for the AMD hipVS CAGRA challenge (drop-in replacement for
 * tools/hipvs_bridge/seekdb_cuvs_bridge.c; the public C ABI in
 * seekdb_cuvs_bridge.h is unchanged).
 *
 * What changed vs. the frozen baseline (634888d9):
 *  1. Query staging: the three device buffers (query / ids / dists) are
 *     allocated once per index handle and reused across queries instead of
 *     three cuvsRMMAlloc + three cuvsRMMFree per query.  hipMalloc/hipFree are
 *     synchronous and cost hundreds of microseconds each on ROCm, which is a
 *     large share of the ~7-12 ms single-row query latency.
 *  2. cuvsCagraSearchParams is created once per handle and re-tuned in place
 *     when top-k changes, instead of create+destroy on every query.
 *  3. Build knobs are overridable through SEEKDB_CUVS_* env variables so the
 *     same binary can benchmark several build presets on the contest machine.
 *     Defaults reproduce the frozen baseline graph (128 / 256 / 100 iters).
 *  4. SEEKDB_CUVS_TRACE=1 prints per-phase timing (build, serialize, search)
 *     to stderr; useful when comparing presets.
 *
 * The search path keeps the baseline tuning rule (itopk >= max(4*topk, 512),
 * width 4) unless SEEKDB_CUVS_ITOPK / SEEKDB_CUVS_WIDTH override it, so the
 * default build+search results are identical to the frozen baseline.
 */
#include "seekdb_cuvs_bridge.h"
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/cagra.h>
#include <dlpack/dlpack.h>
#ifdef __HIP_PLATFORM_AMD__
#include <cuvs/cuda_runtime.h>
#else
#include <cuda_runtime.h>
#endif
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

typedef struct {
  cuvsResources_t        res;
  cuvsCagraIndexParams_t ip;
  cuvsCagraIndex_t       idx;
  long n, dim;

  /* persistent query staging (grown on demand) */
  float*    q_dev;
  size_t    q_cap; /* in floats */
  unsigned* ids_dev;
  size_t    ids_cap; /* in elements */
  float*    dist_dev;
  size_t    dist_cap; /* in elements */
  long      staged_topk; /* top-k the staging buffers were sized for */
  cuvsCagraSearchParams_t sp;
  long      sp_topk; /* top-k the cached search params were tuned for */
  pthread_mutex_t mu;
} sk_cuvs_index;

static int trace_enabled(void)
{
  static int cached = -1;
  if (cached < 0) {
    const char* v = getenv("SEEKDB_CUVS_TRACE");
    cached = (v && v[0] == '1') ? 1 : 0;
  }
  return cached;
}

static long now_us(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (long)ts.tv_sec * 1000000L + ts.tv_nsec / 1000L;
}

static long env_long(const char* name, long def, long lo, long hi)
{
  const char* v = getenv(name);
  if (!v || !*v) return def;
  char* end = NULL;
  long x = strtol(v, &end, 10);
  if (end == v) return def;
  if (x < lo) x = lo;
  if (x > hi) x = hi;
  return x;
}

/* Build knobs: defaults == frozen baseline. */
static void load_build_params(cuvsCagraIndexParams_t ip)
{
  ip->build_algo                = NN_DESCENT;
  ip->graph_degree              = (size_t)env_long("SEEKDB_CUVS_GRAPH_DEGREE", 128, 16, 512);
  ip->intermediate_graph_degree = (size_t)env_long("SEEKDB_CUVS_INTERMEDIATE_DEGREE", 256, 16, 1024);
  ip->nn_descent_niter          = (size_t)env_long("SEEKDB_CUVS_NITER", 100, 1, 1000);
  if (trace_enabled()) {
    fprintf(stderr, "[seekdb_cuvs] build params: graph_degree=%zu intermediate=%zu niter=%zu\n",
            ip->graph_degree, ip->intermediate_graph_degree, ip->nn_descent_niter);
  }
}

static int ensure_staging(sk_cuvs_index* h, long topk)
{
  size_t need_q = (size_t)h->dim; /* single-row queries only need dim; grow lazily */
  size_t need_k = (size_t)topk;
  if (h->q_cap < need_q) {
    float* p = NULL;
    if (cuvsRMMAlloc(h->res, (void**)&p, sizeof(float) * need_q) != CUVS_SUCCESS) return 1;
    if (h->q_dev) cuvsRMMFree(h->res, h->q_dev, sizeof(float) * h->q_cap);
    h->q_dev = p;
    h->q_cap = need_q;
  }
  if (h->ids_cap < need_k) {
    unsigned* p = NULL;
    if (cuvsRMMAlloc(h->res, (void**)&p, sizeof(unsigned) * need_k) != CUVS_SUCCESS) return 2;
    if (h->ids_dev) cuvsRMMFree(h->res, h->ids_dev, sizeof(unsigned) * h->ids_cap);
    h->ids_dev = p;
    h->ids_cap = need_k;
  }
  if (h->dist_cap < need_k) {
    float* p = NULL;
    if (cuvsRMMAlloc(h->res, (void**)&p, sizeof(float) * need_k) != CUVS_SUCCESS) return 3;
    if (h->dist_dev) cuvsRMMFree(h->res, h->dist_dev, sizeof(float) * h->dist_cap);
    h->dist_dev = p;
    h->dist_cap = need_k;
  }
  h->staged_topk = topk;
  return 0;
}

static int tune_search_params(sk_cuvs_index* h, long topk)
{
  if (!h->sp) {
    if (cuvsCagraSearchParamsCreate(&h->sp) != CUVS_SUCCESS) return 1;
    h->sp_topk = -1;
  }
  if (h->sp_topk == topk) return 0;
  /* Baseline rule preserved by default: itopk = max(4*topk, 512), width = 4. */
  long itopk_env  = env_long("SEEKDB_CUVS_ITOPK", 0, 0, 4096);
  long width_env  = env_long("SEEKDB_CUVS_WIDTH", 0, 0, 64);
  size_t itopk    = itopk_env > 0 ? (size_t)itopk_env : (size_t)(topk * 4);
  size_t width    = width_env > 0 ? (size_t)width_env : 4;
  if (itopk_env <= 0 && itopk < 512) itopk = 512;
  if (itopk < (size_t)topk) itopk = (size_t)topk;
  h->sp->itopk_size    = itopk;
  h->sp->search_width  = width;
  h->sp_topk           = topk;
  return 0;
}

void* seekdb_cuvs_build(const float* base, long n, long dim)
{
  sk_cuvs_index* h = (sk_cuvs_index*)calloc(1, sizeof(*h));
  if (!h) return NULL;
  pthread_mutex_init(&h->mu, NULL);
  if (cuvsResourcesCreate(&h->res) != CUVS_SUCCESS) { free(h); return NULL; }

  DLManagedTensor dset; long ds[2] = {n, dim};
  memset(&dset, 0, sizeof(dset));
  dset.dl_tensor.data = (void*)base;
  dset.dl_tensor.device.device_type = kDLCPU; dset.dl_tensor.device.device_id = 0;
  dset.dl_tensor.ndim = 2;
  dset.dl_tensor.dtype.code = kDLFloat; dset.dl_tensor.dtype.bits = 32; dset.dl_tensor.dtype.lanes = 1;
  dset.dl_tensor.shape = ds; dset.dl_tensor.strides = 0; dset.dl_tensor.byte_offset = 0;

  if (cuvsCagraIndexParamsCreate(&h->ip) != CUVS_SUCCESS) goto fail_res;
  load_build_params(h->ip);
  if (cuvsCagraIndexCreate(&h->idx) != CUVS_SUCCESS) goto fail_ip;
  {
    long t0 = now_us();
    if (cuvsCagraBuild(h->res, h->ip, &dset, h->idx) != CUVS_SUCCESS) goto fail_idx;
    if (trace_enabled()) {
      fprintf(stderr, "[seekdb_cuvs] cagra_build n=%ld dim=%ld took %.1f ms\n",
              n, dim, (now_us() - t0) / 1000.0);
    }
  }
  h->n = n; h->dim = dim;
  h->sp_topk = -1;
  return h;

fail_idx: cuvsCagraIndexDestroy(h->idx);
fail_ip:  cuvsCagraIndexParamsDestroy(h->ip);
fail_res: cuvsResourcesDestroy(h->res); free(h);
  fprintf(stderr, "[seekdb_cuvs] build failed: %s\n", cuvsGetLastErrorText());
  return NULL;
}

int seekdb_cuvs_search(void* handle, const float* query, long nq, long topk,
                       unsigned int* out_ids, float* out_dist)
{
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  if (!h || !query || !out_ids || nq <= 0 || topk <= 0) return 1;
  int rc = 0;
  long t0 = trace_enabled() ? now_us() : 0;

  pthread_mutex_lock(&h->mu);
  /* Single-row path is the only one seekdb issues (4096 x nq=1); batch calls
   * fall back to the allocation-per-call path so buffer sizing stays simple. */
  if (nq == 1) {
    if (ensure_staging(h, topk) != 0) { rc = 20; goto unlock; }
    if (tune_search_params(h, topk) != 0) { rc = 21; goto unlock; }
    if (cudaMemcpy(h->q_dev, query, sizeof(float) * h->dim, cudaMemcpyDefault) != cudaSuccess) { rc = 22; goto unlock; }
    {
      DLManagedTensor qt, nt, dt; long qs[2] = {1, h->dim}, nsh[2] = {1, topk}, dsh[2] = {1, topk};
      memset(&qt, 0, sizeof(qt));
      qt.dl_tensor.data = h->q_dev; qt.dl_tensor.device.device_type = kDLCUDA; qt.dl_tensor.device.device_id = 0;
      qt.dl_tensor.ndim = 2; qt.dl_tensor.dtype.code = kDLFloat; qt.dl_tensor.dtype.bits = 32; qt.dl_tensor.dtype.lanes = 1;
      qt.dl_tensor.shape = qs; qt.dl_tensor.strides = 0; qt.dl_tensor.byte_offset = 0;
      nt = qt; nt.dl_tensor.data = h->ids_dev; nt.dl_tensor.dtype.code = kDLUInt; nt.dl_tensor.shape = nsh;
      dt = qt; dt.dl_tensor.data = h->dist_dev; dt.dl_tensor.dtype.code = kDLFloat; dt.dl_tensor.shape = dsh;
      cuvsFilter f; f.type = NO_FILTER; f.addr = (uintptr_t)0;
      if (cuvsCagraSearch(h->res, h->sp, h->idx, &qt, &nt, &dt, f) != CUVS_SUCCESS) { rc = 23; goto unlock; }
      if (cudaMemcpy(out_ids, h->ids_dev, sizeof(unsigned) * topk, cudaMemcpyDefault) != cudaSuccess) { rc = 24; goto unlock; }
      if (out_dist) cudaMemcpy(out_dist, h->dist_dev, sizeof(float) * topk, cudaMemcpyDefault);
    }
  } else {
    float* qd = 0; unsigned* nd = 0; float* dd = 0;
    if (cuvsRMMAlloc(h->res, (void**)&qd, sizeof(float) * nq * h->dim) != CUVS_SUCCESS) { rc = 10; goto unlock; }
    if (cuvsRMMAlloc(h->res, (void**)&nd, sizeof(unsigned) * nq * topk) != CUVS_SUCCESS) { rc = 11; goto unlock; }
    if (cuvsRMMAlloc(h->res, (void**)&dd, sizeof(float) * nq * topk) != CUVS_SUCCESS) { rc = 12; goto unlock; }
    if (cudaMemcpy(qd, query, sizeof(float) * nq * h->dim, cudaMemcpyDefault) != cudaSuccess) { rc = 13; goto unlock; }
    if (tune_search_params(h, topk) != 0) { rc = 14; goto unlock; }
    {
      DLManagedTensor qt, nt, dt; long qs[2] = {nq, h->dim}, nsh[2] = {nq, topk}, dsh[2] = {nq, topk};
      memset(&qt, 0, sizeof(qt));
      qt.dl_tensor.data = qd; qt.dl_tensor.device.device_type = kDLCUDA; qt.dl_tensor.device.device_id = 0;
      qt.dl_tensor.ndim = 2; qt.dl_tensor.dtype.code = kDLFloat; qt.dl_tensor.dtype.bits = 32; qt.dl_tensor.dtype.lanes = 1;
      qt.dl_tensor.shape = qs; qt.dl_tensor.strides = 0; qt.dl_tensor.byte_offset = 0;
      nt = qt; nt.dl_tensor.data = nd; nt.dl_tensor.dtype.code = kDLUInt; nt.dl_tensor.shape = nsh;
      dt = qt; dt.dl_tensor.data = dd; dt.dl_tensor.dtype.code = kDLFloat; dt.dl_tensor.shape = dsh;
      cuvsFilter f; f.type = NO_FILTER; f.addr = (uintptr_t)0;
      if (cuvsCagraSearch(h->res, h->sp, h->idx, &qt, &nt, &dt, f) != CUVS_SUCCESS) { rc = 15; goto unlock; }
      if (cudaMemcpy(out_ids, nd, sizeof(unsigned) * nq * topk, cudaMemcpyDefault) != cudaSuccess) { rc = 16; goto unlock; }
      if (out_dist) cudaMemcpy(out_dist, dd, sizeof(float) * nq * topk, cudaMemcpyDefault);
    }
  }
unlock:
  pthread_mutex_unlock(&h->mu);
  if (rc) fprintf(stderr, "[seekdb_cuvs] search rc=%d: %s\n", rc, cuvsGetLastErrorText());
  if (trace_enabled() && !rc) {
    fprintf(stderr, "[seekdb_cuvs] search nq=%ld topk=%ld itopk=%zu width=%zu took %.3f ms\n",
            nq, topk, h->sp ? h->sp->itopk_size : 0, h->sp ? h->sp->search_width : 0,
            (now_us() - t0) / 1000.0);
  }
  return rc;
}

void seekdb_cuvs_free(void* handle)
{
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  if (!h) return;
  if (h->sp) cuvsCagraSearchParamsDestroy(h->sp);
  if (h->dist_dev) cuvsRMMFree(h->res, h->dist_dev, sizeof(float) * h->dist_cap);
  if (h->ids_dev) cuvsRMMFree(h->res, h->ids_dev, sizeof(unsigned) * h->ids_cap);
  if (h->q_dev) cuvsRMMFree(h->res, h->q_dev, sizeof(float) * h->q_cap);
  cuvsCagraIndexDestroy(h->idx);
  cuvsCagraIndexParamsDestroy(h->ip);
  cuvsResourcesDestroy(h->res);
  pthread_mutex_destroy(&h->mu);
  free(h);
}

int seekdb_cuvs_serialize(void* handle, const char* filename)
{
  sk_cuvs_index* index = (sk_cuvs_index*)handle;
  if (!index || !filename) return 1;
  long t0 = trace_enabled() ? now_us() : 0;
  if (cuvsCagraSerialize(index->res, filename, index->idx, true) != CUVS_SUCCESS) {
    fprintf(stderr, "[seekdb_cuvs] serialize failed: %s\n", cuvsGetLastErrorText());
    return 1;
  }
  if (trace_enabled()) {
    struct stat st; memset(&st, 0, sizeof(st));
    if (stat(filename, &st) != 0) st.st_size = 0;
    fprintf(stderr, "[seekdb_cuvs] serialize %s bytes=%lld took %.1f ms\n",
            filename, (long long)st.st_size, (now_us() - t0) / 1000.0);
  }
  return 0;
}

void* seekdb_cuvs_deserialize(const char* filename, long n, long dim)
{
  if (!filename || n <= 0 || dim <= 0) return NULL;
  sk_cuvs_index* index = (sk_cuvs_index*)calloc(1, sizeof(*index));
  if (!index) return NULL;
  pthread_mutex_init(&index->mu, NULL);
  if (cuvsResourcesCreate(&index->res) != CUVS_SUCCESS) goto fail;
  if (cuvsCagraIndexParamsCreate(&index->ip) != CUVS_SUCCESS) goto fail_res;
  load_build_params(index->ip);
  if (cuvsCagraIndexCreate(&index->idx) != CUVS_SUCCESS) goto fail_ip;
  if (cuvsCagraDeserialize(index->res, filename, index->idx) != CUVS_SUCCESS) goto fail_idx;
  {
    int64_t actual_n = 0;
    int64_t actual_dim = 0;
    if (cuvsCagraIndexGetSize(index->idx, &actual_n) != CUVS_SUCCESS ||
        cuvsCagraIndexGetDims(index->idx, &actual_dim) != CUVS_SUCCESS ||
        actual_n != n || actual_dim != dim) {
      fprintf(stderr, "[seekdb_cuvs] snapshot shape mismatch: %ld x %ld, expected %ld x %ld\n",
              (long)actual_n, (long)actual_dim, n, dim);
      goto fail_idx;
    }
  }
  index->n = n;
  index->dim = dim;
  index->sp_topk = -1;
  return index;
fail_idx:
  cuvsCagraIndexDestroy(index->idx);
fail_ip:
  cuvsCagraIndexParamsDestroy(index->ip);
fail_res:
  cuvsResourcesDestroy(index->res);
fail:
  fprintf(stderr, "[seekdb_cuvs] deserialize failed: %s\n", cuvsGetLastErrorText());
  free(index);
  return NULL;
}

int seekdb_cuvs_cagra_knn(const float* base, long n, long dim,
                          const float* query, long nq, long topk,
                          unsigned int* out_ids)
{
  void* h = seekdb_cuvs_build(base, n, dim);
  if (!h) return 10;
  int rc = seekdb_cuvs_search(h, query, nq, topk, out_ids, NULL);
  seekdb_cuvs_free(h);
  return rc;
}
