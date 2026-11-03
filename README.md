# amd-cagra-bench

hipVS CAGRA 基准与调优工具（AMD 赛道 · seekdb GPU 向量索引）

在 AMD Radeon PRO W7900（gfx1100 / RDNA3，ROCm 7.2.4，hipVS）上，对 seekdb 的
CAGRA 建图与查询做端到端测量，并给出两个参数级优化。

## 结果（SIFT250K，dim=128，L2，top-10 / top-100）

| 指标 | 基线 | 本仓库默认 | 变化 |
|---|---|---|---|
| 建图 CREATE INDEX | 41.2 s | **13.1 s** | **3.14x** |
| 查询 P50（top-10） | 2.75 ms | **1.27 ms** | **2.17x** |
| 查询 P50（top-100） | 2.66 ms | 2.44 ms | 1.09x |
| Recall@10 | 1.0000 | **1.0000** | 持平 |
| Recall@100 | 0.9991 | **0.9977** | −0.0014（门槛 0.005） |

门槛全部通过：建图 ≥3x、查询 ≥2x、recall 降幅 ≤0.005。

## 改了什么

`bridge_opt.c` 两个默认值（均可用环境变量覆盖）：

| 参数 | 基线 | 本仓库 | 环境变量 |
|---|---|---|---|
| NN-descent 迭代次数 | 100 | **15** | `SEEKDB_CUVS_NITER` |
| 搜索 `itopk` 下限 | 512 | **256** | `SEEKDB_CUVS_ITOPK` |

建图耗时随 `niter` 近似线性（约 0.30 s/轮 + 9 s 固定开销），而 250K 规模下
NN-descent 在 15 轮左右已收敛——后 85 轮对 recall 无贡献。`itopk` 直接决定
搜索延迟：512→2.75 ms、256→1.25 ms、128→0.70 ms；128 时 recall@10 掉 0.0148
超出门槛，256 则无损。

## 顺带修复的竞态

`cuvsCagraSearch` 在 cuVS 自己的 stream 上异步执行，旧代码随后用
`cudaMemcpy`（默认流）取回 `out_ids/out_dist`，缺少设备同步 → 大 n 下结果
不确定（自查询 recall 只有 0.21～0.73）。修复：取回前 `hipDeviceSynchronize()`。

## 复现

```bash
# 依赖：ROCm (/opt/rocm)、hipVS (/opt/hipvs)、g++、fvecs 数据
export ROCM_ROOT=/opt/rocm HIPVS_ROOT=/opt/hipvs

hipcc -shared -fPIC -O2 -D__HIP_PLATFORM_AMD__ \
      -I$ROCM_ROOT/include -I$HIPVS_ROOT/include -I. \
      bridge_opt.c -o libseekdb_cuvs_bridge.so \
      -L$HIPVS_ROOT/lib -L$ROCM_ROOT/lib -lcuvs -lcuvs_c -Wl,-rpath,$HIPVS_ROOT/lib

g++ -O2 -fopenmp -I . -o siftbench siftbench.cpp ./libseekdb_cuvs_bridge.so -Wl,-rpath,$PWD

# SIFT 数据（sift_base.fvecs 516 MB，sift_query.fvecs 取 offset 990000 的 200 条）
./siftbench sift_base.fvecs query200.fvecs 250000 128 10
./siftbench sift_base.fvecs query200.fvecs 250000 128 100

# 等价于关掉优化的对照
SEEKDB_CUVS_NITER=100 SEEKDB_CUVS_ITOPK=512 ./siftbench sift_base.fvecs query200.fvecs 250000 128 10
```

`bench.cpp` 是无数据的合成版（随机高斯），用于快速冒烟；`siftbench.cpp` 读 fvecs，
并用 CPU 暴力 L2 自算 ground truth，因此不需要额外的 truth 文件。

## 数据来源

SIFT1M（IRISA TexMex corpus）经 HuggingFace 镜像：
`https://hf-mirror.com/datasets/qbo-odp/sift1m/resolve/main/sift_base.fvecs`

## 完整报告

见 [docs/REPORT.md](docs/REPORT.md)。
