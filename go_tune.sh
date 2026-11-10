#!/bin/sh
set -x
cd /workspace
U=https://api.github.com/repos/xyandys/amd-cagra-bench/contents
curl -sL -H "Accept: application/vnd.github.raw" $U/bfknn3.cu -o bfknn3.cu
wc -c bfknn3.cu
CC="/opt/rocm/bin/hipcc -O2 -fopenmp -D__HIP_PLATFORM_AMD__ -I/opt/hipvs/include -I/opt/rocm/include bfknn3.cu -L/opt/hipvs/lib -lcuvs_c -Wl,-rpath,/opt/hipvs/lib:/opt/rocm/lib"
$CC -o b_r4c16 || exit 1
$CC -DRPT=2 -o b_r2c16 || exit 1
$CC -DRPT=2 -DCT=32 -o b_r2c32 || exit 1
$CC -DRPT=1 -DCT=32 -o b_r1c32 || exit 1
$CC -DRPT=1 -DCT=64 -o b_r1c64 || exit 1
$CC -DKDEG=128 -o b_d128 || exit 1
for b in b_r4c16 b_r2c16 b_r2c32 b_r1c32 b_r1c64; do
  echo "=== $b dim128 ==="
  BF_SKIP_RECALL=1 ./$b 250000 128 64 200 10 512
  echo "=== $b dim256 ==="
  BF_SKIP_RECALL=1 ./$b 250000 256 64 200 10 512
done
echo "=== SIFT download ==="
if [ ! -f sift_base.fvecs ]; then
  curl -sL -o sift_base.fvecs https://hf-mirror.com/datasets/qbo-odp/sift1m/resolve/main/sift_base.fvecs || exit 1
fi
ls -la sift_base.fvecs
echo "=== SIFT recall KDEG64 r4c16 ==="
BF_SIFT=/workspace/sift_base.fvecs BF_SELFTEST=1 BF_QPS=1 ./b_r4c16 250000 128 64 500 10 512
echo "=== SIFT recall KDEG128 ==="
BF_SIFT=/workspace/sift_base.fvecs BF_QPS=1 ./b_d128 250000 128 64 500 10 512
