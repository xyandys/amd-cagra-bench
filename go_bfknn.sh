#!/bin/sh
set -x
cd /workspace
U=https://api.github.com/repos/xyandys/amd-cagra-bench/contents
curl -sL -H "Accept: application/vnd.github.raw" $U/bfknn_test.c -o bfknn_test.c
wc -c bfknn_test.c
/opt/rocm/bin/hipcc -O2 -fopenmp -D__HIP_PLATFORM_AMD__ \
  -I/opt/hipvs/include -I/opt/rocm/include bfknn_test.c \
  -L/opt/hipvs/lib -lcuvs_c -Wl,-rpath,/opt/hipvs/lib:/opt/rocm/lib -o bfknn_test || exit 1
echo === SMALL 20K x 32 ===
BF_SELFTEST=1 ./bfknn_test 20000 32 32 300 10 512
echo === SIFT 250K x 128 ===
./bfknn_test 250000 128 64 500 10 512
echo === HIDDEN 250K x 256 ===
./bfknn_test 250000 256 64 500 10 512
