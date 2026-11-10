#!/bin/bash
# go_fromargs.sh — §6 step-1 driver.
# Dumps the real cuvsCagraIndexFromArgs signature + distance enum first,
# then builds and runs fromargs_test at small scale.
# Usage: bash go_fromargs.sh [n dim degree nq topk itopk]

set -x

echo '===== cagra.h: FromArgs signature ====='
grep -n -B2 -A10 "cuvsCagraIndexFromArgs" /opt/hipvs/include/cuvs/neighbors/cagra.h

echo '===== cuvsDistanceType enum ====='
grep -rn -B2 -A25 "} cuvsDistanceType" /opt/hipvs/include/cuvs/ 2>/dev/null | head -80

echo '===== symbol presence ====='
nm -D /opt/hipvs/lib/libcuvs_c.so | grep -i "FromArgs\|CagraSearch\|CagraBuild" || true

echo '===== build ====='
/opt/rocm/bin/hipcc -O2 -fopenmp -D__HIP_PLATFORM_AMD__ \
  -I/opt/hipvs/include -I/opt/rocm/include \
  fromargs_test.c \
  -L/opt/hipvs/lib -lcuvs_c \
  -Wl,-rpath,/opt/hipvs/lib:/opt/rocm/lib \
  -o fromargs_test || { echo "BUILD FAILED"; exit 1; }

echo '===== warmup (ROCm kernel cache) ====='
./fromargs_test 2000 16 16 100 10 512 >/dev/null 2>&1 || true

echo '===== run ====='
./fromargs_test "${1:-5000}" "${2:-32}" "${3:-32}" "${4:-500}" "${5:-10}" "${6:-512}"

echo '===== optional: graph on device ====='
GRAPH_ON_DEVICE=1 ./fromargs_test "${1:-5000}" "${2:-32}" "${3:-32}" "${4:-500}" "${5:-10}" "${6:-512}"
