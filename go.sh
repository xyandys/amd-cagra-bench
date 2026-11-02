#!/usr/bin/env bash
# One-shot AMD hipVS CAGRA bench: frozen-baseline bridge vs optimized bridge.
# Run from the cloned repo:  bash go.sh
# Logs to $HOME/amdbench/run.log
set -u

D=$(cd "$(dirname "$0")" && pwd)
W=${WORK:-$HOME/amdbench}
mkdir -p "$W"
cd "$W" || exit 1
exec > >(tee "$W/run.log") 2>&1

say() { printf '\n=== %s ===\n' "$*"; }

say "0 env"
uname -m
nproc
HIPCC=${HIPCC:-/opt/rocm/bin/hipcc}
HIPVS=${HIPVS_ROOT:-/opt/hipvs}
ls -l "$HIPCC" || echo "NO_HIPCC"
ls "$HIPVS/lib/libcuvs_c.so" || echo "NO_CUVS"
export LD_LIBRARY_PATH="$HIPVS/lib:/opt/rocm/lib:${LD_LIBRARY_PATH:-}"

mkdir -p base opt
cp -f "$D"/bridge_base.c "$D"/bridge_opt.c "$D"/seekdb_cuvs_bridge.h "$D"/bench.cpp . || { echo "MISSING_SOURCES"; exit 1; }
ls -l bridge_base.c bridge_opt.c seekdb_cuvs_bridge.h bench.cpp

build_so() {
  local src="$1"
  local dir="$2"
  ( cd "$dir" && "$HIPCC" -shared -fPIC -O2 -D__HIP_PLATFORM_AMD__ \
      -I "$W" -I "$HIPVS/include" -I /opt/rocm/include \
      "$W/$src" -L "$HIPVS/lib" -Wl,-soname,libseekdb_cuvs_bridge.so \
      -Wl,-rpath,"$HIPVS/lib:/opt/rocm/lib" -lcuvs_c -o libseekdb_cuvs_bridge.so )
}

build_bench() {
  local dir="$1"
  ( cd "$dir" && "$HIPCC" -O2 -fopenmp -D__HIP_PLATFORM_AMD__ -I "$W" "$W/bench.cpp" \
      ./libseekdb_cuvs_bridge.so -Wl,-rpath,"$dir" -o bench )
  if [ ! -x "$dir/bench" ]; then
    ( cd "$dir" && "$HIPCC" -O2 -D__HIP_PLATFORM_AMD__ -I "$W" "$W/bench.cpp" \
        ./libseekdb_cuvs_bridge.so -Wl,-rpath,"$dir" -o bench )
  fi
  [ -x "$dir/bench" ]
}

say "1 build baseline"
build_so bridge_base.c base || { echo "BASELINE_BUILD_FAIL"; exit 1; }
build_bench base || { echo "DRIVER_BUILD_FAIL"; exit 1; }
ls -l base

say "2 baseline run"
( cd base && SEEKDB_CUVS_TRACE=1 ./bench 250000 128 4096 10 256 ) | tee base/top10.txt
( cd base && SEEKDB_CUVS_TRACE=1 ./bench 250000 128 4096 100 128 ) | tee base/top100.txt

say "3 build optimized"
build_so bridge_opt.c opt || { echo "OPT_BUILD_FAIL"; exit 1; }
build_bench opt || { echo "DRIVER_BUILD_FAIL2"; exit 1; }

say "4 optimized run"
( cd opt && SEEKDB_CUVS_TRACE=1 ./bench 250000 128 4096 10 256 ) | tee opt/top10.txt
( cd opt && SEEKDB_CUVS_TRACE=1 ./bench 250000 128 4096 100 128 ) | tee opt/top100.txt

say "5 SUMMARY"
for d in base opt; do
  for f in top10 top100; do
    printf '%-5s %-7s ' "$d" "$f"
    grep -hE '^(BUILD_MS|QPS|RECALL)' "$d/$f.txt" | tr '\n' ' '
    echo
  done
done
echo
echo "log: $W/run.log"
