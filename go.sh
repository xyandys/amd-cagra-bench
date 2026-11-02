#!/usr/bin/env bash
# One-shot AMD hipVS CAGRA bench: frozen baseline bridge vs optimized bridge.
# Usage (in the Jupyter terminal):
#   curl -sL https://raw.githubusercontent.com/xyandys/amd-cagra-bench/main/go.sh | bash
# Everything is logged to ~/amdbench/run.log
set -u

W=${WORK:-$HOME/amdbench}
PIN=634888d9b037153fa143d2c913ba4b25503e1d62
REPO=xyandys/amd-cagra-bench
mkdir -p "$W"
cd "$W" || exit 1
exec > >(tee "$W/run.log") 2>&1

say() { printf '\n=== %s ===\n' "$*"; }

say "0 env"
uname -m; nproc
HIPCC=${HIPCC:-/opt/rocm/bin/hipcc}
HIPVS=${HIPVS_ROOT:-/opt/hipvs}
ls -l "$HIPCC" || echo "NO_HIPCC"
ls "$HIPVS/lib/libcuvs_c.so" || echo "NO_CUVS"
export LD_LIBRARY_PATH="$HIPVS/lib:/opt/rocm/lib:${LD_LIBRARY_PATH:-}"

say "1 network probe"
PREFIX=""
for p in "https://raw.githubusercontent.com" \
         "https://ghproxy.net/https://raw.githubusercontent.com" \
         "https://gh-proxy.com/https://raw.githubusercontent.com" \
         "https://raw.gitmirror.com"; do
  if curl -sfL -m 10 -o /dev/null "$p/$REPO/main/go.sh"; then
    PREFIX=$p; echo "using $p"; break
  fi
  echo "unreachable: $p"
done
[ -n "$PREFIX" ] || { echo "NO_NETWORK_TO_GITHUB"; exit 1; }

get() { # get <url-path-after-host> <outfile>
  curl -sfL -m 60 "$PREFIX/$1" -o "$2" || { echo "FETCH_FAIL $1"; return 1; }
}

say "2 fetch sources"
mkdir -p src base opt
get "$REPO/main/bench.cpp"            src/bench.cpp   || exit 1
get "$REPO/main/bridge_opt.c"         src/bridge_opt.c || exit 1
get "zihaomu/seekdb/$PIN/tools/hipvs_bridge/seekdb_cuvs_bridge.c" src/bridge_base.c || exit 1
get "zihaomu/seekdb/$PIN/tools/hipvs_bridge/seekdb_cuvs_bridge.h" src/seekdb_cuvs_bridge.h || exit 1
ls -l src

build_so() { # build_so <src.c> <outdir>
  ( cd "$2" && "$HIPCC" -shared -fPIC -O2 -D__HIP_PLATFORM_AMD__ \
      -I "$W/src" -I "$HIPVS/include" -I /opt/rocm/include \
      "$1" -L "$HIPVS/lib" -Wl,-soname,libseekdb_cuvs_bridge.so \
      -Wl,-rpath,"$HIPVS/lib:/opt/rocm/lib" -lcuvs_c -o libseekdb_cuvs_bridge.so )
}

build_bench() { # build_bench <dir>
  ( cd "$2" && OMPFLAG=-fopenmp && \
    { "$HIPCC" -O2 $OMPFLAG -D__HIP_PLATFORM_AMD__ -I "$W/src" "$W/src/bench.cpp" \
        ./libseekdb_cuvs_bridge.so -Wl,-rpath,"$2" -o bench || \
      { echo "retry without openmp"; "$HIPCC" -O2 -D__HIP_PLATFORM_AMD__ -I "$W/src" "$W/src/bench.cpp" \
        ./libseekdb_cuvs_bridge.so -Wl,-rpath,"$2" -o bench; }; } )
}

say "3 build baseline"
build_so "$W/src/bridge_base.c" "$W/base" || { echo "BASELINE_BUILD_FAIL"; exit 1; }
build_bench x "$W/base" || { echo "DRIVER_BUILD_FAIL"; exit 1; }
ls -l "$W/base"

say "4 baseline run"
( cd "$W/base" && SEEKDB_CUVS_TRACE=1 ./bench 250000 128 4096 10 256 ) | tee "$W/base/top10.txt"
( cd "$W/base" && SEEKDB_CUVS_TRACE=1 ./bench 250000 128 4096 100 128 ) | tee "$W/base/top100.txt"

say "5 build optimized"
build_so "$W/src/bridge_opt.c" "$W/opt" || { echo "OPT_BUILD_FAIL"; exit 1; }
build_bench x "$W/opt" || exit 1

say "6 optimized run"
( cd "$W/opt" && SEEKDB_CUVS_TRACE=1 ./bench 250000 128 4096 10 256 ) | tee "$W/opt/top10.txt"
( cd "$W/opt" && SEEKDB_CUVS_TRACE=1 ./bench 250000 128 4096 100 128 ) | tee "$W/opt/top100.txt"

say "7 SUMMARY"
for d in base opt; do
  for f in top10 top100; do
    printf '%-5s %-7s ' "$d" "$f"
    grep -E '^(BUILD_MS|QPS|RECALL)' "$W/$d/$f.txt" | tr '\n' ' '
    echo
  done
done
echo
echo "full log: $W/run.log"
