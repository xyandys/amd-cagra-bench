#!/bin/sh
# go_basevram.sh -- baseline (original V5 NN-descent bridge) VRAM peak at 250K x 128 / 256
cd /workspace
for f in bridge_baseline.c seekdb_cuvs_bridge.h bench.cpp; do
  [ -f "$f" ] || curl -sL -H "Accept: application/vnd.github.raw" \
    "https://api.github.com/repos/xyandys/amd-cagra-bench/contents/$f" -o "$f"
done
HIPVS=/opt/hipvs
ROCM=/opt/rocm
if [ ! -f ./libbridge_base.so ]; then
  $ROCM/bin/hipcc -shared -fPIC -O2 -D__HIP_PLATFORM_AMD__ \
    -I$HIPVS/include -I$ROCM/include bridge_baseline.c \
    -L$HIPVS/lib -Wl,-soname,libbridge_base.so -Wl,-rpath,"$HIPVS/lib:$ROCM/lib" \
    -lcuvs_c -o libbridge_base.so 2>&1 | tail -5
fi
ls -la libbridge_base.so
[ -x ./bench_base ] || hipcc -O2 -fopenmp -I. bench.cpp ./libbridge_base.so \
  -Wl,-rpath,'$ORIGIN' -o bench_base 2>&1 | tail -5
ls -la bench_base

mon() {
  rm -f vram3.txt
  ( while true; do rocm-smi --showmeminfo vram --csv 2>&1 | grep -m1 '^card0' | cut -d, -f3 | tr -d ' ' >> vram3.txt; sleep 0.2; done ) &
  echo $!
}
peak() { sort -n vram3.txt | tail -1; }

echo "=== baseline 128d ==="
M=$(mon)
./bench_base 250000 128 4096 10 64 2>&1 | tail -8
kill $M; sleep 1
echo "VRAM_PEAK_BASE128=$(peak)"

echo "=== baseline 256d ==="
M=$(mon)
./bench_base 250000 256 4096 10 64 2>&1 | tail -8
kill $M; sleep 1
echo "VRAM_PEAK_BASE256=$(peak)"
echo "BASEVRAM_DONE"
