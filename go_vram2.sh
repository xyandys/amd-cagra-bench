#!/bin/sh
# go_vram2.sh -- fixed VRAM peak measurement (rocm-smi writes to stderr too)
cd /workspace
mon() {
  rm -f vram2.txt
  ( while true; do rocm-smi --showmeminfo vram --csv 2>&1 | grep -m1 '^card0' | cut -d, -f3 | tr -d ' ' >> vram2.txt; sleep 0.2; done ) &
  echo $!
}
peak() { sort -n vram2.txt | tail -1; }
n() { wc -l < vram2.txt; }

echo "=== hidden-proxy 256d KDEG64 CB32K ==="
M=$(mon)
BF_SIFT=/workspace/sift256.fvecs BF_SELFTEST=1 ./v_r1c32 250000 256 64 500 10 512 2>&1 | tail -8
kill $M; sleep 1
echo "VRAM_PEAK_HIDDEN=$(peak) SAMPLES=$(n)"

echo "=== SIFT 128d KDEG64 CB32K ==="
M=$(mon)
BF_SIFT=/workspace/sift_base.fvecs BF_SELFTEST=1 ./v_r1c32 250000 128 64 500 10 512 2>&1 | tail -8
kill $M; sleep 1
echo "VRAM_PEAK_SIFT=$(peak) SAMPLES=$(n)"

echo "=== baseline candidates on instance ==="
ls -d /workspace/v5full /workspace/seekdb* /workspace/bench* 2>&1
ls /workspace/*.c /workspace/*.cpp 2>/dev/null
echo "VRAM_DONE"
