#!/bin/sh
set -x
cd /workspace
U=https://api.github.com/repos/xyandys/amd-cagra-bench/contents
curl -sL -H "Accept: application/vnd.github.raw" $U/bfknn3.cu -o bfknn3.cu
wc -c bfknn3.cu
CC="/opt/rocm/bin/hipcc -O2 -fopenmp -D__HIP_PLATFORM_AMD__ -I/opt/hipvs/include -I/opt/rocm/include bfknn3.cu -L/opt/hipvs/lib -lcuvs_c -Wl,-rpath,/opt/hipvs/lib:/opt/rocm/lib"
$CC -DRPT=1 -DCT=32 -o v_r1c32 || exit 1
$CC -DRPT=1 -DCT=32 -DKDEG=128 -o v_d128 || exit 1
$CC -DRPT=1 -DCT=32 -DCBAND=65536L -o v_cb64k || exit 1

vrampeak() {
  awk -F',' '{gsub(/ /,"",$3); if($3+0>m) m=$3+0} END{print "VRAM_PEAK_BYTES", m}' /workspace/vram.log
}

vrammon() {
  ( while true; do rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -n1; sleep 0.2; done ) > /workspace/vram.log 2>&1 &
  echo $!
}

echo "=== 1. SIFT r1c32 KDEG64 CB32K (recall+qps+vram) ==="
M=$(vrammon)
BF_SIFT=/workspace/sift_base.fvecs BF_SELFTEST=1 BF_QPS=1 ./v_r1c32 250000 128 64 500 10 512
kill $M
sleep 1
vrampeak

echo "=== 2. SIFT r1c32 KDEG64 CB64K (recall+vram) ==="
M=$(vrammon)
BF_SIFT=/workspace/sift_base.fvecs ./v_cb64k 250000 128 64 500 10 512
kill $M
sleep 1
vrampeak

echo "=== 3. hidden-proxy 256d data (SIFT concat) ==="
if [ ! -f sift256.fvecs ]; then
python3 - <<'EOF'
import struct
with open('/workspace/sift_base.fvecs','rb') as f:
    data = f.read()
d = struct.unpack('<i', data[:4])[0]
n = len(data)//(4+4*d)
import array
a = array.array('f')
a.frombytes(data[4:4+4*d])  # skip first dim header
# simpler: parse rows
rows = []
off = 0
for i in range(n):
    dd = struct.unpack('<i', data[off:off+4])[0]
    rows.append(data[off+4:off+4+4*dd])
    off += 4+4*dd
m = 250000
out = open('/workspace/sift256.fvecs','wb')
for i in range(m):
    out.write(struct.pack('<i', 256))
    out.write(rows[i][:4*128])
    out.write(rows[(i+m//2)%m][:4*128])
out.close()
print('wrote sift256.fvecs', m, 'x', 256)
EOF
fi
ls -la sift256.fvecs

echo "=== 4. hidden-proxy KDEG64 recall ==="
BF_SIFT=/workspace/sift256.fvecs BF_SELFTEST=1 BF_QPS=1 ./v_r1c32 250000 256 64 500 10 512

echo "=== 5. hidden-proxy KDEG128 recall ==="
BF_SIFT=/workspace/sift256.fvecs BF_QPS=1 ./v_d128 250000 256 64 500 10 512
