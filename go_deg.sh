#!/bin/bash
# Bootstrap data + sweep SK_DEG for the single-pass exact-graph bridge.
# Reports build time, VRAM peak and recall for 250k x 128 and 250k x 256.
set -Eeuo pipefail
WS=/workspace/w
mkdir -p "$WS"
cd "$WS" || exit 9
T=$(cat /workspace/.ght)

get() { # get <repo-path> <local-name>
  curl -sS -m 180 -H "Authorization: bearer $T" \
    "https://api.github.com/repos/xyandys/amd-cagra-bench/contents/$1?ref=main" \
    -o /tmp/r.json || return 1
  python3 -c "import json,base64,sys;d=json.load(open('/tmp/r.json'));open(sys.argv[2],'wb').write(base64.b64decode(d['content']))" "$1" "$2"
}

get seekdb_cuvs_bridge.cpp seekdb_cuvs_bridge.cpp
get seekdb_cuvs_bridge.h seekdb_cuvs_bridge.h
get bridge_selftest.cpp bridge_selftest.cpp
sed -i 's/\r$//' seekdb_cuvs_bridge.cpp seekdb_cuvs_bridge.h bridge_selftest.cpp

if [ ! -s sift_base.fvecs ]; then
  echo "downloading sift_base.fvecs"
  curl -L --retry 3 -m 1800 -o sift_base.fvecs.part \
    https://hf-mirror.com/datasets/qbo-odp/sift1m/resolve/main/sift_base.fvecs
  mv sift_base.fvecs.part sift_base.fvecs
fi
if [ ! -s sift_query.fvecs ]; then
  echo "downloading sift_query.fvecs"
  curl -L --retry 3 -m 600 -o sift_query.fvecs.part \
    https://hf-mirror.com/datasets/qbo-odp/sift1m/resolve/main/sift_query.fvecs
  mv sift_query.fvecs.part sift_query.fvecs
fi
if [ ! -s sift256.fvecs ]; then
  echo "building 256-dim slice from sift"
  python3 - <<'PY'
import numpy as np
m = 250000
raw = np.fromfile('sift_base.fvecs', dtype=np.int32, count=m * 129).reshape(m, 129)
v = raw[:, 1:].view(np.float32)
w = np.hstack([v, np.roll(v, m // 2, axis=0)])
o = np.empty((m, 257), dtype=np.int32)
o[:, 0] = 256
o[:, 1:] = w.view(np.int32)
o.tofile('sift256.fvecs')
rawq = np.fromfile('sift_query.fvecs', dtype=np.int32, count=1000 * 129).reshape(1000, 129)
q = rawq[:, 1:].view(np.float32)
qw = np.hstack([q, q])
oq = np.empty((1000, 257), dtype=np.int32)
oq[:, 0] = 256
oq[:, 1:] = qw.view(np.int32)
oq.tofile('sift256_query.fvecs')
print('wrote sift256.fvecs / sift256_query.fvecs')
PY
fi

for D in 64 96 128; do
  mkdir -p "d$D"
  if ! /opt/rocm/bin/hipcc -shared -fPIC -O2 -D__HIP_PLATFORM_AMD__ -DSK_DEG=$D \
      --offload-arch=gfx1100 -x hip seekdb_cuvs_bridge.cpp \
      -I/opt/hipvs/include -I/opt/rocm/include -L/opt/hipvs/lib \
      -Wl,-soname,libseekdb_cuvs_bridge.so \
      -Wl,-rpath,/opt/hipvs/lib:/opt/rocm/lib -lcuvs_c \
      -o "d$D/libseekdb_cuvs_bridge.so" > "d$D/build.log" 2>&1; then
    echo "COMPILE_FAIL deg=$D"; tail -4 "d$D/build.log"; continue
  fi
  if ! /opt/rocm/bin/hipcc -O2 -D__HIP_PLATFORM_AMD__ bridge_selftest.cpp -I. \
      -L"d$D" -lseekdb_cuvs_bridge -Wl,-rpath,/workspace/w/d$D \
      -o "d$D/selftest" >> "d$D/build.log" 2>&1; then
    echo "SELFTEST_COMPILE_FAIL deg=$D"; tail -4 "d$D/build.log"; continue
  fi
  echo "=== deg=$D sift 250k x 128 ==="
  LD_LIBRARY_PATH="/workspace/w/d$D" "./d$D/selftest" sift_base.fvecs 250000 128 1000 100 sift_query.fvecs 2>&1 |
    grep -E 'BUILD_MS|RECALL|QPS|DESERIALIZE_MS|IDS_MATCH' || true
  echo "=== deg=$D sift 250k x 256 ==="
  LD_LIBRARY_PATH="/workspace/w/d$D" "./d$D/selftest" sift256.fvecs 250000 256 1000 100 sift256_query.fvecs 2>&1 |
    grep -E 'BUILD_MS|RECALL|QPS|DESERIALIZE_MS|IDS_MATCH' || true
done
echo DONE
