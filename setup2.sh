#!/usr/bin/env bash
# one-shot setup on a fresh Radeon Cloud instance
set -euo pipefail
mkdir -p /workspace/w
cd /workspace/w

TOKEN=$(cat /workspace/.ght)
get() {
  local name="$1"
  curl -sS -H "Authorization: bearer $TOKEN" \
    "https://api.github.com/repos/xyandys/amd-cagra-bench/contents/${name}?ref=main" -o /tmp/g.json
  python3 - "$name" <<'PY'
import json, base64, sys
d = json.load(open('/tmp/g.json'))
open(sys.argv[1], 'wb').write(base64.b64decode(d['content']))
PY
  sed -i 's/\r$//' "$name"
  echo "fetched $name"
}

get seekdb_cuvs_bridge.cpp
get seekdb_cuvs_bridge.h
get bridge_selftest.cpp
get go_deg.sh

echo "== data =="
SIFT=https://hf-mirror.com/datasets/qbo-odp/sift1m/resolve/main
[ -s sift_base.fvecs ] || curl -sSL -r 0-129515999 -o sift_base.fvecs "$SIFT/sift_base.fvecs"
[ -s sift_query.fvecs ] || curl -sSL -r 0-515999 -o sift_query.fvecs "$SIFT/sift_query.fvecs"
ls -l sift_base.fvecs sift_query.fvecs

echo "== build =="
mkdir -p d128
/opt/rocm/bin/hipcc -shared -fPIC -O2 -D__HIP_PLATFORM_AMD__ -DSK_DEG=128 \
  --offload-arch=gfx1100 -x hip seekdb_cuvs_bridge.cpp \
  -I/opt/hipvs/include -I/opt/rocm/include -L/opt/hipvs/lib \
  -Wl,-soname,libseekdb_cuvs_bridge.so -Wl,-rpath,/opt/hipvs/lib:/opt/rocm/lib \
  -lcuvs_c -o d128/libseekdb_cuvs_bridge.so 2>&1 | tail -2 || echo BRIDGE_BUILD_FAILED
/opt/rocm/bin/hipcc -O2 -D__HIP_PLATFORM_AMD__ --offload-arch=gfx1100 -x hip bridge_selftest.cpp \
  -I/opt/hipvs/include -I/opt/rocm/include -L/workspace/w/d128 -lseekdb_cuvs_bridge \
  -Wl,-rpath,/workspace/w/d128 -o d128/selftest 2>&1 | tail -2 || echo SELFTEST_BUILD_FAILED
ls -l d128/
echo SETUP_DONE
