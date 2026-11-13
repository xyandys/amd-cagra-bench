#!/usr/bin/env bash
# fetch the bridge, build it, and check that the strided pre-pass keeps the
# graph identical (recall must not move) while cutting the build time.
set -euo pipefail
cd /workspace/w

T=$(cat /workspace/.ght)
curl -sS -H "Authorization: bearer $T" \
  "https://api.github.com/repos/xyandys/amd-cagra-bench/contents/seekdb_cuvs_bridge.cpp?ref=main" \
  -o /tmp/r.json
python3 - <<'PY'
import json, base64
d = json.load(open('/tmp/r.json'))
open('/workspace/w/seekdb_cuvs_bridge.cpp', 'wb').write(base64.b64decode(d['content']))
PY
sed -i 's/\r$//' seekdb_cuvs_bridge.cpp

mkdir -p pre
/opt/rocm/bin/hipcc -shared -fPIC -O2 -D__HIP_PLATFORM_AMD__ --offload-arch=gfx1100 \
  -x hip seekdb_cuvs_bridge.cpp -I/opt/hipvs/include -I/opt/rocm/include \
  -L/opt/hipvs/lib -Wl,-soname,libseekdb_cuvs_bridge.so \
  -Wl,-rpath,/opt/hipvs/lib:/opt/rocm/lib -lcuvs_c \
  -o pre/libseekdb_cuvs_bridge.so 2>&1 | tail -2

echo "== SIFT =="
LD_LIBRARY_PATH=/workspace/w/pre ./d128/selftest sift_base.fvecs 250000 128 1000 100 sift_query.fvecs 2>&1 \
  | grep -E 'BUILD_MS|RECALL@|QPS'
echo "== 768 =="
LD_LIBRARY_PATH=/workspace/w/pre ./d128/selftest g768_base.fvecs 250000 768 1000 100 g768_query.fvecs 2>&1 \
  | grep -E 'BUILD_MS|RECALL@'
