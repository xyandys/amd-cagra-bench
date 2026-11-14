#!/usr/bin/env bash
# How much does the search beam width cost, and does a smaller beam keep the
# recall we promise?  SIFT has ground truth here, so this is the honest test.
set -u
cd /workspace/w

run() {
  local itopk="$1"
  echo "== ITOPK=$itopk =="
  SEEKDB_CUVS_ITOPK="$itopk" LD_LIBRARY_PATH=/workspace/w/pre \
    ./d128/selftest sift_base.fvecs 250000 128 1000 100 sift_query.fvecs 2>&1 \
    | grep -E 'SEARCH10 |SEARCH100 |RECALL@'
}

# 512 is what the bridge floors to today.
for k in 512 384 256 192 128; do run "$k"; done
