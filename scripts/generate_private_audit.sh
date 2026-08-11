#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 || $# > 2 )); then
  echo "Usage: $0 WAREHOUSE.zip [OUTPUT_DIR]" >&2
  exit 2
fi

archive=$1
output_dir=${2:-private_audits/$(basename "${archive%.*}")}

if [[ ! -f "$archive" && ! -d "$archive" ]]; then
  echo "Warehouse not found: $archive" >&2
  exit 2
fi

if command -v cluster-mlip >/dev/null 2>&1; then
  cluster_mlip=(cluster-mlip)
else
  cluster_mlip=(python -m cluster_mlip.cli)
fi

"${cluster_mlip[@]}" audit "$archive" \
  -o "$output_dir" \
  --elements Fe,N,O \
  --require-elements Fe \
  --max-atoms 20
