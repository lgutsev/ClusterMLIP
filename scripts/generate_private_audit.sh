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

mkdir -p "$output_dir"

if command -v cluster-mlip >/dev/null 2>&1; then
  cluster_mlip=(cluster-mlip)
else
  cluster_mlip=(python -m cluster_mlip.cli)
fi

"${cluster_mlip[@]}" analyze "$archive" -o "$output_dir/full"
"${cluster_mlip[@]}" analyze "$archive" \
  -o "$output_dir/feno_pilot" \
  --elements Fe,N,O \
  --require-elements Fe \
  --max-atoms 20

{
  echo "generated_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "source=$archive"
  if command -v sha256sum >/dev/null 2>&1 && [[ -f "$archive" ]]; then
    sha256sum "$archive"
  fi
} > "$output_dir/provenance.txt"

echo "Private audit written to: $output_dir"
echo "Do not copy this directory into the public repository."
