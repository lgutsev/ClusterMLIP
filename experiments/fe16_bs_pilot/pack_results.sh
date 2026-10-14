#!/usr/bin/env bash
# Pack returned logs, status and manifests (no checkpoints) for transfer to the laptop.
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")"
files=()
for c in validation smoke pilot charge_init; do
  [[ -d $c ]] || continue
  (cd "$c" && bash gaussian_batch_status.sh > status.txt 2>&1 || true)
  while IFS= read -r f; do files+=("$f"); done < <(find "$c" -maxdepth 3 \( -name '*.log' -o -name '*.rc' \
      -o -name 'status.txt' -o -name 'spin_jobs.csv' -o -name 'slurm_plan.json' -o -name 'scheduler-*' \
      -o -name 'expected.json' -o -name 'sites.csv' \) -print)
done
tar -czf fe16_bs_tandem_results.tar.gz "${files[@]}"
echo "packed ${#files[@]} files into fe16_bs_tandem_results.tar.gz"
