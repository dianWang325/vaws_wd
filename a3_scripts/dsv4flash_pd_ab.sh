#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
: "${PD_EXPERIMENT_ROOT:?Set PD_EXPERIMENT_ROOT to the experiment input/output directory}"
export PD_EXPERIMENT_ROOT="$(realpath "$PD_EXPERIMENT_ROOT")"
export PYTHONDONTWRITEBYTECODE=1
for file in formal.csv warm16.csv formal_workload.jsonl source_snapshots/suffix.A.py source_snapshots/suffix.B.py toolbox/config.py toolbox/aisbench_test.py aisbench_runtime/ais_bench/__init__.py; do
    test -s "$PD_EXPERIMENT_ROOT/$file" || { printf 'Missing input: %s\n' "$file" >&2; exit 1; }
done
run_dir="$PD_EXPERIMENT_ROOT/pd_pcp"
for variant in A B; do
    mkdir -p "$run_dir"
    mkdir "$run_dir/formal_$variant"
    PD_VARIANT="$variant" unshare -m bash ./dsv4flash_pd_pcp.sh > "$run_dir/formal_$variant/controller.log" 2>&1
done
python ./dsv4flash_pd_audit.py "$run_dir" > "$run_dir/analysis.log" 2>&1
