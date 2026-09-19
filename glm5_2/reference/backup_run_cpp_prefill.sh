#!/usr/bin/env bash
# Usage: API_URL=http://HOST:9000/v1/completions bash backup_run_cpp_prefill.sh fixed|variable OUTPUT_DIR
# Start a fresh healthy CPP service before each run. This script sends 1M warmups first.
set -euo pipefail

stage="${1:?expected fixed or variable}"
output_dir="${2:?expected output directory}"
api_url="${API_URL:-http://127.0.0.1:9000/v1/completions}"
tokenizer="${TOKENIZER:-/mnt/weight/GLM-5.2-W4A8C8-0713-MTP}"
model="${MODEL:-glm-52}"
base="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

case "$stage" in
  fixed) dataset="$base/data/prefill_fixed.jsonl"; input_tokens=65536; name=fix ;;
  variable) dataset="$base/data/prefill_variable.jsonl"; input_tokens=81920; name=variable ;;
  *) echo "stage must be fixed or variable" >&2; exit 2 ;;
esac

mkdir -p "$output_dir"
python "$base/scripts/backup_bench_prefill_glm52.py" \
  --warmup-data "$base/data/prefill_warmup_1m.jsonl" \
  --fixed-data "$dataset" \
  --tokenizer "$tokenizer" \
  --output-dir "$output_dir/warmup" \
  --api-url "$api_url" \
  --model "$model" \
  --input-tokens 1023999 \
  --warmup-count 5 \
  --warmup-output-tokens 1 \
  --warmup-only \
  --timeout-seconds 3600

python - "$output_dir/warmup/warmup_summary.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1], encoding="utf-8"))
if s["success"] != 5 or s["failed"] != 0:
    raise SystemExit("five successful 1M warmups required before measurement")
PY

python "$base/scripts/backup_bench_prefill_glm52.py" \
  --warmup-data "$base/data/prefill_warmup_1m.jsonl" \
  --fixed-data "$dataset" \
  --tokenizer "$tokenizer" \
  --output-dir "$output_dir/$name" \
  --api-url "$api_url" \
  --model "$model" \
  --input-tokens "$input_tokens" \
  --fixed-count 24 \
  --fixed-output-tokens 1 \
  --fixed-concurrency 1 \
  --fixed-stage-name "$name" \
  --fixed-only \
  --timeout-seconds 3600
