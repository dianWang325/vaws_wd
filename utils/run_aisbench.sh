#!/usr/bin/env bash
# Integrated AISBench performance stages for aisbench_auto_tools_prefix.
# Usage: bash /mnt/share/w00985415/scripts/run_aisbench.sh STAGE
set -euo pipefail

stage="${1:?expected warmup, fix-128-c1, fix-256-c1, fix-4k-c16, or var-32k-c4}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tool_dir="${AISBENCH_TOOL_DIR:-$script_dir/../aisbench_auto_tools_prefix}"

case "$stage" in
  warmup)
    input_len="$((${MODEL_MAX_LEN:-200000} - 1))"
    output_len=1
    data_num=5
    concurrency=1
    ;;
  fix-128-c1)
    input_len=131072
    output_len=1
    data_num=5
    concurrency=1
    ;;
  fix-256-c1)
    input_len=262144
    output_len=1
    data_num=5
    concurrency=1
    ;;
  fix-4k-c16)
    input_len=4096
    output_len=1
    data_num=64
    concurrency=16
    ;;
  var-32k-c4)
    input_len=32768
    output_len=1
    data_num="${VAR_DATA_NUM:-64}"
    concurrency=4
    ;;
  *)
    echo "invalid stage: $stage" >&2
    echo "expected warmup, fix-128-c1, fix-256-c1, fix-4k-c16, or var-32k-c4" >&2
    exit 2
    ;;
esac

[[ -d "$tool_dir" ]] || { echo "toolbox directory not found: $tool_dir" >&2; exit 1; }
cd "$tool_dir"
[[ -f aisbench_test.py && -f config.py && -f GSM8K.jsonl ]] || {
  echo "toolbox directory must contain aisbench_test.py, config.py, and GSM8K.jsonl: $tool_dir" >&2
  exit 1
}

args=(
  --input_len "$input_len"
  --output_len "$output_len"
  --data_num "$data_num"
  --concurrency "$concurrency"
  --request_rate 0
  --dataset_type normal
  --test_type stream
  --api_mode completion
)

# Generate the bounded Gaussian dataset once and reuse it on later runs.
if [[ "$stage" == var-32k-c4 ]]; then
  dataset="$(DATA_NUM="$data_num" python3 - <<'PY'
import os
from pathlib import Path

from config import DATASET_PATH, MODEL_PATH
from generate_dataset import create_multi_prefix_dataset

data_num = int(os.environ["DATA_NUM"])
root = Path(DATASET_PATH)
root.mkdir(parents=True, exist_ok=True)
model_name = Path(MODEL_PATH.rstrip("/")).name
path = root / f"GSM8K-G32768_32768_C4096_65536-num{data_num}-{model_name}.jsonl"

if not path.is_file():
    _, generated = create_multi_prefix_dataset(
        MODEL_PATH,
        32768,
        data_num,
        str(root),
        0,
        1,
        0,
        1,
        1,
        length_mean=32768,
        length_std=32768,
        length_min=4096,
        length_max=65536,
    )
    if Path(generated) != path or not path.is_file():
        raise SystemExit(f"toolbox failed to generate the variable dataset: {path}")

print(path.resolve())
PY
)"
  args+=(--dataset "$dataset")
fi

echo "stage=$stage input_len=$input_len output_len=$output_len data_num=$data_num concurrency=$concurrency"
exec python3 aisbench_test.py "${args[@]}"
