#!/usr/bin/env bash
# Configure aisbench_auto_tools_prefix/config.py as described in its README first.
# Usage: bash scripts/run_aisbench.sh STAGE
set -euo pipefail

stage="${1:?expected warmup, prefill_fix, prefill_variable, fix, or variable}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tool_dir="${AISBENCH_TOOL_DIR:-$script_dir/../aisbench_auto_tools_prefix}"
case "$stage" in
  warmup)
    input_len="$((${MODEL_MAX_LEN:-204800} - 1))"
    output_len=1
    data_num=5
    concurrency=1
    ;;
  prefill_fix)
    input_len=65536
    output_len=1
    data_num=24
    concurrency="${CONCURRENCY:-2}"
    ;;
  prefill_variable|fix|variable)
    input_len=65536
    data_num=24
    concurrency="${CONCURRENCY:-1}"
    case "$stage" in
      prefill_variable) output_len=1 ;;
      fix|variable) output_len=2560 ;;
    esac
    ;;
  *) echo "invalid stage: $stage" >&2; exit 2 ;;
esac

[[ -d "$tool_dir" ]] || { echo "toolbox directory not found: $tool_dir" >&2; exit 1; }
cd "$tool_dir"
[[ -f aisbench_test.py && -f config.py && -f GSM8K.jsonl ]] || {
  echo "toolbox directory must contain aisbench_test.py, config.py, and GSM8K.jsonl: $tool_dir" >&2
  exit 1
}

args=(--input_len "$input_len" --output_len "$output_len" --data_num "$data_num"
  --concurrency "$concurrency" --request_rate 0 --dataset_type normal
  --api_type completions --test_type stream)

# The toolbox checks a fixed-length filename even for variable-length datasets.
# Select its variable dataset explicitly; create it with its own generator only if absent.
if [[ "$stage" == prefill_variable || "$stage" == variable ]]; then
  dataset="$(python3 - <<'PY'
from pathlib import Path
from config import DATASET_PATH, MODEL_PATH
from generate_dataset import create_multi_prefix_dataset

root = Path(DATASET_PATH)
root.mkdir(parents=True, exist_ok=True)
name = Path(MODEL_PATH.rstrip('/')).name
path = root / f'GSM8K-G65536_16384_C40960_81920-num24-{name}.jsonl'
if not path.is_file():
    _, generated = create_multi_prefix_dataset(
        MODEL_PATH, 65536, 24, str(root), 0, 1, 0, 1, 1,
        length_mean=65536, length_std=16384,
        length_min=40960, length_max=81920)
    if Path(generated) != path or not path.is_file():
        raise SystemExit('toolbox failed to generate the variable dataset')
print(path.resolve())
PY
)"
  args+=(--dataset "$dataset")
fi

python3 aisbench_test.py "${args[@]}"
