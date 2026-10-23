#!/usr/bin/env bash
# Warm up the profiling-based chunk predictor with both single-request and
# mixed-batch shapes. Restart the serving process before running this script;
# online calibration stops collecting samples after its configured window.
#
# Usage:
#   bash utils/run_aisbench_warmup.sh
#
# Optional environment variables:
#   MODEL_MAX_LEN=131073
#   WARMUP_BASELINE_REQUESTS=3
#   WARMUP_BASELINE_INPUT_LEN=65536
#   WARMUP_MIX_CONCURRENCY=2
#   WARMUP_MIX_LENGTHS=131072,30848,119424,4736,98304,1920,65536,32768
#   AISBENCH_TOOL_DIR=/path/to/aisbench_auto_tools_prefix
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tool_dir="${AISBENCH_TOOL_DIR:-$script_dir/../aisbench_auto_tools_prefix}"
model_max_len="${MODEL_MAX_LEN:-131073}"
baseline_requests="${WARMUP_BASELINE_REQUESTS:-3}"
mix_concurrency="${WARMUP_MIX_CONCURRENCY:-2}"

[[ "$model_max_len" =~ ^[0-9]+$ ]] && ((model_max_len > 1)) || {
  echo "MODEL_MAX_LEN must be an integer greater than 1: $model_max_len" >&2
  exit 2
}
[[ "$baseline_requests" =~ ^[0-9]+$ ]] && ((baseline_requests >= 3)) || {
  echo "WARMUP_BASELINE_REQUESTS must be an integer no smaller than 3: $baseline_requests" >&2
  exit 2
}
[[ "$mix_concurrency" =~ ^[0-9]+$ ]] && ((mix_concurrency >= 2)) || {
  echo "WARMUP_MIX_CONCURRENCY must be an integer no smaller than 2: $mix_concurrency" >&2
  exit 2
}

input_cap="$((model_max_len - 1))"
baseline_input_len="${WARMUP_BASELINE_INPUT_LEN:-$((input_cap / 2))}"

[[ "$baseline_input_len" =~ ^[0-9]+$ ]] && ((baseline_input_len >= 1 && baseline_input_len <= input_cap)) || {
  echo "WARMUP_BASELINE_INPUT_LEN must be in [1, $input_cap]: $baseline_input_len" >&2
  exit 2
}

[[ -d "$tool_dir" ]] || {
  echo "toolbox directory not found: $tool_dir" >&2
  exit 1
}
cd "$tool_dir"
[[ -f aisbench_test.py && -f config.py && -f generate_dataset.py && -f GSM8K.jsonl ]] || {
  echo "toolbox directory is incomplete: $tool_dir" >&2
  exit 1
}

run_aisbench() {
  local label="$1"
  local input_len="$2"
  local data_num="$3"
  local concurrency="$4"
  local dataset="${5:-}"
  local args=(
    --input_len "$input_len"
    --output_len 1
    --data_num "$data_num"
    --concurrency "$concurrency"
    --request_rate 0
    --dataset_type normal
    --test_type stream
    --api_mode completion
  )

  if [[ -n "$dataset" ]]; then
    args+=(--dataset "$dataset")
  fi

  echo "[CPP warmup] phase=$label input_len=$input_len data_num=$data_num concurrency=$concurrency"
  python3 aisbench_test.py "${args[@]}"
}

echo "[CPP warmup] restart the serving process before this run so the online calibration window is empty"
echo "[CPP warmup] phase 1/2: collect fresh and history-aware single-request samples"
run_aisbench single-baseline "$baseline_input_len" "$baseline_requests" 1

# Generate a deterministic, ordered variable-length dataset. With concurrency
# two, each long request is submitted immediately before a short request. The
# long request normally consumes the first scheduling budget and accumulates
# history before the short request joins it, creating the target
# long-history-plus-fresh mixed shape.
mixed_dataset="$(
  MODEL_INPUT_CAP="$input_cap" \
  WARMUP_MIX_LENGTHS="${WARMUP_MIX_LENGTHS:-}" \
  python3 - <<'PY'
import os
from pathlib import Path

from config import DATASET_PATH, MODEL_PATH
from generate_dataset import AutoTokenizer, _truncate_or_pad_text, create_dataset, write_data

input_cap = int(os.environ["MODEL_INPUT_CAP"])
configured = os.environ.get("WARMUP_MIX_LENGTHS", "").strip()

if configured:
    try:
        lengths = [int(item.strip()) for item in configured.split(",") if item.strip()]
    except ValueError as exc:
        raise SystemExit(f"invalid WARMUP_MIX_LENGTHS: {configured}") from exc
else:
    # Ratios reproduce the useful 128K calibration shapes at other context
    # limits while preserving their long/short ordering.
    reference_lengths = [131072, 30848, 119424, 4736, 98304, 1920, 65536, 32768]
    lengths = [max(1, round(input_cap * length / 131072)) for length in reference_lengths]

if len(lengths) < 4 or len(lengths) % 2 != 0:
    raise SystemExit("WARMUP_MIX_LENGTHS must contain an even number of at least four lengths")
if any(length < 1 or length > input_cap for length in lengths):
    raise SystemExit(f"all warmup lengths must be in [1, {input_cap}]: {lengths}")

root = Path(DATASET_PATH)
root.mkdir(parents=True, exist_ok=True)
model_name = Path(MODEL_PATH.rstrip("/")).name
length_tag = "_".join(str(length) for length in lengths)
dataset_path = root / f"CPP-warmup-{length_tag}-{model_name}.jsonl"

if not dataset_path.is_file():
    source_texts = create_dataset(MODEL_PATH, max(lengths), len(lengths), 0)
    if source_texts is None:
        raise SystemExit("failed to generate CPP warmup dataset; check GSM8K.jsonl and picked_ids.txt")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    dataset = [
        _truncate_or_pad_text(tokenizer, text, target_length)
        for text, target_length in zip(source_texts, lengths, strict=True)
    ]
    write_data(str(dataset_path), dataset, len(dataset))

print(dataset_path.resolve())
PY
)"

mixed_data_num="$(WARMUP_DATASET="$mixed_dataset" python3 - <<'PY'
import os
from pathlib import Path

path = Path(os.environ["WARMUP_DATASET"])
with path.open("r", encoding="utf-8") as file:
    print(sum(1 for line in file if line.strip()))
PY
)"

echo "[CPP warmup] phase 2/2: collect low/medium/high shared-max-waste mixed samples"
echo "[CPP warmup] mixed_dataset=$mixed_dataset"
run_aisbench mixed-shared-max "$input_cap" "$mixed_data_num" "$mix_concurrency" "$mixed_dataset"

echo "[CPP warmup] completed"
