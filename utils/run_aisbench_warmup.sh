#!/usr/bin/env bash
# CPP online-calibration warmup for single-request and mixed-batch shapes.
#
# Why this script exists
# ----------------------
# The history-aware CPP model contains a shared-max-waste feature. Single
# concurrency always has x_waste=0, so the old warmup pattern (five
# max_model_len requests at concurrency one) cannot identify that feature.
# It may also consume the complete online-calibration window before any mixed
# request is observed.
#
# This script runs two phases:
#   1. Three half-context requests at concurrency one. These provide the three
#      H=0 steps used to establish target latency and a small number of
#      single/history samples.
#   2. An ordered long/short dataset at concurrency two. A long request is
#      submitted before a short request so that the short request can join a
#      long-history request in a later forward.
#
# Run requirements
# ----------------
# 1. Restart the serving process immediately before running this script. CPP
#    stops collecting execution-time samples after max_fit_chunk is reached;
#    running this script against an already calibrated service cannot replace
#    its old samples.
# 2. Use the same tokenizer/model path in aisbench config.py as the service.
# 3. Enable chunked prefill. Otherwise a request longer than one scheduling
#    token budget cannot produce the intended history progression.
# 4. WARMUP_MIX_CONCURRENCY must not exceed the service max_num_seqs.
# 5. This is calibration traffic, not a throughput benchmark. Run the normal
#    performance stages only after this script completes.
#
# Deployment parameters and warmup parameters
# -------------------------------------------
# Let:
#   L = max_model_len - output_len
#   M = max_num_batched_tokens
#
# max_model_len limits the complete request. With output_len=1, every input
# length in this script must be in [1, max_model_len - 1]. Set MODEL_MAX_LEN to
# the exact value from the serving command. If WARMUP_MIX_LENGTHS is omitted,
# mixed lengths are scaled automatically from the 128K reference shapes using
# MODEL_MAX_LEN.
#
# max_num_batched_tokens limits sum(C_i) in one scheduler step; it does not
# limit the complete prompt length. This script does not pass that value to
# AISBench, because it is a server-side option, but it must be considered when
# choosing WARMUP_BASELINE_INPUT_LEN and WARMUP_MIX_LENGTHS.
#
# Recommended baseline length:
#   WARMUP_BASELINE_INPUT_LEN = min(L, 2 * M)
#
# This normally gives each baseline request one H=0 step and one or a few H>0
# steps. Avoid a baseline much larger than 2*M: three baseline requests may
# then exhaust max_fit_chunk before phase 2 starts. Avoid a baseline <= M: it
# may finish in one step and provide no history sample.
#
# Recommended mixed pairs, written as long,short,long,short,...:
#   min(L, 4*M), min(L, M),
#   min(L, 3*M), max(1, M/8),
#   min(L, 3*M), max(1, M/16),
#   min(L, 2*M), max(1, M/2)
#
# Prompt lengths only make a mixed shape likely; they do not directly set the
# per-step C_i. The scheduler still chooses C_i from its remaining token
# budget, CPP prediction, long_prefill_token_threshold, and request remainder.
# If long_prefill_token_threshold is enabled below M, use that threshold as
# the effective per-request chunk scale when selecting the lengths above.
#
# Examples
# --------
# max_model_len=131073, max_num_batched_tokens=32768:
#   MODEL_MAX_LEN=131073 \
#   WARMUP_BASELINE_INPUT_LEN=65536 \
#   WARMUP_MIX_LENGTHS=131072,32768,98304,4096,98304,2048,65536,16384 \
#   bash utils/run_aisbench_warmup.sh
#
# max_model_len=131073, max_num_batched_tokens=65536:
#   MODEL_MAX_LEN=131073 \
#   WARMUP_BASELINE_INPUT_LEN=131072 \
#   WARMUP_MIX_LENGTHS=131072,65536,131072,8192,131072,4096,131072,32768 \
#   bash utils/run_aisbench_warmup.sh
#
# Basic usage with the built-in 128K defaults:
#   bash utils/run_aisbench_warmup.sh
#
# Optional environment variables
# ------------------------------
# MODEL_MAX_LEN
#   Service max_model_len. Default: 131073. The script reserves one token for
#   output and validates all generated input lengths against this value.
#
# WARMUP_BASELINE_REQUESTS
#   Number of sequential baseline requests. Default/minimum: 3, matching the
#   current target-latency calibration requirement of three H=0 samples.
#
# WARMUP_BASELINE_INPUT_LEN
#   Input length of every baseline request. Default: (MODEL_MAX_LEN - 1) / 2.
#   Override it with min(L, 2*M) when max_num_batched_tokens differs from the
#   128K deployment baseline.
#
# WARMUP_MIX_CONCURRENCY
#   Maximum concurrency during the mixed phase. Default/minimum: 2. Keep it no
#   larger than max_num_seqs. Two is preferred because it produces easier to
#   interpret request pairs without introducing unnecessary batch-size terms.
#
# WARMUP_MIX_LENGTHS
#   Comma-separated, ordered prompt lengths. It must contain an even number of
#   at least four values, each in [1, MODEL_MAX_LEN - 1]. Adjacent values are
#   intended as long/short pairs. If omitted, the 128K reference sequence
#   131072,30848,119424,4736,98304,1920,65536,32768 is scaled by input_cap.
#
# AISBENCH_TOOL_DIR
#   Directory containing aisbench_test.py, config.py, generate_dataset.py, and
#   GSM8K.jsonl. Default: ../aisbench_auto_tools_prefix relative to this file.
#
# Dataset reuse
# -------------
# The generated mixed JSONL filename contains every requested length and the
# model directory name. Re-running the same configuration reuses that file.
# Change WARMUP_MIX_LENGTHS or remove the cached JSONL if a new dataset is
# required for the same model.
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
