#!/usr/bin/env bash
set -euo pipefail

# Run inside w00985415_1124. Defaults fit a short smoke test.
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}"
PORT="${PORT:-18481}"
OUT="${OUT:-/home/w00985415/env_1124/dspine-smoke-output}"
mkdir -p "$OUT"
if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "Port $PORT already has a healthy service; choose another PORT." >&2
  exit 1
fi

vllm serve /home/weight/Qwen3.8-27B \
  --served-model-name dspine-smoke \
  --host 127.0.0.1 --port "$PORT" \
  --language-model-only --dtype bfloat16 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.30 \
  --max-model-len 2048 --max-num-seqs 1 \
  --max-num-batched-tokens 1024 \
  --enforce-eager --no-enable-prefix-caching \
  --generation-config vllm --seed 0 \
  --speculative-config '{"method":"dflash","model":"/home/weight/Qwen3.8-27B-DSpine-b16","num_speculative_tokens":15,"enable_adaptive_verification":false,"draft_sample_method":"greedy"}' \
  >"$OUT/server.log" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" >"$OUT/server.pid"
cleanup() {
  if [[ "${KEEP_SERVER:-0}" != 1 ]]; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

ready=0
for ((i=0; i<180; i++)); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    tail -n 100 "$OUT/server.log"
    exit 1
  fi
  if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 5
done
if [[ "$ready" != 1 ]]; then
  echo "Server readiness timed out; see $OUT/server.log" >&2
  exit 1
fi

curl -fsS --max-time 10 "http://127.0.0.1:$PORT/v1/models" >"$OUT/models.json"
curl -fsS --max-time 180 "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"dspine-smoke","messages":[{"role":"user","content":"Count from 1 to 20, separated by commas."}],"temperature":0,"max_tokens":128,"chat_template_kwargs":{"enable_thinking":false}}' \
  >"$OUT/response.json"
curl -fsS --max-time 10 "http://127.0.0.1:$PORT/metrics" >"$OUT/metrics.txt"

# Check that generation succeeded and speculative drafting actually ran.
python - "$OUT" <<'PY'
import json
import pathlib
import re
import sys

out = pathlib.Path(sys.argv[1])
response = json.loads((out / "response.json").read_text())
assert response.get("choices"), response
assert response["usage"]["completion_tokens"] > 0, response
assert response["choices"][0]["message"].get("content"), response
metrics = (out / "metrics.txt").read_text()
drafted = [
    float(line.split()[-1])
    for line in metrics.splitlines()
    if re.match(r"vllm:spec_decode_num_draft_tokens_total(?:\{|\s)", line)
]
assert sum(drafted) > 0, "No speculative draft tokens recorded; inspect metrics.txt"
print(json.dumps(response, ensure_ascii=False, indent=2))
print("PASS: health, models, chat generation, and speculative draft metrics")
PY
