set -euo pipefail
export PD_SCRIPTS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${PD_EXPERIMENT_ROOT:?Set PD_EXPERIMENT_ROOT to the experiment input/output directory}"
cd "$PD_EXPERIMENT_ROOT"
mkdir -p pd_pcp
cd pd_pcp
export EXPERIMENT_ROOT="$PWD" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PD_OUTPUT_DIR="$PWD"
if [[ -n "${PD_VARIANT:-}" ]]; then
    [[ "$PD_VARIANT" == A || "$PD_VARIANT" == B ]]
    export PD_OUTPUT_DIR="$PWD/formal_$PD_VARIANT"
fi
mkdir -p "$PD_OUTPUT_DIR"
export TMPDIR="$PWD/tmp" XDG_CACHE_HOME="$PWD/cache" VLLM_CACHE_ROOT="$PWD/cache/vllm"
export ASCEND_PROCESS_LOG_PATH="$PD_OUTPUT_DIR/ascend_logs" HF_HOME="$PWD/cache/hf" ASCEND_WORK_PATH="$PWD"
export ASCEND_REPO="${ASCEND_REPO:-/mnt/share/w00985415/proj_1104/vllm-ascend}"
export VLLM_REPO="${VLLM_REPO:-/mnt/share/w00985415/proj_1104/vllm}"
export PYTHONPATH="$ASCEND_REPO:$VLLM_REPO:${PYTHONPATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_SERVER_DEV_MODE=1
export VLLM_ENGINE_READY_TIMEOUT_S=1800
export SUFFIX_SPEC='{"method":"suffix","num_speculative_tokens":8,"suffix_decoding_max_tree_depth":24,"suffix_decoding_max_cached_requests":0,"suffix_decoding_corpus_tokens":0,"enable_adaptive_verification":true}'
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$ASCEND_PROCESS_LOG_PATH"
mount --make-rprivate /
mount --bind "$TMPDIR" /tmp
export TMPDIR=/tmp
cp "../source_snapshots/suffix.${PD_VARIANT:-A}.py" "$PD_OUTPUT_DIR/private_suffix.py"
mount --bind "$PD_OUTPUT_DIR/private_suffix.py" "$ASCEND_REPO/vllm_ascend/ops/triton/spec_decode/suffix.py"
exec python -u "$PD_SCRIPTS/dsv4flash_pd_controller.py"
