#!/usr/bin/env bash
# GLM-5.2, DP2/PP2/TP8 with DSA-CP on 80.5.17.110 and 80.5.17.107.
# Each 16-card node hosts one DP replica with two local pipeline stages.
# Run as: bash glm5_2/scripts/glm52_dsacp_V2.sh head|worker
set -euo pipefail

role="${1:?expected head or worker}"
case "$role" in
  head)
    local_ip=80.5.17.110
    node_rank=0
    server_role_args=(--api-server-count 1)
    ;;
  worker)
    local_ip=80.5.17.107
    node_rank=1
    server_role_args=(--headless)
    ;;
  *) echo "invalid role: $role" >&2; exit 2 ;;
esac

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
log_dir="${script_dir}/../logs"
mkdir -p "$log_dir"
exec > "${log_dir}/glm52_dp2_pp2_tp8_dsacp_${role}.log" 2>&1

export VLLM_HOST_IP="$local_ip"
export HCCL_IF_IP="$local_ip"
export GLOO_SOCKET_IFNAME=enp48s3u1u1
export TP_SOCKET_IFNAME=enp48s3u1u1
export HCCL_SOCKET_IFNAME=enp48s3u1u1
export VLLM_ASCEND_ENABLE_NZ=1
export VLLM_USE_V2_MODEL_RUNNER=1
export HCCL_OP_EXPANSION_MODE=AIV
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=20
export HCCL_BUFFSIZE=768
export HCCL_HOST_SOCKET_PORT_RANGE=auto
export HCCL_NPU_SOCKET_PORT_RANGE=auto
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=204
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TASK_QUEUE_ENABLE=1
export VLLM_ENGINE_READY_TIMEOUT_S=100000
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export VLLM_PP_LAYER_PARTITION=38,40

vllm serve /mnt/weight/GLM-5.2-W4A8C8-0713-MTP \
  --seed 1024 \
  --host 0.0.0.0 \
  --port 9000 \
  --served-model-name glm-52 \
  --max-model-len 204800 \
  --max-num-batched-tokens 20480 \
  --gpu-memory-utilization 0.75 \
  "${server_role_args[@]}" \
  --max-num-seqs 8 \
  --no-enable-prefix-caching \
  --data-parallel-size 2 \
  --data-parallel-size-local 1 \
  --data-parallel-start-rank "$node_rank" \
  --data-parallel-address 80.5.17.110 \
  --data-parallel-rpc-port 16591 \
  --pipeline-parallel-size 2 \
  --tensor-parallel-size 8 \
  --distributed-executor-backend mp \
  --nnodes 2 \
  --node-rank "$node_rank" \
  --master-addr 80.5.17.110 \
  --master-port 7060 \
  --prefill-context-parallel-size 1 \
  --decode-context-parallel-size 8 \
  --cp-kv-cache-interleave-size 128 \
  --enforce-eager \
  --additional-config '{
    "enable_flashcomm1": true,
    "enable_dsa_cp": true,
    "ascend_compilation_config": {
      "enable_npugraph_ex": true,
      "enable_static_kernel": false
    },
    "scheduler_config": {"profiling_chunk_config": {"enabled": true}},
    "multistream_overlap_shared_expert": true,
    "enable_mc2_hierarchy_comm": false,
    "enable_sparse_sfa_c8": true,
    "enable_sparse_li_c8": true,
    "enable_cpu_binding": true,
    "recompute_scheduler_enable": false
  }' \
  --speculative-config '{"num_speculative_tokens": 1, "method": "deepseek_mtp", "enforce_eager": true}' \
  --quantization ascend \
  --enable-expert-parallel \
  --safetensors-load-strategy prefetch
