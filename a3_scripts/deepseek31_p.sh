#!/bin/bash
set -euo pipefail
# Set CPP_ENABLED=false for the archived CPP-off comparison.
CPP_ENABLED="${CPP_ENABLED:-true}"
unset https_proxy
unset http_proxy
nic_name="${SERVICE_NIC:-data0.172}"
local_ip="${SERVICE_IP:-$(ifconfig "$nic_name" | awk '$1 == "inet" { print $2; exit }')}"
[[ -n "$local_ip" ]] || { echo "Unable to resolve IP for NIC: $nic_name" >&2; exit 1; }

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

export OMP_PROC_BIND=false
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=512
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_USE_V1=1
export TASK_QUEUE_ENABLE=1
export ASCEND_LAUNCH_BLOCKING=0
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120
export ACL_OP_INIT_MODE=1
export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["hccs:device"]}'
export ASCEND_LOCAL_COMM_RES='{"version":"1.2"}'
export VLLM_USE_V2_MODEL_RUNNER=1

additional_config=$(cat <<EOF
{
    "enable_flashcomm1": true,
    "scheduler_config": {
      "profiling_chunk_config": {"enabled":${CPP_ENABLED}, "smooth_factor":1.0, "min_chunk":4096}
    }
  }
EOF
)

vllm serve /mnt/weight/DeepSeek-V3.1-Terminus-w8a8-QuaRot-lfs \
  --host 0.0.0.0 \
  --port 8003 \
  --served-model-name model \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --enable-expert-parallel \
  --max-num-seqs 32 \
  --max-model-len 131072 \
  --max-num-batched-tokens 32768 \
  --gpu-memory-utilization 0.85 \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --no-async-scheduling \
  --trust-remote-code \
  --quantization ascend \
  --additional-config "$additional_config" \
  --kv-transfer-config \
  '{
    "kv_connector": "MooncakeConnectorV2",
    "kv_role": "kv_producer",
    "kv_port": "36000",
    "engine_id": "0",
    "kv_connector_extra_config": {
      "prefill": {
        "pp_size": 2,
        "dp_size": 1,
        "tp_size": 8
      },
      "decode": {
        "dp_size": 32,
        "tp_size": 1
      }
    }
  }'