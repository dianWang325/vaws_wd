#!/bin/bash
set -euo pipefail

# Set NODE_RANK=0 on the API/master node and NODE_RANK=1 on the headless node.
# Update the IP/NIC values below for your machines; both nodes use the same master.
node_rank="${NODE_RANK:-0}"
master_addr="${MASTER_ADDR:-10.246.63.24}"
master_port="${MASTER_PORT:-7060}"
nic_name="${SERVICE_NIC:-data0.172}"
local_ip="${SERVICE_IP:-$(ifconfig "$nic_name" | awk '$1 == "inet" { print $2; exit }')}"
if [[ "$node_rank" != 0 && "$node_rank" != 1 ]]; then
  echo "NODE_RANK must be 0 or 1" >&2
  exit 2
fi
[[ -n "$local_ip" ]] || { echo "Set SERVICE_IP for this machine" >&2; exit 1; }

# Set the same PROFILE_TAG on both P nodes when one combined capture is needed.
profile_tag="${PROFILE_TAG:-glm53_pp2_tp16_$(date +%Y%m%d_%H%M%S)}"
torch_profiler_dir="${TORCH_PROFILER_DIR:-/mnt/share/w00985415/profiling/${profile_tag}}"
mkdir -p "$torch_profiler_dir"

profiler_config="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${torch_profiler_dir}\",\"torch_profiler_with_stack\":false,\"torch_profiler_record_shapes\":true,\"torch_profiler_with_flops\":false,\"torch_profiler_with_memory\":false,\"detailed_trace_annotation\":true}"

echo "Node rank: $node_rank"
echo "Local IP/NIC: $local_ip/$nic_name"
echo "Master: $master_addr:$master_port"
echo "Torch profiler directory: $torch_profiler_dir"

unset ftp_proxy
unset https_proxy
unset http_proxy
unset DYNAMIC_EPLB
unset EXPERT_MAP_RECORD
unset HCCL_ALGO

export PROMETHEUS_MULTIPROC_DIR=/dev/shm/vllm_metrics
mkdir -p "$PROMETHEUS_MULTIPROC_DIR"

# Match the working wd_test1009 source compatibility and Mooncake HCCS transport.
export VLLM_VERSION=0.31.0
export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["hccs:device"]}'
export ASCEND_LOCAL_COMM_RES='{"version":"1.2"}'

export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export VLLM_ENGINE_READY_TIMEOUT_S=10000
export VLLM_RPC_TIMEOUT=3600000
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_PP_LAYER_PARTITION="42,36"

export HCCL_BUFFSIZE=512
export HCCL_BUFFSIZE_EP=300
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_HOST_IP="$local_ip"
export HCCL_IF_IP="$local_ip"
export TP_SOCKET_IFNAME="$nic_name"
export HCCL_HOST_SOCKET_PORT_RANGE=auto
export HCCL_SOCKET_IFNAME="$nic_name"
export GLOO_SOCKET_IFNAME="$nic_name"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_DFS_CONFIG="task_exception:off,inconsistent_check:off"

role_args=()
if [[ "$node_rank" == "1" ]]; then
  role_args+=(--headless)
fi

vllm serve /mnt/weight/GLM-5.3-w8a8c8-Eco-Tech \
  --host 0.0.0.0 \
  --port 30060 \
  --no-enable-prefix-caching \
  --enable-expert-parallel \
  --tensor-parallel-size 16 \
  --pipeline-parallel-size 2 \
  --nnodes 2 \
  --node-rank "$node_rank" \
  --master-addr "$master_addr" \
  --master-port "$master_port" \
  --distributed-executor-backend mp \
  --seed 1024 \
  --served-model-name glm-5.2 \
  --max-model-len 200000 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64 \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --quantization ascend \
  --kv-cache-dtype int8 \
  --attention_config.indexer_kv_dtype int8 \
  --enable-chunked-prefill \
  --no-async-scheduling \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --profiler-config "$profiler_config" \
  --additional-config '{
    "enable_dsa_cp": true,
    "enable_flashcomm1": true,
    "enable_fused_mc2": 1,
    "multistream_overlap_shared_expert": false,
    "weight_nz_mode": 1,
    "enable_cpu_binding": true,
    "scheduler_config": {
      "profiling_chunk_config": {
        "enabled": true,
        "smooth_factor":1.0
      }
    }
  }' \
  --speculative-config '{"num_speculative_tokens":1,"method":"deepseek_mtp","enforce_eager":true}' \
  --kv-transfer-config \
  '{
    "kv_connector": "MooncakeConnectorV2",
    "kv_role": "kv_producer",
    "kv_port": "30100",
    "engine_id": "1",
    "kv_connector_extra_config": {
      "prefill": {
        "dp_size": 1,
        "tp_size": 16,
        "pp_size": 2,
        "pp_layer_partition": "42,36"
      },
      "decode": {
        "dp_size": 2,
        "tp_size": 4,
        "pp_size": 1
      }
    }
  }' \
  "${role_args[@]}"
