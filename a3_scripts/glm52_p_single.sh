#!/bin/bash
set -euo pipefail
export ASCEND_LOCAL_COMM_RES='{"version":"1.2"}'
export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["hccs:device"]}'
nic_name="data0.172"
local_ip="172.27.5.139"
# Single-node prefill service (PP=2, TP=8, 16 local devices).
# Update the IP/NIC values below for this machine.
nic_name="${SERVICE_NIC:-enp162s0f0}"
local_ip="${SERVICE_IP:-10.246.63.28}"

# Historical V1/synchronous and V2/asynchronous profiles.
PROFILE="${PROFILE:-mrv1}"
CPP_ENABLED="${CPP_ENABLED:-true}"
MTP_ENABLED="${MTP_ENABLED:-true}"
if [[ "$PROFILE" == mrv1 ]]; then
  runner=0
  scheduling=--no-async-scheduling
  max_model_len=200000
  max_num_batched_tokens=16384
elif [[ "$PROFILE" == mrv2 ]]; then
  runner=1
  scheduling=--async-scheduling
  max_model_len=266240
  max_num_batched_tokens=32768
else
  echo "PROFILE must be mrv1 or mrv2" >&2
  exit 2
fi
[[ -n "$local_ip" ]] || { echo "Set SERVICE_IP for this machine" >&2; exit 1; }

# Override PROFILE_TAG when a specific capture name is needed.
profile_tag="${PROFILE_TAG:-glm52_single_pp2_tp8_$(date +%Y%m%d_%H%M%S)}"
torch_profiler_dir="${TORCH_PROFILER_DIR:-/mnt/share/w00985415/profiling/${profile_tag}}"
mkdir -p "$torch_profiler_dir"

profiler_config="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${torch_profiler_dir}\",\"torch_profiler_with_stack\":false,\"torch_profiler_record_shapes\":true,\"torch_profiler_with_flops\":false,\"torch_profiler_with_memory\":false,\"detailed_trace_annotation\":true}"

echo "Local IP/NIC: $local_ip/$nic_name"
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
export VLLM_USE_V2_MODEL_RUNNER="$runner"
export VLLM_PP_LAYER_PARTITION="42,36"

export HCCL_BUFFSIZE=512
export HCCL_BUFFSIZE_EP=300
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_IF_IP="$local_ip"
export HCCL_HOST_SOCKET_PORT_RANGE=auto
export HCCL_SOCKET_IFNAME="$nic_name"
export GLOO_SOCKET_IFNAME="$nic_name"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_DFS_CONFIG="task_exception:off,inconsistent_check:off"

speculative_args=()
if [[ "$MTP_ENABLED" == true ]]; then
  speculative_args+=(--speculative-config '{"num_speculative_tokens":1,"method":"deepseek_mtp","enforce_eager":true}')
fi

additional_config=$(cat <<EOF
{
    "enable_fused_mc2": 1,
    "multistream_overlap_shared_expert": false,
    "weight_nz_mode": 1,
    "enable_cpu_binding": true,
    "scheduler_config": {
      "profiling_chunk_config": {
        "enabled": ${CPP_ENABLED},
        "smooth_factor":0.8
      }
    }
  }
EOF
)

vllm serve /mnt/weight/GLM-5.2-w8a8c8 \
  --host 0.0.0.0 \
  --port 30060 \
  --no-enable-prefix-caching \
  --enable-expert-parallel \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --distributed-executor-backend mp \
  --seed 1024 \
  --served-model-name glm-5.2 \
  --max-model-len "$max_model_len" \
  --max-num-batched-tokens "$max_num_batched_tokens" \
  --max-num-seqs 16 \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --quantization ascend \
  --kv-cache-dtype int8 \
  --attention_config.indexer_kv_dtype int8 \
  --enable-chunked-prefill \
  "$scheduling" \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --profiler-config "$profiler_config" \
  --additional-config "$additional_config" \
  "${speculative_args[@]}" \
  --kv-transfer-config \
  '{
    "kv_connector": "MooncakeConnectorV2",
    "kv_role": "kv_producer",
    "kv_port": "30100",
    "engine_id": "1",
    "kv_connector_extra_config": {
      "prefill": {
        "dp_size": 1,
        "tp_size": 8,
        "pp_size": 2,
        "pp_layer_partition": "42,36"
      },
      "decode": {
        "dp_size": 2,
        "tp_size": 4,
        "pp_size": 1
      }
    }
  }'
