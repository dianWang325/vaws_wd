#!/bin/bash
set -euo pipefail

nic_name="${SERVICE_NIC:-data0.172}"
local_ip="${SERVICE_IP:-$(ifconfig "$nic_name" | awk '$1 == "inet" { print $2; exit }')}"
[[ -n "$local_ip" ]] || { echo "Unable to resolve IP for NIC: $nic_name" >&2; exit 1; }

export HCCL_IF_IP="$local_ip"
export GLOO_SOCKET_IFNAME="$nic_name"
export TP_SOCKET_IFNAME="$nic_name"
export HCCL_SOCKET_IFNAME="$nic_name"
# export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_USE_V2_MODEL_RUNNER=0
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export PYTHONHASHSEED=0
export ACL_OP_INIT_MODE=1
export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["hccs:device"]}'
export ASCEND_LOCAL_COMM_RES='{"version":"1.2"}'

exec vllm serve /mnt/share/w00985415/weights/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 13700 \
    --gpu-memory-utilization 0.9 \
    --served-model-name qwen \
    --tensor-parallel-size 4 \
    --speculative-config '{"method":"suffix","num_speculative_tokens":8}' \
    --enforce-eager \
    --max-model-len 32768 \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 64 \
    --distributed_executor_backend "mp" \
    --no-enable-prefix-caching \
    --no-async-scheduling \