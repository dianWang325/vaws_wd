#!/bin/bash
set -euo pipefail

nic_name="${SERVICE_NIC:-data0.172}"
local_ip="${SERVICE_IP:-$(ifconfig "$nic_name" | awk '$1 == "inet" { print $2; exit }')}"
[[ -n "$local_ip" ]] || { echo "Unable to resolve IP for NIC: $nic_name" >&2; exit 1; }

export HCCL_IF_IP="$local_ip"
export GLOO_SOCKET_IFNAME="$nic_name"
export TP_SOCKET_IFNAME="$nic_name"
export HCCL_SOCKET_IFNAME="$nic_name"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_VERSION=0.31.0
export VLLM_USE_V2_MODEL_RUNNER=1
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
export PYTHONHASHSEED=0
export ACL_OP_INIT_MODE=1
export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["hccs:device"]}'
export ASCEND_LOCAL_COMM_RES='{"version":"1.2"}'

exec vllm serve /mnt/weight/Qwen3-30B-A3B \
    --host 0.0.0.0 \
    --port 13701 \
    --served-model-name qwen \
    --data-parallel-size 2 \
    --tensor-parallel-size 2 \
    --enable-prefix-caching \
    --max-model-len 131072 \
    --max-num-batched-tokens 256 \
    --gpu-memory-utilization 0.9 \
    --compilation-config '{
        "cudagraph_mode": "FULL_DECODE_ONLY"
    }' \
    --hf-overrides '{
        "rope_theta": 1000000,
        "rope_scaling": {
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 40960
        },
        "max_model_len": 262144
    }' \
    --kv-transfer-config '{
        "kv_connector": "MooncakeConnectorV2",
        "kv_role": "kv_consumer",
        "kv_port": "30100",
        "engine_id": "1",
        "kv_connector_extra_config": {
            "prefill": {
                "pp_size": 2,
                "dp_size": 1,
                "tp_size": 2
            },
            "decode": {
                "dp_size": 2,
                "tp_size": 2
            }
        }
    }'
