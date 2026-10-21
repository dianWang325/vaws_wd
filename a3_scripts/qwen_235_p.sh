nic_name="<your_nic_name>"
local_ip="<your_ip>"

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

export HCCL_BUFFSIZE=512
export HCCL_OP_EXPANSION_MODE="AIV"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V2_MODEL_RUNNER=1

export OMP_NUM_THREADS=1
# echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
# sysctl -w vm.swappiness=0
# sysctl -w kernel.numa_balancing=0
# sysctl kernel.sched_migration_cost_ns=50000
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export TASK_QUEUE_ENABLE=1
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH

export ASCEND_RT_VISIBLE_DEVICES=$1
## 10.246.63.24:/mnt/weight/Qwen3-235B-A22B-W8A8
## 80.5.9.x/80.5.17.x(80.5.9.128除外):/mnt/weight/Qwen3-235B-A22B-W8A8
vllm serve /mnt/weight/Qwen3-235B-A22B-W8A8 \
    --host 0.0.0.0 \
    --port 30060 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --pipeline-parallel-size 2 \
    --served-model-name qwen3_235b \
    --max-model-len 131073 \
    --max-num-batched-tokens 32768 \
    --max-num-seqs 24 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --quantization ascend \
    --async-scheduling \
    --no-enable-prefix-caching \
    --enforce-eager \
    --additional-config '{"enable_flashcomm1": true, "enable_fused_mc2": 1}' \
    --hf-overrides '{
        "max_position_embeddings": 131073,
        "rope_parameters": {
            "rope_type": "yarn",
            "rope_theta": 1000000.0,
            "factor": 4.0,
            "original_max_position_embeddings": 32768
        }
    }' \
    --kv-transfer-config \
        '{"kv_connector": "MooncakeConnectorV1",
        "kv_role": "kv_producer",
        "kv_port": "30000",
        "engine_id": "0",
        "kv_connector_extra_config": {
             "use_ascend_direct": true,
             "prefill": {
                    "pp_size": 2,
                    "tp_size": 8
             },
             "decode": {
                    "dp_size": 8,
                    "tp_size": 4
             }
        }
        }'