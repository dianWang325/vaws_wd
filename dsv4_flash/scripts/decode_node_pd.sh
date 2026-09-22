# PD 分离部署 - Decode 节点（TCP 跨机，MooncakeHybridConnector）
# 目标机：80.48.33.120（容器 wd_test0921 内运行，/home 已挂载进容器）
# 与 decode_node.sh 的差异：网卡/IP 环境变量填实 + 新增 socket/超时环境变量
#   + 新增 DP 寻址参数（--data-parallel-address/--data-parallel-rpc-port）+ 新增 --kv-transfer-config
# 并行配置未改动：DP2 x TP8（内置 DP，与 vllm-ascend 官方 PD 部署指南一致）
nic_name="enp48s3u1u2"  # 80.48.33.120 实测业务网卡
local_ip="80.48.33.120"
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export HCCL_IF_IP=$local_ip

export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000

export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096
export VLLM_USE_V2_MODEL_RUNNER=1

vllm serve /mnt/share/weight/DeepSeek-V4-Flash-DSpark-w4a8-scope-ci \
    --host 0.0.0.0 \
    --max-model-len 204800 \
    --max-num-batched-tokens 20480 \
    --served-model-name dsv4 \
    --gpu-memory-utilization 0.85 \
    --api-server-count 1 \
    --max-num-seqs 64 \
    --data-parallel-size 2 \
    --data-parallel-address 80.48.33.120 \
    --data-parallel-rpc-port 12321 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --enable-chunked-prefill \
    --tokenizer-mode deepseek_v4 \
    --model-loader-extra-config='{"enable_multithread_load": true, "num_threads": 128}' \
    --quantization ascend \
    --port 18080 \
    --no-enable-prefix-caching \
    --speculative-config '{"method": "dspark", "num_speculative_tokens": 5, "enforce_eager": true}' \
    --block-size 32 \
    --attention_config.indexer_kv_dtype int8 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --additional-config '
    {"ascend_compilation_config":{
        "enable_npugraph_ex": true,
        "enable_static_kernel": false
        },
    "enable_cpu_binding": true,
    "multistream_overlap_shared_expert": true}' \
    --kv-transfer-config '
    {"kv_connector": "MooncakeHybridConnector",
     "kv_role": "kv_consumer",
     "kv_port": "30200",
     "engine_id": "1",
     "kv_connector_extra_config": {
         "prefill": {"dp_size": 1, "tp_size": 8, "pp_size": 2},
         "decode":  {"dp_size": 2, "tp_size": 8, "pp_size": 1}
     }}'
