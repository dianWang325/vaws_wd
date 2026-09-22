nic_name="eth0"  # network card name
local_ip="192.0.0.1"
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
    --max-model-len 204800 \
    --max-num-batched-tokens 20480 \
    --served-model-name dsv4 \
    --gpu-memory-utilization 0.85 \
    --api-server-count 1 \
    --max-num-seqs 64 \
    --data-parallel-size 2 \
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
    "multistream_overlap_shared_expert": true}'