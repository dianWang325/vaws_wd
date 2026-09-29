export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096

vllm serve /mnt/weight/dsv4_flash_w4a8_0801/DeepSeek-V4-Flash-0731-w4a8 \
    --max-model-len 87040 \
    --max-num-batched-tokens 10240 \
    --served-model-name dsv4 \
    --gpu-memory-utilization 0.9 \
    --api-server-count 1 \
    --max-num-seqs 64 \
    --tensor-parallel-size 2 \
    --prefill-context-parallel-size 2 \
    --pipeline-parallel-size 2 \
    --enable-expert-parallel \
    --enable-chunked-prefill \
    --tokenizer-mode deepseek_v4 \
    --reasoning-parser deepseek_v4 \
    --model-loader-extra-config='{"enable_multithread_load": true, "num_threads": 128}' \
    --quantization ascend \
    --enforce_eager \
    --port 8000 \
    --block-size 32 \
    --attention_config.indexer_kv_dtype int8 \
    --speculative-config '{"method": "dspark", "num_speculative_tokens": 1, "enforce_eager": true}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --additional-config '
    {"ascend_compilation_config":{
        "enable_npugraph_ex": true,
        "enable_static_kernel": false
        },
    "enable_cpu_binding": true,
    "multistream_overlap_shared_expert": true}'