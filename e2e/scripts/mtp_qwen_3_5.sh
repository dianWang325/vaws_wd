#!/bin/sh
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_USE_V2_MODEL_RUNNER=1
export HCCL_BUFFSIZE=512
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True


vllm serve /mnt/weight/Qwen3.5-27B \
    --host 0.0.0.0 \
    --port 10800 \
    --data-parallel-size 1 \
    --tensor-parallel-size 2 \
    --prefill-context-parallel-size 2 \
    --pipeline-parallel-size 2 \
    --seed 1024 \
    --quantization ascend \
    --served-model-name qwen3.5 \
    --enable-expert-parallel \
    --enable-chunked-prefill \
    --no-enable-prefix-caching \
    --max-num-seqs 32 \
    --max-model-len 87040 \
    --max-num-batched-tokens 10240 \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    --no-enable-prefix-caching \
    --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_cpu_binding":true, "scheduler_config":{"profiling_chunk_config":{"enabled":true}}}'