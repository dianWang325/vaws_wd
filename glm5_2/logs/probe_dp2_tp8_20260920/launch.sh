#!/usr/bin/env bash
set -euo pipefail
export VLLM_HOST_IP=80.5.9.129
export HCCL_IF_IP=80.5.9.129
export GLOO_SOCKET_IFNAME=enp194s0f0
export TP_SOCKET_IFNAME=enp194s0f0
export HCCL_SOCKET_IFNAME=enp194s0f0
export VLLM_ASCEND_ENABLE_NZ=1
export VLLM_USE_V2_MODEL_RUNNER=1
export HCCL_OP_EXPANSION_MODE=AIV
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=20
export HCCL_BUFFSIZE=768
export HCCL_HOST_SOCKET_PORT_RANGE=auto
export HCCL_NPU_SOCKET_PORT_RANGE=auto
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=204
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TASK_QUEUE_ENABLE=1
export VLLM_ENGINE_READY_TIMEOUT_S=100000
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export VLLM_PP_LAYER_PARTITION=38,40
exec timeout --signal=TERM 900s vllm serve /mnt/weight/GLM-5.2-W4A8C8-0713-MTP \
  --seed 1024 --host 0.0.0.0 --port 19000 --served-model-name glm-52 \
  --max-model-len 204800 --max-num-batched-tokens 20480 --gpu-memory-utilization 0.75 \
  --api-server-count 1 --max-num-seqs 8 --no-enable-prefix-caching \
  --data-parallel-size 2 --data-parallel-size-local 1  \
  --data-parallel-address 80.5.9.129 --data-parallel-rpc-port 18591 \
  --pipeline-parallel-size 2 --tensor-parallel-size 8 \
  --distributed-executor-backend mp --nnodes 2 --node-rank 0 \
  --master-addr 80.5.9.129 --master-port 17060 \
  --prefill-context-parallel-size 1 --decode-context-parallel-size 8 \
  --cp-kv-cache-interleave-size 128 --enforce-eager \
  --additional-config '{"enable_flashcomm1":true,"enable_dsa_cp":true,"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"scheduler_config":{"profiling_chunk_config":{"enabled":true}},"multistream_overlap_shared_expert":true,"enable_mc2_hierarchy_comm":false,"enable_sparse_sfa_c8":true,"enable_sparse_li_c8":true,"enable_cpu_binding":true,"recompute_scheduler_enable":false}' \
  --quantization ascend --enable-expert-parallel --safetensors-load-strategy prefetch
