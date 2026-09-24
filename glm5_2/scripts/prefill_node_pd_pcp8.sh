#!/usr/bin/env bash
# PD 分离部署 - Prefill 节点 PCP8 版（TCP 跨机，MooncakeConnectorV2）
# 目标机：80.5.9.135（容器 wd_test0921 内运行，/home 已挂载进容器）
# 依赖分支：test/cpp_async_mtp_pcp0924 @ 4ec50e551（含完整 PR #17410：SFA-PCP veto + PCP PIECEWISE 图支持）
# 拓扑：PP2 x TP1 x PCP8（单机 16 卡；对齐 DSV4 nightly 的 TP1+PCP 口径，PR #16846）
# P 节点保留 CPP(profiling_chunk)+SRF(short_request_first)
# 日志：glm5_2/logs/glm52_pd_pcp8_prefill.log
set -euo pipefail

nic_name="enp194s0f0"  # 80.5.9.135 实测业务网卡
local_ip="80.5.9.135"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
log_dir="${script_dir}/../logs"
mkdir -p "$log_dir"
exec > "${log_dir}/glm52_pd_pcp8_prefill.log" 2>&1

export VLLM_HOST_IP="$local_ip"
export HCCL_IF_IP="$local_ip"
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000

export VLLM_ASCEND_ENABLE_NZ=1
export VLLM_USE_V2_MODEL_RUNNER=1
export HCCL_OP_EXPANSION_MODE=AIV
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=20
export HCCL_BUFFSIZE=768
export HCCL_HOST_SOCKET_PORT_RANGE=auto
export HCCL_NPU_SOCKET_PORT_RANGE=auto
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=1200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:${LD_PRELOAD:-}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TASK_QUEUE_ENABLE=1
export VLLM_ENGINE_READY_TIMEOUT_S=100000
export VLLM_PP_LAYER_PARTITION=38,40

vllm serve /mnt/weight/GLM-5.2-W4A8C8-0713-MTP \
  --seed 1024 \
  --host 0.0.0.0 \
  --port 18080 \
  --served-model-name glm-52 \
  --max-model-len 106496 \
  --max-num-batched-tokens 20480 \
  --gpu-memory-utilization 0.85 \
  --api-server-count 1 \
  --max-num-seqs 8 \
  --no-enable-prefix-caching \
  --pipeline-parallel-size 2 \
  --tensor-parallel-size 1 \
  --prefill-context-parallel-size 8 \
  --cp-kv-cache-interleave-size 128 \
  --enable-chunked-prefill \
  --async-scheduling \
  --distributed-executor-backend mp \
  --enforce-eager \
  --additional-config '{
    "enable_flashcomm1": true,
    "ascend_compilation_config": {
      "enable_npugraph_ex": true,
      "enable_static_kernel": false
    },
    "scheduler_config": {
      "profiling_chunk_config": {"enabled": true},
      "short_request_first_config": {
        "enabled": true,
        "threshold": 66560,
        "long_max_wait_ms": 2000
      }
    },
    "multistream_overlap_shared_expert": true,
    "enable_mc2_hierarchy_comm": false,
    "enable_sparse_sfa_c8": true,
    "enable_sparse_li_c8": true,
    "enable_cpu_binding": true,
    "recompute_scheduler_enable": false
  }' \
  --speculative-config '{"num_speculative_tokens": 1, "method": "deepseek_mtp", "enforce_eager": true}' \
  --quantization ascend \
  --enable-expert-parallel \
  --safetensors-load-strategy prefetch \
  --kv-transfer-config '
  {"kv_connector": "MooncakeConnectorV2",
   "kv_role": "kv_producer",
   "kv_port": "30100",
   "engine_id": "0",
   "kv_connector_extra_config": {
       "prefill": {"dp_size": 1, "tp_size": 1, "pp_size": 2, "pcp_size": 8},
       "decode":  {"dp_size": 2, "tp_size": 8, "pp_size": 1}
   }}'
