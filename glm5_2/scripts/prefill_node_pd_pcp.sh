#!/usr/bin/env bash
# PD 分离部署 - Prefill 节点 PCP 版（TCP 跨机，MooncakeConnectorV2）
# 目标机：80.5.9.138（容器 wd_test0921 内运行，/home 已挂载进容器）
# 依赖分支：test/cpp_async_mtp_pcp0924（rebase 到 upstream 99cef8c，PR #17355 Mooncake V2 支持 PCP）
# 与 prefill_node_pd.sh 的差异：
#   + 开启原生 PCP：--prefill-context-parallel-size 2，拓扑 PP2 x TP4 x PCP2（单机 16 卡）
#   + 按 PR #16846 口径：prefill 走原生 PCP，不开 enable_dsa_cp（DSA-CP 暂不支持 P 节点）
#   + TP8->TP4 后单卡权重翻倍，gpu-memory-utilization 0.75->0.85
#   + P 节点保留 CPP(profiling_chunk)+SRF(short_request_first)，D 节点不开
# 日志：glm5_2/logs/glm52_pd_pcp_prefill.log
set -euo pipefail

nic_name="enp194s0f0"  # 80.5.9.138 实测业务网卡
local_ip="80.5.9.138"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
log_dir="${script_dir}/../logs"
mkdir -p "$log_dir"
exec > "${log_dir}/glm52_pd_pcp_prefill.log" 2>&1

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
export HCCL_EXEC_TIMEOUT=204
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
  --max-model-len 204800 \
  --max-num-batched-tokens 20480 \
  --gpu-memory-utilization 0.85 \
  --api-server-count 1 \
  --max-num-seqs 8 \
  --no-enable-prefix-caching \
  --pipeline-parallel-size 2 \
  --tensor-parallel-size 4 \
  --prefill-context-parallel-size 2 \
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
       "prefill": {"dp_size": 1, "tp_size": 4, "pp_size": 2, "pcp_size": 2},
       "decode":  {"dp_size": 2, "tp_size": 8, "pp_size": 1}
   }}'
