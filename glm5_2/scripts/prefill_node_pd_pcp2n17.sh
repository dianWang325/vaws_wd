#!/usr/bin/env bash
# PD 分离部署 - Prefill 节点 PCP 双机版（17 网段，TCP 跨机，MooncakeConnectorV2）
# 目标机：head=80.5.17.109 / worker=80.5.17.119（容器 wd_test0921 内运行，/home 已挂载进容器）
# 依赖分支：test/cpp_async_mtp_pcp0924 @ 4ec50e551（双机已 rsync 对齐 109 工作树）
# 拓扑：PP2 x TP4 x PCP4（2 台 16 卡，PP stage 按机分界，TP4xPCP4 在单机内）
# 背景：单机 PCP4TP2 在 87k/92k 超长 prefill 时 MoE token_dispatcher HcclBroadcast HBM OOM，
#   扩双机 TP8 后每卡权重约 4.5GiB，显存压力解除
# P 节点保留 CPP(profiling_chunk)+SRF(short_request_first)
# Run as: bash scripts/prefill_node_pd_pcp2n17.sh head|worker
# 日志：glm5_2/logs/glm52_pd_pcp2n17_prefill_{head,worker}_<timestamp>.log（时间戳命名，不覆盖历史）
set -euo pipefail

role="${1:?expected head or worker}"
case "$role" in
  head)
    local_ip=80.5.17.109
    node_rank=0
    server_role_args=(--api-server-count 1)
    ;;
  worker)
    local_ip=80.5.17.119
    node_rank=1
    server_role_args=(--headless)
    ;;
  *) echo "invalid role: $role" >&2; exit 2 ;;
esac

nic_name="enp48s3u1u1"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
log_dir="${script_dir}/../logs"
mkdir -p "$log_dir"
exec > "${log_dir}/glm52_pd_pcp2n17_prefill_${role}_$(date +%Y%m%d_%H%M%S).log" 2>&1

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
  --max-model-len 92160 \
  --max-num-batched-tokens 20480 \
  --gpu-memory-utilization 0.75 \
  "${server_role_args[@]}" \
  --max-num-seqs 8 \
  --no-enable-prefix-caching \
  --pipeline-parallel-size 2 \
  --tensor-parallel-size 4 \
  --prefill-context-parallel-size 4 \
  --cp-kv-cache-interleave-size 128 \
  --enable-chunked-prefill \
  --async-scheduling \
  --distributed-executor-backend mp \
  --nnodes 2 \
  --node-rank "$node_rank" \
  --master-addr 80.5.17.109 \
  --master-port 7060 \
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
       "prefill": {"dp_size": 1, "tp_size": 4, "pp_size": 2, "pcp_size": 4},
       "decode":  {"dp_size": 2, "tp_size": 8, "pp_size": 1}
   }}'
