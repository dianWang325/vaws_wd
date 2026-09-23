#!/usr/bin/env bash
# PD 分离部署 - Prefill 节点（TCP 跨机，MooncakeConnectorV2）
# 目标机：80.5.17.110（容器内运行，/home 已挂载进容器）
# 并行方式与 DSV4 PD 一致：PP2 x TP8，单机 16 卡，kv_producer
# 与 glm52_dsacp_V2.sh(head) 的差异：去掉跨机 --nnodes/--node-rank/--master-*（PP 收敛到单机内）
#   + 新增 ASCEND_CONNECT_TIMEOUT/ASCEND_TRANSFER_TIMEOUT + 新增 --kv-transfer-config
#   + 服务端口改为 18080（9000 留给 proxy）
#   + DSA-CP 暂无法与 PD 组合（待后续适配），相关参数已关闭（见 vllm serve 上方说明）
set -euo pipefail

nic_name="enp48s3u1u1"  # 80.5.17.110 实测业务网卡
local_ip="80.5.17.110"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
log_dir="${script_dir}/../logs"
mkdir -p "$log_dir"
exec > "${log_dir}/glm52_pd_prefill.log" 2>&1

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
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TASK_QUEUE_ENABLE=1
export VLLM_ENGINE_READY_TIMEOUT_S=100000
export VLLM_PP_LAYER_PARTITION=38,40

# DSA-CP 暂无法与 PD 组合（待后续适配），以下开关暂时关闭，适配完成后恢复：
#   vllm 参数：--prefill-context-parallel-size 1 --decode-context-parallel-size 8 --cp-kv-cache-interleave-size 128
#   --additional-config 内："enable_dsa_cp": true
vllm serve /mnt/weight/GLM-5.2-W4A8C8-0713-MTP \
  --seed 1024 \
  --host 0.0.0.0 \
  --port 18080 \
  --served-model-name glm-52 \
  --max-model-len 204800 \
  --max-num-batched-tokens 20480 \
  --gpu-memory-utilization 0.75 \
  --api-server-count 1 \
  --max-num-seqs 8 \
  --no-enable-prefix-caching \
  --pipeline-parallel-size 2 \
  --tensor-parallel-size 8 \
  --distributed-executor-backend mp \
  --enforce-eager \
  --additional-config '{
    "enable_flashcomm1": true,
    "ascend_compilation_config": {
      "enable_npugraph_ex": true,
      "enable_static_kernel": false
    },
    "scheduler_config": {"profiling_chunk_config": {"enabled": true}},
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
       "prefill": {"dp_size": 1, "tp_size": 8, "pp_size": 2},
       "decode":  {"dp_size": 2, "tp_size": 8, "pp_size": 1}
   }}'
