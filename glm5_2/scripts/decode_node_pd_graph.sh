#!/usr/bin/env bash
# PD 分离部署 - Decode 节点图模式版（TCP 跨机，MooncakeConnectorV2）
# 目标机：80.5.9.141（容器 wd_test0921 内运行，/home 已挂载进容器）
# 依赖分支：test/cpp_async_mtp_pcp0924
# 与 decode_node_pd.sh 的差异：
#   + 去掉 --enforce-eager，开启 FULL_DECODE_ONLY 图模式压 TPOT（对齐 DSV4 decode 做法；
#     PR #7851 确认 PCP/DCP 不支持 full graph，D 节点不开 PCP/DCP，不受影响）
#   + D 节点不开 CPP(profiling_chunk)/SRF(short_request_first)/PCP/DCP（约束保持不变）
# 日志：glm5_2/logs/glm52_pd_graph_decode.log
set -euo pipefail

nic_name="enp194s0f0"  # 80.5.9.141 实测业务网卡
local_ip="80.5.9.141"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
log_dir="${script_dir}/../logs"
mkdir -p "$log_dir"
exec > "${log_dir}/glm52_pd_graph_decode.log" 2>&1

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

vllm serve /mnt/weight/GLM-5.2-W4A8C8-0713-MTP \
  --seed 1024 \
  --host 0.0.0.0 \
  --port 18080 \
  --served-model-name glm-52 \
  --max-model-len 114688 \
  --max-num-batched-tokens 20480 \
  --gpu-memory-utilization 0.85 \
  --api-server-count 1 \
  --max-num-seqs 8 \
  --no-enable-prefix-caching \
  --data-parallel-size 2 \
  --enable-chunked-prefill \
  --async-scheduling \
  --data-parallel-address 80.5.9.141 \
  --data-parallel-rpc-port 16591 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --additional-config '{
    "enable_flashcomm1": true,
    "ascend_compilation_config": {
      "enable_npugraph_ex": true,
      "enable_static_kernel": false
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
  --safetensors-load-strategy prefetch \
  --kv-transfer-config '
  {"kv_connector": "MooncakeConnectorV2",
   "kv_role": "kv_consumer",
   "kv_port": "30200",
   "engine_id": "1",
   "kv_connector_extra_config": {
       "prefill": {"dp_size": 1, "tp_size": 1, "pp_size": 2, "pcp_size": 8},
       "decode":  {"dp_size": 2, "tp_size": 8, "pp_size": 1}
   }}'
