set -eo pipefail
export ASCEND_LOCAL_COMM_RES='{"version":"1.2"}'
export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["hccs:device"]}'
nic_name="data0.172"
local_ip="172.27.5.129"
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
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:${LD_PRELOAD:-}
export HCCL_BUFFSIZE=200
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096
export VLLM_USE_V2_MODEL_RUNNER=1


if [[ -z "${SUFFIX_ADDITIONAL_CONFIG:-}" ]]; then
    SUFFIX_ADDITIONAL_CONFIG='{"enable_cpu_binding":true,"multistream_overlap_shared_expert":true}'
fi

connector='MooncakeConnectorV2'
parallel='{"prefill":{"dp_size":1,"tp_size":4,"pp_size":1,"pcp_size":2},"decode":{"dp_size":1,"tp_size":4,"pp_size":1}}'
case "$PD_ROLE" in
  P) tp=4; pcp=2; port=18766; kvport=30700; role=kv_producer; max_seqs=16; budget=8192; expert=(--enable-expert-parallel); mode=(--enforce-eager); spec=() ;;
  D) tp=4; pcp=1; port=18767; kvport=30800; role=kv_consumer; max_seqs=32; budget=512; expert=()
     mode=(--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":288}')
     spec=(--speculative-config "$SUFFIX_SPEC") ;;
  *) exit 2 ;;
esac
printf -v kv '{"kv_connector":"%s","kv_role":"%s","kv_port":"%s","engine_id":"gpqa-%s","kv_connector_extra_config":%s}' "$connector" "$role" "$kvport" "$PD_ROLE" "$parallel"

exec python "$PD_SCRIPTS/dsv4flash_pd_runtime.py" --model /mnt/weight/DeepSeek-V4-Flash-w4a8 \
    --host 127.0.0.1 \
    --max-model-len 24576 \
    --max-num-batched-tokens "$budget" \
    --served-model-name suffix-ab \
    --gpu-memory-utilization 0.85 \
    --api-server-count 1 \
    --max-num-seqs "$max_seqs" \
    --pipeline-parallel-size 1 \
    --tensor-parallel-size "$tp" \
    --prefill-context-parallel-size "$pcp" \
    "${expert[@]}" \
    --enable-chunked-prefill \
    --no-async-scheduling \
    "${mode[@]}" \
    --tokenizer-mode deepseek_v4 \
    --quantization ascend \
    --port "$port" \
    --no-enable-prefix-caching \
    "${spec[@]}" \
    --no-disable-hybrid-kv-cache-manager \
    --kv-transfer-config "$kv" \
    --block-size 32 \
    --attention_config.indexer_kv_dtype int8 \
    --distributed-executor-backend mp \
    --reasoning-parser deepseek_v4 \
    --additional-config "$SUFFIX_ADDITIONAL_CONFIG"
