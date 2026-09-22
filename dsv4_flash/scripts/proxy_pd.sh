# PD 分离部署 - Proxy（可运行在任一能连通 119/120 的机器上，也可放在 119 的 wd_test0921 容器里）
# 使用 proj_0825 仓库自带的 vllm-ascend 官方 proxy 示例（透传 kv_transfer_params 做 P/D 握手）
# 客户端统一访问本 proxy 的 9000 端口
python3 /home/w00985415/proj_0825/deps/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
  --host 0.0.0.0 \
  --port 9000 \
  --prefiller-hosts 80.48.33.119 \
  --prefiller-ports 18080 \
  --decoder-hosts 80.48.33.120 \
  --decoder-ports 18080
