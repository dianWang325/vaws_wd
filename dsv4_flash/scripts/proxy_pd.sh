# PD 分离部署 - Proxy（可运行在任一能连通 119/120 的机器上，也可放在 119 的 wd_test0921 容器里）
# 使用与本脚本同目录的 dp_load_balance_proxy_server.py（透传 kv_transfer_params 做 P/D 握手）
# 注意：不要写 vllm-ascend 仓库内的绝对路径，不同机器/容器上该路径不一致
# 客户端统一访问本 proxy 的 9000 端口
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$SCRIPT_DIR/dp_load_balance_proxy_server.py" \
  --host 0.0.0.0 \
  --port 9000 \
  --prefiller-hosts 80.5.9.138 \
  --prefiller-ports 18080 \
  --decoder-hosts 80.5.9.143 \
  --decoder-ports 18080
