python3 /home/w00985415/proj_0825/deps/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
  --host 0.0.0.0 \
  --port 9000 \
  --prefiller-hosts 192.168.1.10 \
  --prefiller-ports 8100 \
  --decoder-hosts 192.168.1.11 \
  --decoder-ports 8200