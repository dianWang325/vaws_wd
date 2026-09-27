# GLM-5.2 P/D 启动脚本说明

本目录脚本用于部署 GLM-5.2 W4A8C8 MTP 的 Prefill/Decode 分离服务。除特别说明外，P 节点负责生成 KV Cache（`kv_producer`），D 节点负责接收 KV Cache 并继续解码（`kv_consumer`）。

## Decode 节点

| 脚本 | 执行模式 | D 节点拓扑 | 最大模型长度 | 配套 Prefill 拓扑 | 说明 |
| --- | --- | --- | ---: | --- | --- |
| `decode_node_pd.sh` | Eager（`--enforce-eager`） | DP2 × TP8 | 204800 | PP2 × TP8 | 基准 Eager Decode 方案 |
| `decode_node_pd_graph.sh` | `FULL_DECODE_ONLY`；MTP 仍为 Eager | DP2 × TP8 | 92160 | PP2 × TP2 × PCP4 | 使用 Decode 图降低 TPOT；D 节点本身不开 PCP/DCP/CPP/SRF |

两者的主要区别是主模型 Decode 是否启用图模式，但并不只有这一项差异：目标机器/IP、最大模型长度、日志命名以及 `kv-transfer-config` 中声明的 Prefill 拓扑也不同。使用时必须与对应的 Prefill 脚本保持拓扑一致。

## Prefill 节点

这些脚本均启用 Eager、Chunked Prefill、异步调度、CPP（Profiling Chunk）、SRF、MTP 和 Mooncake KV Producer；主要区别是节点数量及 PP/TP/PCP 划分。

| 脚本 | 节点数 | Prefill 拓扑 | 总卡数 | 最大模型长度 | Max batched tokens | 显存比例 | 主要用途 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| `prefill_node_pd.sh` | 1 | PP2 × TP8，无 PCP | 16 | 204800 | 20480 | 0.75 | 不使用 PCP 的基准方案 |
| `prefill_node_pd_pcp.sh` | 1 | PP2 × TP4 × PCP2 | 16 | 204800 | 20480 | 0.85 | 单机 PCP2 |
| `prefill_node_pd_pcp4tp2.sh` | 1 | PP2 × TP2 × PCP4 | 16 | 87040 | 20480 | 0.85 | 单机 PCP4；在 PCP 并行度与 TP 权重分片之间折中 |
| `prefill_node_pd_pcp8.sh` | 1 | PP2 × TP1 × PCP8 | 16 | 92160 | 12288 | 0.84 | 单机最大化 PCP；TP1 的权重和通信显存压力更高 |
| `prefill_node_pd_pcp_2node.sh` | 2 | PP2 × TP8 × PCP2 | 32 | 204800 | 20480 | 0.75 | 9 网段双机，使用 TP8 降低单卡权重占用 |
| `prefill_node_pd_pcp2n17.sh` | 2 | PP2 × TP4 × PCP4 | 32 | 92160 | 20480 | 0.75 | 17 网段双机，兼顾 PCP 并行与显存压力 |

直观选择：

- 不需要 PCP：使用 `prefill_node_pd.sh`。
- 单机常规 PCP：使用 `prefill_node_pd_pcp.sh`。
- 单机提高 PCP 并行度：使用 `prefill_node_pd_pcp4tp2.sh`；`prefill_node_pd_pcp8.sh` 更激进。
- 双机且优先降低权重显存：使用 `prefill_node_pd_pcp_2node.sh`。
- 双机且希望保持 PCP4：使用 `prefill_node_pd_pcp2n17.sh`。

## 使用注意事项

1. P、D 两端 `kv_connector_extra_config` 中的 `dp_size`、`tp_size`、`pp_size` 和 `pcp_size` 必须完全对应。
2. 双机脚本需分别以 `head` 和 `worker` 参数启动，并确保 `master-addr`、网卡名和节点 IP 与现场环境一致。
3. 当前 Prefill 脚本全部使用 `--enforce-eager`；`decode_node_pd_graph.sh` 仅为主模型 Decode 开启 `FULL_DECODE_ONLY`。
4. 切换拓扑时需同步检查最大模型长度、批处理 token 上限和显存比例，不能只修改 PCP/TP 参数。
