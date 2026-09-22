# DSV4-Flash PD 分离部署实验记录（2026-09-22）

## 部署拓扑

| 角色 | 机器 / 容器 | 端口 | 并行配置 |
|---|---|---|---|
| Prefill | 80.5.9.113 / wd_test0921 | 18080 | PP2 x TP8（16 芯片）|
| Decode  | 80.5.9.147 / wd_test0921 | 18080 | DP2 x TP8（16 芯片，DP RPC 12321）|
| Proxy   | 80.5.9.113 / wd_test0921 | 9000  | dp_load_balance_proxy_server.py |

- 模型：`/mnt/weight/DeepSeek-V4-Flash-DSpark-w4a8-scope-ci`（w4a8 + DSpark 投机解码）
- vllm：`0.28.1rc1.dev676+g84030bbe3`（源码 `/home/w00985415/proj_0921/deps/vllm`）
- vllm-ascend：`test/cpp` 分支（`/home/w00985415/proj_0921/deps/vllm-ascend`）
- KV 传输：MooncakeConnectorV2（TCP），kv_port prefill=30100 / decode=30200
- 已知环境限制：80.5.9.x 段 RoCE 参数面 link DOWN（交换机侧），本方案跨机通信全部走 TCP，不受影响；节点内 HCCL 走 HCCS 正常

## 相对原始脚本的对齐改动（仅限路径/网卡/IP/端口/采样参数）

- 模型路径 `/mnt/share/weight/` → `/mnt/weight/`（含 config.py 中 MODEL_PATH，原先误配为 w8a8-mtp）
- 业务网卡 `enp48s3u1u2` → `enp194s0f0`（两台一致，host 网络模式）
- IP：prefill 80.5.9.113，decode 80.5.9.147（含 --data-parallel-address、proxy prefiller/decoder hosts）
- config.py：HOST_IP=80.5.9.113，HOST_PORT=9000（指向 proxy）
- aisbench_test.py 精度模式采样参数：`temperature=1.0, top_p=0.95`（对齐模型权重 generation_config.json）

## 关键事件时间线

1. 初次选机 80.5.9.143 + 80.5.9.147（当时全段仅这两台全空）
2. 143 prefill 首启崩溃：vllm-ascend 校验 `profiling_chunk_config requires pipeline parallelism (pp>1)`，dspark 草稿模型内部重建配置 pp=1 触发 → 按要求切 vllm-ascend 到 `test/cpp` 分支解决（校验逻辑在 vllm-ascend 的 ascend_config.py / platform.py）
3. 143 于 20:39 被其他 kubepods pod 抢占 4 卡（VLLMWorker_DP，非本任务）→ prefill 迁移
4. 80.5.9.139 容器 vllm 为 site-packages 0.23.0 安装版（与 147 源码版 0.28.1rc1 不一致，不识别 `--attention_config.indexer_kv_dtype`）→ 弃用
5. 最终 113（prefill+proxy）+ 147（decode），21:14 起正常拉起

## 测试结果

- Warmup（5 条 x 204799 输入 tokens，经 proxy 全链路）：FAIL=0，TTFT avg 47.5s，prefill 吞吐约 4308 tok/s
- 性能压测（fix / variable 多轮，64K 输入）：见 `results/perf_summary_all.csv`
- 精度（GSM8K 200 条子集，`temperature=1.0, top_p=0.95`，并发 64）：

| 轮次 | 接口 | accuracy | 说明 |
|---|---|---|---|
| try1 | completions | 13.5% | 输出被 chat 格式碎片污染，判分提取失败 |
| try2 | completions | 10.5% | 同上（人工抽查推理过程正确）|
| try3 | chat | **93.0%** | chat template 包装后输出规范，为真实精度水平 |

教训：dsv4 是 chat 微调模型，裸 completions + `{question}` 直发会产出对话格式碎片（`"role": "assistant"` 等），gsm8k 判分（取最后一个数字）大面积失效；精度评测必须走 chat 接口。

## 文件清单

- `service_logs/`：prefill / decode / proxy 服务日志（成功的 r2 轮次）
- `test_logs/`：warmup、fix（定长输出）、variable（变长）、acc 各轮测试日志
- `results/`：性能汇总 csv + 三轮精度 summary csv

注：首次 143 崩溃日志（prefill_0922.log）保留在 80.5.9.143 的 logs/ 下，未纳入本目录。
