# GLM-5.2 1M：CPP 开启、前缀缓存关闭后的 variable prefill 测量

2026-09-17，在 `80.5.17.110`（API/head）和 `80.5.17.107`（worker）的 `wd_test0825` 容器上，结束上一轮服务后重新启动 `DP=1 / TP=16 / PP=2`、`max_model_len=1,024,000` 的 GLM-5.2 服务。两端使用同一份 [`CPP 启动脚本`](../../../../plain_scripts/glm52_1m_dp1_tp16_pp2_rc1_cpp_on_17_110_107.sh)：`--additional-config` 中保留原配置并启用 `scheduler_config.profiling_chunk_config.enabled=true`，同时使用 `--no-enable-prefix-caching`。新启动日志确认 `[ProfilingChunk] Running startup profiling with real model forward` 和 `enable_prefix_caching=False`。`GET /v1/models` 返回 HTTP 200 后才开始预热；测试结束后再次返回 HTTP 200，服务仍在运行。

| 阶段 | 输入 | 输出 | 并发 | 成功 | TTFT 平均 / P50 / P95 | 输入吞吐 | 总耗时 |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| 预热 | 5 × 1,023,999 token | 1 token | 1 | 5/5 | 79.535 / 78.684 / 81.664 s | 12,873.63 token/s | 397.712 s |
| variable 正式复测 | 24 条，43,490–77,411 token；平均 66,418 token | 1 token | 1 | 24/24 | 4.385 / 4.411 / 4.979 s | 15,143.60 token/s | 105.261 s |

variable 输入来自仓库历史 `aisbench_workspace/datasets/formal/variable_long-8f489c0a2f4a-6658b459dfe1.jsonl`，此处保存为 [`prefill_variable.jsonl`](prefill_variable.jsonl)，SHA256 为 `ccfe4a0730eca27895124dd72d7647655829ad5e41c2e5db2b82dd14a0247965`。配置分布为 40,960–81,920 token；使用当前 GLM-5.2 tokenizer 实测 24 条为 43,490–77,411 token，总计 1,594,025 token，全部未截断。预热沿用上一轮的 [`prefill_warmup_1m.jsonl`](../prefill_cpp_on_20260917/prefill_warmup_1m.jsonl)，每条服务端 `usage` 均为 1,023,999 输入、1 输出。

客户端使用 [`backup_bench_prefill_glm52.py`](../scripts/backup_bench_prefill_glm52.py) 调用 `/v1/completions`，设置 `model=glm-52`、`max_tokens=1`、`temperature=0`、`ignore_eos=true`、`stream=true`。先以并发 1 完成全部 5 条预热，再以并发 1 运行 variable。输入吞吐为成功请求的服务端 `prompt_tokens` 总数除以该阶段墙钟耗时；输出只有 1 token，TPOT 不适用。

首次 variable 测量 24/24 成功，但两条请求的唯一生成 token 解码为空文本，旧客户端只对非空文本块计时，导致这两条 TTFT 为空。首次结果保留在 [`variable_initial_requests.jsonl`](variable_initial_requests.jsonl) 和 [`variable_initial_summary.json`](variable_initial_summary.json)。随后更新客户端：若该 1-token 输出为空文本，就以携带 `finish_reason` 的流式事件到达时间计 TTFT，并在逐条记录的 `ttft_event` 标为 `finish_event_empty_text`。同一服务、同一数据、同一并发再次运行 variable，上表采用复测结果；复测 24 条 TTFT 均有值，其中 3 条使用完成事件计时。

原始结果与配置：[`warmup_requests.jsonl`](warmup_requests.jsonl)、[`warmup_summary.json`](warmup_summary.json)、[`warmup_manifest.json`](warmup_manifest.json)，以及 [`variable_requests.jsonl`](variable_requests.jsonl)、[`variable_summary.json`](variable_summary.json)、[`variable_manifest.json`](variable_manifest.json)。完整服务日志：[`head.log`](head.log)、[`worker.log`](worker.log)。
