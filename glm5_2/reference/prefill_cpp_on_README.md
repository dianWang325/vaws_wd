# GLM-5.2 1M：CPP 开启、前缀缓存关闭后的 prefill 测量

2026-09-17，在 `80.5.17.110`（API/head）和 `80.5.17.107`（worker）的 `wd_test0825` 容器运行。模型为 `/mnt/weight/GLM-5.2-W4A8C8-0713-MTP`，拓扑 `DP=1 / TP=16 / PP=2`，`max_model_len=1,024,000`，服务端模型 ID `glm-52`，API `http://80.5.17.110:9000`。测试结束时 `GET /v1/models` 仍返回 HTTP 200，服务保持运行。

先停止两端旧 PP2 进程并重新启动容器，再分别以 `head`、`worker` 参数运行 [`glm52_1m_dp1_tp16_pp2_rc1_cpp_on_17_110_107.sh`](../../../../plain_scripts/glm52_1m_dp1_tp16_pp2_rc1_cpp_on_17_110_107.sh)。脚本在原有 `--additional-config` JSON 中合并了 `"scheduler_config":{"profiling_chunk_config":{"enabled":true}}`，并添加 `--no-enable-prefix-caching`。启动日志出现 `[ProfilingChunk] Running startup profiling with real model forward`，引擎配置记录 `enable_prefix_caching=False`；健康接口返回 HTTP 200 后开始发送请求。

| 阶段  | 输入                                   |      输出 |  并发 |    成功 | TTFT 平均 / P50 / P95        |              输入吞吐 |       总耗时 |
| --- | ------------------------------------ | ------: | --: | ----: | -------------------------- | ----------------: | --------: |
| 预热  | 5 × 1,023,999 token                  | 1 token |   1 |   5/5 | 79.684 / 79.092 / 81.661 s | 12,849.71 token/s | 398.452 s |
| fix | 历史 `prefill_fixed`，24 × 65,536 token | 1 token |   1 | 24/24 | 4.362 / 4.357 / 4.394 s    | 15,019.97 token/s | 104.718 s |

两组数据均使用 [`backup_bench_prefill_glm52.py`](../scripts/backup_bench_prefill_glm52.py) 调用 `/v1/completions`，流式接收首个非空 token 测量 TTFT；吞吐为成功请求的服务端输入 token 总数除以阶段总耗时。服务端 `usage` 确认预热每条输入 1,023,999、输出 1 token，fix 每条输入 65,536、输出 1 token。输出为 1 token，因此 TPOT 不适用。预热先于 fix 运行，两个阶段均串行。

数据：[`prefill_warmup_1m.jsonl`](prefill_warmup_1m.jsonl) 为 5 条 1M 预热输入，使用 [`generate_warmup_1m_data.py`](../prefill_benchmark/generate_warmup_1m_data.py) 生成；[`prefill_fixed.jsonl`](prefill_fixed.jsonl) 为用户指定的历史 GLM-5.2 文件。SHA256 分别为 `f0df4faa44c8dc650288a62fab65f3978780b51f6047acbfd5b6e81baae63ebd` 和 `87efebbb3339444b4154cbfb16d0a87ef4215c599081acb413e647ff27b357db`。实际 tokenizer 长度可查 [`warmup_manifest.json`](warmup_manifest.json)、[`fix_manifest.json`](fix_manifest.json)。

原始逐条记录及完整汇总：[`warmup_requests.jsonl`](warmup_requests.jsonl)、[`warmup_summary.json`](warmup_summary.json)、[`fix_requests.jsonl`](fix_requests.jsonl)、[`fix_summary.json`](fix_summary.json)。启动及请求日志：[`head.log`](head.log)、[`worker.log`](worker.log)。第一次预热尝试误用了模型 ID `glm-5.2`，5 条请求均返回 HTTP 404，未进入推理；随即按 `/v1/models` 返回的 `glm-52` 重跑，上表与保存的逐条结果均来自成功的重跑。

