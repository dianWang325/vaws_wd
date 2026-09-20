# GLM-5.2 1M 历史达标实验复现材料

此目录保存 2026-09-17 在 vLLM `568afb3a13806beb53bb2e6bd518269357b237c0` 与 vllm-ascend `f2f74a16c3c50a76f4349d807918e83edec1e35c` 上得到低 TTFT 的历史测量记录和输入数据。当前仅保留非 CPP 启动脚本，并在该脚本中显式关闭前缀缓存；它与历史 CPP 测量的服务配置不同。`SHA256SUMS.txt` 记录当前归档文件的校验值；AISBench 脚本不在该清单中。此目录不是 Git 仓库。

## 历史配置

- GLM-5.2 W4A8C8 模型：`/mnt/weight/GLM-5.2-W4A8C8-0713-MTP`。
- 双节点 `DP1/TP16/PP2`；`max_model_len=1,024,000`、`max_num_batched_tokens=16,384`、PP 层划分 `41,37`、HCCL `AIV`、MTP 1 token、DSA-CP、CPP，前缀缓存关闭。
- 当前唯一启动脚本：[`scripts/glm52.sh`](scripts/glm52.sh)，不开启 CPP，显式关闭前缀缓存。其中节点地址是历史地址，不能直接用于新节点。下表指标来自历史 CPP 服务，不代表当前脚本的测量结果。
- 当前压测入口：[`scripts/run_aisbench.sh`](scripts/run_aisbench.sh)，调用 [aisbench_auto_tools_prefix](https://github.com/rayn-zzz/aisbench_auto_tools_prefix) 的生成器与 `aisbench_test.py`。五类测试各自运行；同输入规格的数据集存在时复用，否则由工具箱生成。历史自定义客户端 `backup_bench_prefill_glm52.py` 与 `backup_run_cpp_prefill.sh` 仅供回溯。

## 数据与历史指标

| 阶段 | 数据文件 | 并发 | 输出 | 历史 TTFT 均值 | 输入吞吐 |
| --- | --- | ---: | ---: | ---: | ---: |
| 预热 | [`data/prefill_warmup_1m.jsonl`](data/prefill_warmup_1m.jsonl)，5 条 | 1 | 1 | 79.684 s | 12,849.71 token/s |
| 定长 | [`data/prefill_fixed.jsonl`](data/prefill_fixed.jsonl)，24 条 × 65,536 | 1 | 1 | 4.362 s | 15,019.97 token/s |
| 变长 | [`data/prefill_variable.jsonl`](data/prefill_variable.jsonl)，24 条、43,490–77,411 token | 1 | 1 | 4.385 s | 15,143.60 token/s |
| 原始 PP2 基线 | [`data/prefill_fix_32k64.jsonl`](data/prefill_fix_32k64.jsonl)，64 条 × 32,768 | 4 | 1 | 6.351 s | 20,152.46 token/s |

定长和变长是**分别重启服务、各自完成 5 条长预热后**测得。两轮均关闭前缀缓存。基线是另一种缓存与并发口径，不用于和 CPP 两轮做直接加速比。详细历史结果见 [`reference/latency_and_heads_summary_20260917.md`](reference/latency_and_heads_summary_20260917.md)。

## 当前复现状态

仅完成本地复现材料归档，尚未在当前 vLLM `2cf0a6915ce544dc493a0990f2ea38d81601128a` / vllm-ascend `d6acae3b6ada1139704f98370cf8ec66bed1bdb8` 上启动或测量服务，因此上表均为历史结果，不能作为当前版本的复现成绩。当前版本需要先核对 SP/DP 约束；若必须把 DP 扩至 2，则拓扑与历史 DP1 不再相同，应另列实验结果，并保留一轮可行的 DP1 对照。远端运行不得占用主任务正在使用的节点或进程。

## 五类独立 AISBench 测试

| 阶段 | 每条输入 token | 每条输出 token | 请求数 | 默认并发 | 重点指标 |
| --- | ---: | ---: | ---: | ---: | --- |
| `warmup` | `MODEL_MAX_LEN - 1`，默认 204,799 | 1 | 5 | 1 | 成功数、TTFT |
| `prefill_fix` | 65,536 | 1 | 24 | 2 | TTFT、prefill 吞吐 |
| `prefill_variable` | 40,960～81,920，目标均值 65,536 | 1 | 24 | 1 | TTFT、prefill 吞吐 |
| `fix` | 65,536 | 2,560 | 24 | 1 | TTFT、TPOT、吞吐 |
| `variable` | 40,960～81,920，目标均值 65,536 | 2,560 | 24 | 1 | TTFT、TPOT、吞吐 |

每次命令仅运行指定阶段。当前 `warmup` 使用独立的 200K 输入数据集；历史 1M 预热数据保留供回溯。`prefill_fix` 与 `fix` 共用 64K 输入数据集，`prefill_variable` 与 `variable` 共用变长输入数据集。工具箱以 GSM8K 文本和模型 tokenizer 生成数据。定长数据集由工具箱自身按文件名判断是否复用；变长数据集由入口脚本按工具箱生成的文件名判断是否存在，缺失时调用工具箱的 `create_multi_prefix_dataset` 生成，再通过工具箱支持的 `--dataset` 参数压测。这一步是为了避开工具箱当前变长分支误用定长文件名判断存在性的行为。变长采用工具箱的截断高斯参数：目标均值 65,536、标准差 16,384、下限 40,960、上限 81,920；24 条的实际算术均值及端点覆盖不保证精确值。前缀缓存由服务启动脚本控制。

工具箱位于本目录内的 `aisbench_auto_tools_prefix/`；也可用 `AISBENCH_TOOL_DIR` 指向其他位置。先按[工具箱 README](https://github.com/rayn-zzz/aisbench_auto_tools_prefix#%E4%B8%80%E4%BF%AE%E6%94%B9configpy)准备 AISBench 环境，并设置工具箱 `config.py` 中的 `DATASET_PATH`、`WORK_PATH`、`MODEL_NAME`、`MODEL_PATH`、`HOST_IP`、`HOST_PORT` 与 `OUTPUT_DIR`。待服务健康后，分别执行：

```bash
bash scripts/run_aisbench.sh warmup
bash scripts/run_aisbench.sh prefill_fix
bash scripts/run_aisbench.sh prefill_variable
bash scripts/run_aisbench.sh fix
bash scripts/run_aisbench.sh variable
```

`MODEL_MAX_LEN` 默认 204,800，需与服务的 `--max-model-len` 保持一致；`prefill_fix` 默认并发 2，其他非 warmup 阶段默认并发 1，均可通过 `CONCURRENCY` 覆盖；`warmup` 固定并发 1。工具箱的 `aisbench.log`、`aisbench_all.log`、`aisbench_result.csv` 和 `OUTPUT_DIR` 保存测试结果。工具箱默认使用流式 Chat API、`temperature=0`、`ignore_eos=True`，并在 AISBench 命令中指定 `--num-warmups 0`；若所装 AISBench 版本不支持该选项，参照工具箱 FAQ 处理。

工具箱的性能结果包含 TTFT、TPOT 和吞吐等指标。输出 1 token 的阶段不看 TPOT；输出 2,560 token 的阶段还看 TPOT。指标由工具箱调用的 AISBench 计算，阶段参数不单独开关指标。

参考：[AISBench 自定义数据集文档](https://ais-bench-benchmark-rf.readthedocs.io/en/latest/advanced_tutorials/custom_dataset.html)、[官方自定义配置示例](https://github.com/AISBench/benchmark/blob/master/ais_bench/configs/api_examples/perf_vllm_api_custom_dataset.py)、[官方流式 vLLM 模型配置](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_stream.py)。
