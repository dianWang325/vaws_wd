# GLM-5.2 1M 部署、时延与源码 HEAD 汇总

统计范围：本目录下 2026-09-16 至 17 日的双节点 1M 部署与 prefill 实验。模型均为 `/mnt/weight/GLM-5.2-W4A8C8-0713-MTP`，服务端 `max_model_len=1,024,000`。表中的 TTFT 是客户端从发送请求到收到首个非空文本流块的时间；对于生成 token 解码为空的 variable 请求，采用携带 `finish_reason` 的完成事件计时。输入吞吐为成功请求的服务端输入 token 总数除以该阶段的墙钟耗时。

## 源码与部署脚本

| 实验组 | 节点（head / worker） | 部署脚本 | vLLM HEAD | vllm-ascend HEAD | 说明 |
| --- | --- | --- | --- | --- | --- |
| 原文 DP4/TP8 验证 | `80.5.17.110` / `80.5.17.107` | [110](../../../plain_scripts/glm52_1m_dp4_tp8_rc1_17_110.sh)、[107](../../../plain_scripts/glm52_1m_dp4_tp8_rc1_17_107.sh)；[原文完整脚本](../../../plain_scripts/glm52_1m_dp4_tp8_doc_5_2_2.sh) | `568afb3a13806beb53bb2e6bd518269357b237c0`（tag `v0.26.0`） | `f2f74a16c3c50a76f4349d807918e83edec1e35c`（tag `v0.26.0rc1`） | 旧镜像 `v0.26.0rc1-a3`；服务日志显示 vLLM `0.26.0`。 |
| DP1/TP16/PP2 原始配置及 prefill 基线 | `80.5.17.110` / `80.5.17.107` | [PP2 脚本](../../../plain_scripts/glm52_1m_dp1_tp16_pp2_rc1_17_110_107.sh) | 同上 | 同上 | `VLLM_PP_LAYER_PARTITION=41,37`，前缀缓存为默认配置。 |
| DP1/TP16/PP2，CPP 开启，fix 与 variable | `80.5.17.110` / `80.5.17.107` | [CPP 脚本](../../../plain_scripts/glm52_1m_dp1_tp16_pp2_rc1_cpp_on_17_110_107.sh) | 同上 | 同上 | `profiling_chunk_config.enabled=true`，关闭前缀缓存，PP `41,37`。 |
| DP1/TP16/PP2，CPP+SRF，仅单条成功预热 | `80.5.17.111` / `80.5.17.109` | [CPP+SRF 脚本](../../../plain_scripts/glm52_1m_dp1_tp16_pp2_cpp_srf_17_111_109.sh) | `a97dacb7106ee49f39f3d1fc6ae1800ff724e01d` | `660c4582aa580ce98edc9b681bb6ff6d03153575` | CPP+SRF 合并配置、关闭前缀缓存、禁用异步调度、PP `42,36`。 |

**HEAD 证据。**旧容器 `110` 和 `107` 的 Git reflog 均显示：vLLM 从 `568afb3a...` 切到 `b2f68583...`、vllm-ascend 从 `f2f74a16...` 切到 `657f927b...`，均发生在上述旧版服务日志和完整测量之后；旧版服务日志也显示 vLLM `v0.26.0`。这两个旧 HEAD 分别指向本地 tag `v0.26.0` 和 `v0.26.0rc1`。较新实验前直接在 `111/109` 容器核对了 `git rev-parse HEAD`、`pip show` 和 Ascend 的 `.github/vllm-main-verified.commit`：vLLM 包版本 `0.28.1rc1.dev570+ga97dacb71.empty`，vllm-ascend 包版本 `0.19.1rc2.dev2103+g660c4582a`，该 verified 文件正是 `a97dacb...`。109 的 Ascend 仓库另有一处既存未提交的 Mooncake KV 传输接收路径修改；本次配置没有启用 KV 传输，也没有更动该文件。110 后来已升级到较新 HEAD，**当前 HEAD 不能代替旧实验当时的 HEAD**。`657f927b...` 与 `b2f68583...` 为中间升级状态，本目录没有该组合下完成的时延测试。

## 成功请求的时延与吞吐

以下各 prefill 阶段均输出 1 token。`默认缓存`阶段存在前缀复用；`独立 salt` 阶段用每请求不同的 `cache_salt` 阻断跨请求复用；CPP 阶段直接关闭前缀缓存。不同输入分布、并发和缓存口径的行不应作为同条件加速比。

| 配置 / 阶段 | 输入与并发 | 成功 | TTFT 均值 / P50 / P95（秒） | 输入吞吐（token/s） | 阶段耗时（秒） |
| --- | --- | ---: | ---: | ---: | ---: |
| 原始 PP2，warmup，默认缓存 | 5 条，85,937–87,039 token，并发 1 | 5/5 | 6.466 / 4.310 / 12.932 | 13,422.57 | 32.341 |
| 原始 PP2，历史 fixed 65K，默认缓存 | 24 × 65,536 token，并发 4 | 24/24 | 2.556 / 2.036 / 5.519 | 100,065.06 | 15.718 |
| 原始 PP2，warmup，独立 salt | 同上 warmup，并发 1 | 5/5 | 4.276 / 4.282 / 4.367 | 20,294.79 | 21.389 |
| 原始 PP2，历史 fixed 65K，独立 salt | 24 × 65,536 token，并发 4 | 24/24 | 12.532 / 13.195 / 13.225 | 19,626.01 | 80.142 |
| 原始 PP2，正式 fix 32K，独立 salt | 64 × 32,768 token，并发 4 | 64/64 | 6.351 / 6.445 / 6.478 | 20,152.46 | 104.064 |
| CPP，1M 预热，fix 轮 | 5 × 1,023,999 token，并发 1 | 5/5 | 79.684 / 79.092 / 81.661 | 12,849.71 | 398.452 |
| CPP，历史 fixed 65K | 24 × 65,536 token，并发 1 | 24/24 | 4.362 / 4.357 / 4.394 | 15,019.97 | 104.718 |
| CPP，1M 预热，variable 轮 | 5 × 1,023,999 token，并发 1 | 5/5 | 79.535 / 78.684 / 81.664 | 12,873.63 | 397.712 |
| CPP，variable 正式复测 | 24 条，43,490–77,411 token，并发 1 | 24/24 | 4.385 / 4.411 / 4.979 | 15,143.60 | 105.261 |
| CPP+SRF，1M 预热（用户中止） | 1,023,999 token，并发 1 | 1/5 | **1813.254** / — / — | — | — |

原始 PP2 基线逐条记录和汇总见[基线测量](prefill_benchmark/README.md)。CPP fix 轮见[CPP fix 结果](prefill_cpp_on_20260917/README.md)，CPP variable 轮见[CPP variable 结果](prefill_cpp_variable_20260917/README.md)。CPP+SRF 完成的单条预热记录见[部分结果](prefill_cpp_srf_variable_20260917/warmup_partial_111_109_requests.jsonl)；第 2 条预热中断，variable 阶段未开始，因此不能报告完整阶段的 TTFT 分布或吞吐。此单条 TTFT 比旧版 CPP 的约 79.5 秒高很多，但源码、PP 划分和调度配置同时改变，不能把差值单独归因于 SRF。

另有两次短文本部署验收：原文 DP4/TP8 返回 16 token 的完整响应耗时 **1.456 秒**；后续 DP1/TP16/PP2 同类响应耗时 **23.428 秒**。它们是单次端到端响应时延，输入短、输出 16 token，**不是 TTFT，也不是 1M prefill 测量**；原始响应和日志见[部署复现记录](README.md)。

## 当前状态

按用户要求，`111/109` 上本轮服务及评测客户端已停止，NPU 显存回到空闲基线；自动续跑任务已暂停。现有文件和旧版结果均保留。
