# A3 服务拉起脚本 / A3 service launchers

依据 `D:\experiment` 中有成功请求记录的实验整理，采用直接设置环境变量、直接执行 `vllm serve` 的部署方式。参考 [vLLM Ascend 模型部署文档](https://docs.vllm.ai/projects/ascend/en/latest/getting_started/overview.html)的编写结构，具体参数仍来自归档成功实验，不以最新文档替换历史验证配置。

## 文件与整理原则

| 入口 | 配置来源与用途 |
| --- | --- |
| `qwen3_p.sh` | 原样保留 Qwen3-30B-A3B P 脚本，TP2/PP2，设备 0–3，端口 13700 |
| `qwen3_d.sh` | 原样保留 Qwen3-30B-A3B D 脚本，DP2/TP2，设备 4–7，端口 13701 |
| `load_balance_proxy_server_example.py` | 原样保留代理实现，直接通过 Python 启动 |
| `deepseek31_p.sh` | DeepSeek-V3.1 成功 P 脚本，仅增加 `CPP_ENABLED` 开关；保留 0.85 显存比例、HCCL_BUFFSIZE=512 |
| `glm52_p.sh` | GLM-5.2 双机 P，合并 MRV1/MRV2 和 CPP/MTP 对照配置 |
| `glm53_p.sh` | GLM-5.3 Eco-Tech 双机 P，仅保留 24/25 成功的 MRV1、同步、CPP on、MTP on 配置 |

每个 Shell 文件均可单独复制，无公共脚本依赖，无自定义函数或命令执行包装层。GLM-5.2 与 GLM-5.3 分开保存，同一模型的节点通过 `NODE_RANK` 和少量数组参数区分；不自动连接其他机器。原实验文件保持不变，来源及 SHA256 见 [sources.json](sources.json)。

这些脚本是可直接编辑的部署材料。先准备好对应 Linux A3 容器、软件版本和权重，再修改脚本中的模型路径、网卡、IP、设备、端口等实际值。没有自动安装、环境发现或预演入口；执行脚本即启动服务。保留原脚本使用的 `ifconfig` 和 `awk`，不提供 `DRY_RUN` 开关。

## Qwen3：三个独立角色

P/D 原文默认网卡为 `data0.172`，本机 IPv4 从该网卡查询；可用 `SERVICE_NIC`、`SERVICE_IP` 覆盖。模型路径、设备、服务名和端口在脚本中直接修改。

```bash
# 分别在两个终端中运行；完整 PD 需要两个角色均就绪
bash a3_scripts/qwen3_d.sh
bash a3_scripts/qwen3_p.sh

# 两个后端均就绪后，在第三个终端启动代理
python a3_scripts/load_balance_proxy_server_example.py \
  --host 0.0.0.0 --port 18080 \
  --prefiller-hosts 127.0.0.1 --prefiller-ports 13700 \
  --decoder-hosts 127.0.0.1 --decoder-ports 13701
```

P/D 默认权重 `/mnt/weight/Qwen3-30B-A3B`、服务名 `qwen`；KV 端口为 30000/30100、engine_id 为 0/1。代理只需单独复制 Python 文件；软件依赖见其头部说明，修改后端端口后同步调整代理命令。

## DeepSeek-V3.1：同脚本切换 CPP

```bash
SERVICE_NIC=data0.172 CPP_ENABLED=true bash a3_scripts/deepseek31_p.sh

# 另一轮关闭 CPP；不能与上一轮占用相同设备/端口
SERVICE_NIC=data0.172 CPP_ENABLED=false bash a3_scripts/deepseek31_p.sh
```

`CPP_ENABLED` 使用 JSON 布尔值 `true/false`，默认 `true`。其他配置保留原文，包括权重 `/mnt/weight/DeepSeek-V3.1-Terminus-w8a8-QuaRot-lfs`、端口 8003、服务名 `model`、TP8/PP2、同步调度。网卡和 IP 的覆盖方式同 Qwen。脚本未新增可见设备限制，需在目标环境确保原拓扑所需的 16 个逻辑设备可用。

## GLM：每个模型独立脚本

两端显式设置 `NODE_RANK=0/1`，默认值为 0；节点 1 追加 `--headless`。不再按本机历史 IP 自动推断 rank。两端保持相同 master 地址、端口、模型配置及 `PROFILE_TAG`。

GLM-5.2 配置档：

| `PROFILE` | ModelRunner | 调度 | max_model_len | max_num_batched_tokens | max_num_seqs |
| --- | --- | --- | ---: | ---: | ---: |
| `mrv1`（默认） | V1 | 同步 | 200000 | 16384 | 16 |
| `mrv2` | V2 | 异步 | 266240 | 32768 | 16 |

```bash
# 节点 0；IP/网卡为归档历史值，换机器时必须修改
NODE_RANK=0 SERVICE_IP=10.246.63.28 SERVICE_NIC=enp162s0f0 \
MASTER_ADDR=10.246.63.28 PROFILE=mrv1 CPP_ENABLED=true MTP_ENABLED=true \
PROFILE_TAG=glm52_case bash a3_scripts/glm52_p.sh

# 节点 1
NODE_RANK=1 SERVICE_IP=10.246.63.29 SERVICE_NIC=enp162s0f0 \
MASTER_ADDR=10.246.63.28 PROFILE=mrv1 CPP_ENABLED=true MTP_ENABLED=true \
PROFILE_TAG=glm52_case bash a3_scripts/glm52_p.sh
```

`CPP_ENABLED`、`MTP_ENABLED` 均使用 `true/false`，默认 `true`。切换 V2 时两端一起设置 `PROFILE=mrv2`。CPP 关闭时 `smooth_factor=0.8` 配置仍保留在 JSON 中，但不代表 CPP 已启用。

GLM-5.3 保留固定成功配置：max_model_len=200000、max_num_batched_tokens=16384、max_num_seqs=64，CPP smooth_factor=1.0：

```bash
# 节点 0
NODE_RANK=0 SERVICE_NIC=data0.172 MASTER_ADDR=10.246.63.24 \
PROFILE_TAG=glm53_case bash a3_scripts/glm53_p.sh

# 节点 1
NODE_RANK=1 SERVICE_NIC=data0.172 MASTER_ADDR=10.246.63.24 \
PROFILE_TAG=glm53_case bash a3_scripts/glm53_p.sh
```

GLM-5.2 默认按 rank 使用 `.28/.29` 本机 IP、网卡 `enp162s0f0`；GLM-5.3 默认从 `data0.172` 查询本机 IP，master 为 `.24`。两者都可用 `SERVICE_NIC`、`SERVICE_IP`、`MASTER_ADDR`、`MASTER_PORT` 覆盖；不支持之前派生版本中的 `LOCAL_IP/NIC_NAME` 别名。网卡和 IP 独立设置，不验证两者是否匹配。

GLM 模型路径、服务端口、设备和服务名在脚本中直接修改；默认服务端口 30060、master 端口 7060、可见设备 0–15。GLM-5.3 仍保留原实验服务名 `glm-5.2`，需改名时同步调整客户端。原 profiler 配置及目录创建逻辑保留，可设置 `PROFILE_TAG`、`TORCH_PROFILER_DIR`；脚本启动不代表已执行采集。

DeepSeek 与 GLM 均为 P-only 归档，未包含已经验证配套的 D 服务，不能据此声称完整 PD 已复现。

## 证据与适用范围


| 归档目录（位于 `D:\experiment`） | 纳入依据与限制 |
| --- | --- |
| `202610172029_Qwen3-30B-A3B` | 128K/C1 5/5、变长 32K/C4 64/64；脚本及代理为事后快照，`execution_match=unverified`。FlashComm1 实际被禁用，不能称为 FlashComm1 成功方案。 |
| `202610172149_DeepSeek-V3.1-Terminus-w8a8-QuaRot-lfs` | CPP off 两轮 5/5；归档脚本后来被修改为 on，原件被标为 mismatch，因此 off 分支按成功日志配置从 on 脚本派生，不冒充历史原件。 |
| `202610172206_DeepSeek-V3.1-Terminus-w8a8-QuaRot-lfs` | CPP on 两轮各 5/5，0.85/512；脚本与日志配置一致，但缺少启动瞬间哈希，`execution_match=unverified`。 |
| `202610182026_GLM-5.2-w8a8c8`、`202610182114_GLM-5.2-w8a8c8` | MRV1 同步，覆盖 CPP off/on 及 MTP off/on，成功压测；来源脚本/日志关联见各记录。 |
| `202610182321_GLM-5.2-w8a8c8` | MRV2 四组均有成功 warmup/变长轮次；off/off 初次通信失败后同配置重试成功。固定 128K 阶段均因客户端 output_len=0 校验失败，不能声称该阶段验证成功；MTP 执行期状态未确认。 |
| `202610191525_GLM-5.3-w8a8c8-Eco-Tech` | 24/25 上 warmup、固定 128K、变长 32K 均成功；CPP 为 mixed，warmup 曾回退固定 chunk，不能解释为 CPP 全程生效或完整 CPP 收益。 |

未纳入 GLM-5.3 在 128/129 上就绪后请求期间发生 SDMA/507057 的两轮，以及 MRV1+PP+异步配置校验失败轮次。服务启动成功与请求完成分别判断。

GLM 归档使用 vLLM `c4df37dfcc7f72ca41cd43801a468526ecf99a2b`、vllm-ascend `f558b58c177916424ae9dcf9bb8e66c0b9031236`，完整版本和补丁状态以各 record.json 为准。DeepSeek CPP off 记录中的版本快照为 vLLM `ced6857afa0ea7b2e3f0846a62e1394e90f15607`、vllm-ascend `dd87d93ca1485de915aa145d1a98db9364f2e45f`；不要自动推广为其他轮次的历史版本。Qwen 运行日志记录 vLLM 0.30.0，历史 commit 信息不完整。脚本中的 `VLLM_VERSION=0.31.0` 是归档保留的兼容设置，不表示安装了该版本。

网络配置差异和 A3 HCCS 排错参见 [通信排错说明](../docs/a3-a5-comm.md)。脚本保留原实验 HCCS 设置和各自网卡默认值，不把某组 IP 当作所有机器通用配置。

## English

These launchers use direct environment exports followed by `vllm serve`, following the presentation style of the [official model deployment tutorials](https://docs.vllm.ai/projects/ascend/en/latest/getting_started/overview.html). Runtime settings come from successful archived experiments, rather than replacing them with current tutorial settings.

Each Shell file is independent, with no helper files, custom functions, execution wrappers or dry-run mode. Qwen P/D and the proxy Python file are restored unchanged from archive snapshots. DeepSeek retains its successful script with a `CPP_ENABLED=true|false` switch. GLM-5.2 and GLM-5.3 now have separate entry points: `glm52_p.sh` and `glm53_p.sh`.

Use the commands above inside a prepared Linux A3 container. Edit model paths, devices, service ports and served names directly in the files. Qwen/DeepSeek use `SERVICE_NIC` (default `data0.172`) and `SERVICE_IP`, otherwise querying the interface IPv4 address. GLM requires explicit node ranks 0 and 1; rank 1 is headless. Both nodes must use the same master address/port and profile settings. GLM-5.2 defaults to historical IPs `.28/.29` and NIC `enp162s0f0`; GLM-5.3 queries NIC `data0.172` and defaults its master to `.24`. Update these values for new hosts. The previous `LOCAL_IP/NIC_NAME` aliases are removed.

For GLM-5.2, `PROFILE=mrv1|mrv2` selects the historical synchronous V1 or asynchronous V2 configuration. `CPP_ENABLED` and `MTP_ENABLED` use `true/false` and default to `true`. When CPP is disabled, the retained smooth-factor value does not enable CPP. GLM-5.3 keeps only the successful V1/synchronous/CPP-on/MTP-on configuration; its archived served name remains `glm-5.2`.

The Qwen proxy is run directly from its standalone Python file; the full command is shown above. DeepSeek and GLM are P-only recipes, not validated complete PD deployments. Software dependencies must already be installed. Executing a Shell file starts the service; no preview layer is provided.

Archive sources, SHA256 hashes, version constraints and evidence limitations are documented above and in [sources.json](sources.json). In particular, GLM-5.2 V2 fixed-128K testing failed client validation, GLM-5.3 CPP execution included fallback, and some source snapshots do not prove exact historical launch bytes. The restored/edited files have not been rerun on NPUs in this task.
