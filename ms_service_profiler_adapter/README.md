# 按实验配置的 MS Service Profiler

本目录只在当前 Windows 本地仓库准备，**没有同步到远端，也没有启动或切换任何服务**。设计遵循 [vLLM Ascend 的 MS Service Profiler 文档](https://docs.vllm.ai/projects/ascend/zh-cn/latest/developer_guide/performance_and_debug/service_profiling_guide.html#ms-service-profiler)：服务启动前设置 `SERVICE_PROF_CONFIG_PATH`，可选设置 `PROFILING_SYMBOLS_PATH`；运行中通过 JSON 的 `enable` 开关采集；最后用 `msserviceprofiler parse` 提取结果。

## 每类探测一份 YAML

`experiments/service_overview.yaml` 是不指定模型运行器和并行拓扑的基础探测，使用安装包自带的默认符号。`experiments/ttft_worker_v2.yaml` 示范如何为 MRV2 增加 worker 和 model runner 计时点。`experiments/glm52_dp2_pp2_tp8_dsacp_ttft.yaml` 沿用这些计时点，为 GLM-5.2 的 DP2/PP2/TP8 DSA-CP 实验使用独立的采集目录；并行拓扑和 `VLLM_USE_V2_MODEL_RUNNER=1` 由 `glm5_2/scripts/glm52_dsacp_V2.sh` 设置。该实验在两个节点分别使用 `--node head` 和 `--node worker` 准备及开关采集。可复制 YAML 形成新的实验，分别设置采集目录、采集级别、时长和追加符号：

```yaml
name: my_experiment
profile:
  prof_dir: ./profiles/{experiment}/{node}
  acl_task_time: 0
  acl_prof_task_time_level: ""
symbols:
  add: []
```

`profilectl.py` 把实验 YAML 转为 profiler 原生 JSON。只有 `symbols.add` 非空时，才复制**当前运行环境安装包**的默认 YAML 并追加实验埋点，然后设置 `PROFILING_SYMBOLS_PATH`。这保留默认请求/批次点位，也允许每次实验自行配置 YAML。符号表在服务启动时加载；**修改实验 YAML 或符号后必须重启该服务实例**。`start`/`stop` 只切换采集，不热更新埋点。

## 命令

在实际运行 vLLM 的环境中，每个服务实例分别执行，`--node` 是自定义实例标识，不假定 PP/TP/DP 配置。以下仅为将来使用的命令，当前没有执行：

```bash
# 用实验配置启动原本的部署命令；服务初始 enable=0。
python profilectl.py launch -c experiments/service_overview.yaml --node host_a -- \
  bash /path/to/existing_serve_script.sh head

# 另一个终端，在目标请求前开启、请求结束后关闭。
python profilectl.py start -c experiments/service_overview.yaml --node host_a
python profilectl.py status -c experiments/service_overview.yaml --node host_a
python profilectl.py stop -c experiments/service_overview.yaml --node host_a

# 对该实例本次运行生成的原始数据目录解析；多节点逐个解析。
python profilectl.py parse --input-path /path/to/raw-run --output-path /path/to/parsed
```

如果服务由外部调度器启动，先运行 `prepare`，从打印的 manifest 读取 `config_path` 和可选的 `symbols_path`，把它们作为 `SERVICE_PROF_CONFIG_PATH`、`PROFILING_SYMBOLS_PATH` 注入原有启动命令；启动后仍用 `start`/`stop`。可通过 `--state-root` 把生成的 JSON/YAML 放到实验代码目录之外；同一实例的所有命令需传相同的 `--state-root`。在多节点部署中，每个节点都要准备自己的配置文件和输出目录，并分别开关。

解析后的 `request.csv`、`batch.csv`、`profiler.db`、`chrome_tracing.json` 等是通用结果。先确认目标请求和所需 span 确实存在，再按请求 ID、进程/rank 和时间分析；并行 rank 的时间不能直接相加。`acl_task_time=0` 对应服务框架事件。需要 NPU 算子及通信事件时，可在**另一份实验 YAML** 中按文档改用 `acl_task_time=1` 及合适的 `acl_prof_task_time_level`；较重的采集可能改变 TTFT，基线需单独记录。

## 当前环境的核对结果与限制

- 已检查的 `wd_test0825` 有 CANN 自带的 `ms_service_profiler 26.0.0`，但缺少 `msguard`，所以 `msserviceprofiler parse` 的 CLI 目前无法加载。当前没有安装依赖；将来使用前须先验证 `msserviceprofiler parse --help`。
- 该版本默认符号包含 V1 的 `NPUModelRunner.execute_model`，没有 MRV2 对应点位。基础探测仍可用于通用框架时间线；想看 MRV2 worker 级 span，可选 `ttft_worker_v2.yaml`。追加点位是否实际命中，要在启动后检查 trace，不能仅凭 YAML 断定。
- 该工具不能给已启动但未设置 `SERVICE_PROF_CONFIG_PATH` 的进程补采历史请求。其主机侧 span 也不能单独证明 NPU 计算时间或异步通信完成时间。
