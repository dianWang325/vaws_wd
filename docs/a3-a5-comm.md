# A3/A5 通信排错 / Communication troubleshooting

## 中文

本文整理用户提供的部署故障证据与规避方案，未在本次文档整理中重新执行验证。结论适用于下述已观察场景，不代表所有 A3/A5 环境的默认行为。

### 快速定位

| 场景 | 现象 | 优先处理 |
| --- | --- | --- |
| A3：Mooncake/ADXL 设备通信初始化 | RoCE 端口不可用，Transfer Engine 初始化失败，EngineCore 无法启动 | 显式选择 `hccs:device`，设置本地通信资源格式 |
| A5：跨节点 CPU/Gloo 通信 | `preamble.length ... vs 4/8`、收到的数据大小不匹配、worker 退出 | 核对管理网路径，按已验证方案切换数据网 IP 和网卡 |

### A3：显式使用 HCCS 设备通信

在已观察的环境中，Mooncake 未显式指定设备通信协议时进入 RoCE 初始化路径。设备 RoCE 端口不可用，导致 ADXL/Transfer Engine 初始化失败，进而导致 vLLM EngineCore 无法启动。

在需要该设备通信路径的进程启动前设置以下环境变量，并确保启动脚本将其传递到实际运行进程：

```bash
export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["hccs:device"]}'
export ASCEND_LOCAL_COMM_RES='{"version":"1.2"}'

# 原环境保留的 ACL 初始化设置；不是设备通信协议的选择开关
export ACL_OP_INIT_MODE=1
```

| 配置 | 本案例中的作用 |
| --- | --- |
| `ASCEND_GLOBAL_RESOURCE_CONFIG` | 显式指定 Mooncake/ADXL 设备通信协议为 `hccs:device` |
| `ASCEND_LOCAL_COMM_RES` | 设置本案例 A3 HCCS 使用的本地通信资源格式 |
| `ACL_OP_INIT_MODE=1` | 调整 ACL 算子初始化模式；原部署用于加快算子编译，与通信协议选择关系不大 |

复查初始化日志，确认不再因不可用的设备 RoCE 端口失败，并确认 Transfer Engine 与 EngineCore 启动成功。该设置针对本案例 A3 机器内部的设备通信，不能据此推断跨节点数据链路也已切换。

### A5：跨节点通信切换到数据网

#### 报错特征

```text
terminate called after throwing an instance of 'gloo::EnforceNotMet'
what(): [enforce fail at /pytorch/third_party/gloo/gloo/transport/tcp/pair.cc:456]
op.preamble.length <= op.nbytes.
10075403489557808663 vs 8.
Received data size doesn't match expected size.
Is there a distributed collective mismatch in your code?

Worker proc VllmWorker-8 died unexpectedly, shutting down executor.
```

在该故障样本中，`8` 是接收端准备接收的 8 字节长度字段，前面的巨大数字是 Gloo 从异常字节中解析出的消息长度。仅凭这个报错不能确定底层故障点，需结合以下对照证据判断。

#### 已有证据与结论边界

- 不加载 vLLM、Torch 和 NPU 的普通 TCP 程序，在容器和宿主机上都曾收到错误字节。
- 相同 CPU/Gloo 程序在两台机器各自单机运行均通过。
- 改用数据网 `172.27.8.12` / `172.27.8.21` 后，16 个 rank 连续一万轮校验通过。
- 未开启 CPP 时也出现过同类 `preamble.length ... vs 4/8` 错误。

据此，本案例最高优先级怀疑是 `141.61.33.x` 管理网路径的数据完整性异常。具体故障点可能涉及管理网卡、驱动、内核、offload 或交换链路，尚未进一步定位。切换数据网是已有对照验证的规避方案，不等于底层原因已完全查明。

CPP、SRF、PCP 或 scheduler 可能改变流量模式与复现时间，但不是该问题的必要触发条件。

#### 配置示例

在两台机器对应的启动环境中，成套替换本机 IP、网卡及主节点地址：

节点 0（管理 IP：`141.61.33.12`）：

```bash
export NODE_RANK=0
export LOCAL_IP=172.27.8.12
export NIC_NAME=data0.3001
export MASTER_ADDR=172.27.8.12
```

节点 1（管理 IP：`141.61.33.21`）：

```bash
export NODE_RANK=1
export LOCAL_IP=172.27.8.21
export NIC_NAME=data0.3001
export MASTER_ADDR=172.27.8.12
```

这些地址和网卡名来自本案例，其他部署需替换为实际值。`LOCAL_IP` 和 `NIC_NAME` 是启动脚本使用的配置变量；仅导出变量不能保证所有通信组件自动采用它们。检查脚本是否将这些值传入实际的进程绑定、通信后端网卡选择与启动参数，并检查其他位置是否仍引用旧管理网 IP。两个节点的 `MASTER_ADDR` 必须一致。

切换后，先在相同容器和网络环境中复跑已有普通 TCP 与 CPU/Gloo 完整性测试，再复跑 vLLM 部署。记录实际 IP、网卡、rank 数、校验轮数和结果；若仍有异常，应继续定位网络路径，而不能仅根据错误文本归因于 collective 调用不一致。

## English

This guide records deployment evidence and workarounds supplied by the user. The tests were not rerun while preparing this document. Findings apply to the observed cases below, rather than every A3/A5 environment.

### Quick triage

| Scenario | Symptom | First action |
| --- | --- | --- |
| A3: Mooncake/ADXL device initialization | Unavailable device RoCE ports; Transfer Engine initialization fails; EngineCore cannot start | Explicitly select `hccs:device` and set the local communication resource format |
| A5: cross-node CPU/Gloo communication | `preamble.length ... vs 4/8`, received-size mismatch, worker exit | Check the management network path and apply the validated data-network IP/NIC workaround |

### A3: explicitly select HCCS device communication

In the observed environment, Mooncake entered the RoCE initialization path when no device communication protocol was explicitly selected. Unavailable device RoCE ports caused ADXL/Transfer Engine initialization to fail, preventing vLLM EngineCore startup.

Set the following variables before starting processes that require this device communication path. Ensure the launcher passes them to the actual runtime processes:

```bash
export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["hccs:device"]}'
export ASCEND_LOCAL_COMM_RES='{"version":"1.2"}'

# ACL initialization setting retained from the original deployment;
# this does not select the device communication protocol.
export ACL_OP_INIT_MODE=1
```

`ASCEND_GLOBAL_RESOURCE_CONFIG` selects `hccs:device` for Mooncake/ADXL device communication. `ASCEND_LOCAL_COMM_RES` supplies the local resource format used for A3 HCCS in this case. `ACL_OP_INIT_MODE=1` changes ACL operator initialization behavior; the original deployment retained it to accelerate operator compilation, rather than to select a communication protocol.

Check initialization logs to confirm that unavailable device RoCE ports no longer cause failure and that Transfer Engine and EngineCore start successfully. This configuration addresses device communication within the A3 machine in this case; it does not establish that cross-node traffic has also moved to another network.

### A5: move cross-node communication to the data network

#### Error signature

The error excerpt in the Chinese section is the original signature: Gloo fails at `op.preamble.length <= op.nbytes`, reports `10075403489557808663 vs 8`, and the worker exits.

In this sample, `8` is the length field the receiver was prepared to receive, while the extremely large number is the message length Gloo decoded from abnormal bytes. The error alone does not identify the underlying fault; use the controlled evidence below.

#### Evidence and limits of the conclusion

- A plain TCP program without vLLM, Torch or NPU usage received incorrect bytes in both container and host tests.
- The same CPU/Gloo program passed when run on each machine independently.
- After switching to `172.27.8.12` / `172.27.8.21`, 16 ranks passed 10,000 consecutive validation rounds.
- Similar `preamble.length ... vs 4/8` errors occurred with CPP disabled.

The highest-priority suspect in this case is a data-integrity problem on the `141.61.33.x` management network path. The precise fault could involve the management NIC, driver, kernel, offload or switching path and remains unlocalized. Moving to the data network is a workaround supported by the reported comparison, rather than a complete identification of the underlying cause.

CPP, SRF, PCP or scheduler behavior may change traffic patterns and time to reproduction, but none is a necessary trigger for this problem.

#### Configuration and verification

Use the two node-specific shell examples in the Chinese section above. The complete mapping is:

| Setting | Node 0 | Node 1 |
| --- | --- | --- |
| Management IP, for identification | `141.61.33.12` | `141.61.33.21` |
| `NODE_RANK` | `0` | `1` |
| `LOCAL_IP` | `172.27.8.12` | `172.27.8.21` |
| `NIC_NAME` | `data0.3001` | `data0.3001` |
| `MASTER_ADDR` | `172.27.8.12` | `172.27.8.12` |

Replace these case-specific addresses and NIC names for other deployments. `LOCAL_IP` and `NIC_NAME` are launcher configuration variables; exporting them alone does not guarantee that every communication component uses them. Check that the launcher applies them to actual process bindings, backend interface selection and launch arguments. Check for remaining references to the old management IPs elsewhere. Both nodes must use the same `MASTER_ADDR`.

After switching, rerun the existing plain TCP and CPU/Gloo integrity tests in the same container and network environment before retrying vLLM. Record the actual IPs, NICs, rank count, validation rounds and outcomes. If the problem persists, continue investigating the network path rather than attributing it to mismatched collectives solely from the error text.
