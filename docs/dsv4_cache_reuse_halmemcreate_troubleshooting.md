# DeepSeek-V4 Cache-Reuse e2e 测试 Memcache 初始化失败排查报告

> 日期：2026-09-20
> 环境：wd_test0917 容器 @ 80.5.9.138（prefill，node-97-39）/ 80.5.9.133（decode）
> 测试：`tests/e2e/weekly/multi_node/external_dp/config/DeepSeek-V4-Flash-Memcache-PD-cache-reuse-w4a8-16npu.yaml`

---

## 1. 背景

### 1.1 测试框架结构（配置驱动）

vllm-ascend 的 e2e 测试采用 **yaml 配置 + 通用运行器** 模式：

- **yaml（配置）**：声明集群拓扑（`num_nodes`/`npu_per_node`/`dp`/`tp`）、PD 分离路由、kv_pool（memcache）、vllm serve 启动参数模板、AISBench 基准（baseline/threshold）
- **真正的测试脚本**（weekly 复用 nightly，无独立运行器）：
  - `tests/e2e/nightly/multi_node/external_dp/scripts/test_external_dp.py` —— pytest 入口 `test_external_dp()`
  - `external_dp_config.py` —— `ExternalDPConfigLoader.from_yaml()` 解析配置
  - `runtime.py` —— 拉起 vllm server / proxy / kv_pool
- **yaml 定位方式**：环境变量 `CONFIG_YAML_PATH` + `CONFIG_BASE_PATH`
- weekly 目录下唯一脚本 `check_dsv4_cache_reuse_log.py`：测试后从 prefill rank 日志归档中校验 KV 复用压缩日志是否出现

### 1.2 本次测试配置（16 卡简化版）

正式版本为 2 节点 × 16 NPU（w8a8）；为方便在普通服务器测试，改用 **2 节点 × 8 NPU（w4a8）** 版本：

| 项 | 值 |
|---|---|
| 模型 | `gdydems/DeepSeek-V4-Flash-w4a8-mtp` |
| 拓扑 | node0（138）= prefiller：dp2 × tp4；node1 = decoder：dp2 × tp4 |
| kv_pool | memcache，`protocol: device_sdma`，`dram.size: 1GB`，`world_size: 256` |
| KV 连接器 | MultiConnector = MooncakeHybridConnector + AscendStoreConnector（layerwise，4 个共享 buffer） |

---

## 2. 问题现象

测试运行时，**prefill 节点 138 上 Memcache 初始化反复失败**，服务未就绪，无推理精度结果：

```
ERROR [HYBM hybm_vmm_based_segment.cpp:196 MallocFromHost] Try HalMemCreate failed ret:6 device:N size:1073741824
ERROR [HYBM hybm_vmm_based_segment.cpp:276 AllocLocalMemory] Assert ret == BM_OK, HalMemCreate failed: 6 segType:1
```

- 3 次运行（19:37 / 19:45 / 19:51），**8 个设备（0-7）全部失败，100% 复现**
- 每次失败点相同：每个 worker 进程向驱动申请 1GB 主机内存（`dram.size`）时出错

---

## 3. 排查过程与发现

### 3.1 错误码确认

`ret:6` = **`DRV_ERROR_OUT_OF_MEMORY`**（驱动头文件 `ascend_hal_error.h:40`）

### 3.2 失败时序（run3 日志）

```
19:47:45  vllm serve 启动
19:50:41  模型权重加载完成（25.03 GB/卡，HBM）✅
19:50:48  KV cache 分配完成（22.10 GiB 可用）✅
19:50:50  Mooncake 注册设备侧显存（mem type:device，不占主机内存）✅
19:50:53  Memcache store 初始化开始
19:51:05  HalMemCreate(1GB host) 失败 ❌（~12 秒后）
```

→ 模型与 KV cache 均正常，**仅主机内存申请失败**

### 3.3 已排除的假设

| 假设 | 排除依据 |
|---|---|
| 主机内存不足 | 138 共 2TB 内存，空闲 ~1.8TB |
| memlock 限制（ulimit -l 仅 64MB） | 实测裸进程可直接通过 halMemCreate 分配 32GB；特权容器（CAP_IPC_LOCK）不受 rlimit 约束 |
| 驱动池被持久耗尽/泄漏 | **设备空闲时用 ctypes 直调 `halMemCreate`：1GB/256MB/64MB 立即成功；单进程连续分配 32GB 全部成功** |

### 3.4 最小复现验证（关键证据）

测试脚本：`/tmp/test_halmem.py`（容器内），通过 `libascend_hal.so` 直接调用：

```python
halMemCreate(host DDR, 1GB, dev0) → ret=0 ✅   # 空闲时立即成功
halMemCreate × 32 个 1GB chunk   → 全部成功 ✅  # 池容量 ≥ 32GB
```

**结论：只有测试运行期间主机内存池才被耗尽；测试一停，池子即恢复。**

### 3.5 环境背景（重要）

- **138 是共享机器**：40+ 个容器在跑，**至少 9 个容器共享 davinci0-7** 设备（ylf_vllm_profiling、y00937252、codex-jenga、ycl-sparseattn、y30084614_M3 等）
- 测试时段前后出现异常事件：
  - ~19:46（run2 时段）133 的 wd_test0917 容器被 SIGKILL（exit 137）
  - ~19:53 138 的 k8s-apiserver 重启
- 所有芯片存在 **~3GB HBM 残留且无进程**（历史进程被强杀、资源未正常释放的痕迹）

---

## 4. 结论

**故障性质：运行态瞬态资源耗尽，而非机器/配置损坏。**

Memcache 所需的 8 × 1GB 主机钉住内存，在测试运行的那个时间窗口内申请不到；最可能的原因是**共享机上其他容器同期占用**（或叠加测试自身 layerwise 共享 buffer 等隐性占用），把驱动主机内存池吃满。

---

## 5. 下一步计划

1. **测池子上限**：空闲时持续分配直到失败，确定驱动主机内存池总容量基线
2. **复现时监控**：重跑测试，同时监控主机内存/各容器进程，定位当时的占用方
3. **错峰验证**：挑机器空闲时段重跑一次 —— 若通过，即坐实为共享资源争抢问题
4. （可选）评估减小 `dram.size` 或调整 `layerwise_num_shared_buffers` 降低主机内存需求
