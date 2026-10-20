# 数据集 / Datasets

所有模型的数据集直接存放在此目录，不再创建模型子目录。沿用工具生成的文件名区分模型与输入规格，例如：

```text
datasets/
  GSM8K-in65536-num24-GLM-5.2-W4A8C8-0713-MTP.jsonl
  GSM8K-in65536-num24-DeepSeek-V4-Flash-DSpark-w4a8-scope-ci.jsonl
```

三个 `scripts/run_aisbench.sh` 入口默认调用仓库根目录的 `aisbench_auto_tools_prefix` 子模块，也可通过 `AISBENCH_TOOL_DIR` 覆盖工具路径。先在仓库根目录执行 `git submodule update --init --recursive`。

运行前，按目标模型修改根目录子模块的 `config.py`。将 `DATASET_PATH` 设置为**运行机器上**对应数据目录的绝对路径，例如：

```python
# Linux 示例；请按实际仓库位置修改
DATASET_PATH = "/home/w00985415/vaws_wd/datasets"
```

不同模型共用相同的 `DATASET_PATH`。切换模型时核对 `MODEL_NAME`、`MODEL_PATH`、`WORK_PATH`、`HOST_IP`、`HOST_PORT`、`URL`、`POD_INFO` 和 `OUTPUT_DIR`，尤其确保 `MODEL_PATH` 指向当前模型的权重及 tokenizer。工具生成的文件名包含模型权重目录名；当前不增加额外的数据集身份校验。根目录子库的通用配置不会自动继承旧模型目录里的配置。

共享的 `config.py` 同时只表示一套配置；切换模型时需更新配置。工具还会在其目录写入日志和临时文件，因此当前入口不适合直接并发运行不同模型的测试。

已有数据集暂未搬迁，历史复现实验数据保留在原处。工具按生成文件名判断数据集是否存在，存在时复用，缺失时生成。生成数据默认不提交 Git，仅跟踪此说明。

## English

Store datasets for all models directly in this directory, without model subdirectories. Keep the generated filenames, which include the model weights directory name and input specifications, as illustrated above.

All three `scripts/run_aisbench.sh` entry points use the repository-root `aisbench_auto_tools_prefix` submodule by default. `AISBENCH_TOOL_DIR` can override its location. Initialize it with `git submodule update --init --recursive` from the repository root.

Before running, edit the root submodule's `config.py` for the target model. Set `DATASET_PATH` to the absolute shared dataset directory on the machine running the test, for example `/home/w00985415/vaws_wd/datasets`. Use the same path for all models. Check `MODEL_NAME`, `MODEL_PATH`, `WORK_PATH`, `HOST_IP`, `HOST_PORT`, `URL`, `POD_INFO`, and `OUTPUT_DIR`. In particular, `MODEL_PATH` must identify the current model weights and tokenizer. Generated filenames contain the weights directory name; no additional dataset identity validation is added.

The shared configuration does not inherit settings from the old model-specific toolbox copies. Update the model configuration when switching models. The toolbox also writes shared logs and temporary files, so these entry points currently do not support running different model tests concurrently in the shared tool directory.

Existing datasets have not been moved; historical experiment datasets remain in their original locations. The tool reuses datasets when the generated filename exists and generates missing datasets. Dataset contents are ignored by Git; only this guide is tracked.
