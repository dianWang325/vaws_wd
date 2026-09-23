from pathlib import Path

# 数据集文件夹路径，需可访问(请使用绝对路径)
DATASET_PATH = "/home/w00985415/vaws_wd/glm5_2/datasets"

# aisbench 工作路径, 为 git clone aisbench 后得到的 benchmark 目录的绝对路径
# 可通过命令 `pip show ais-bench-benchmark` 查询location
WORK_PATH = "/home/w00985415/benchmark"

# 服务化配置的模型名称
MODEL_NAME = "glm-52"
# 模型权重路径, 用于读取 tokenizer
MODEL_PATH = "/mnt/weight/GLM-5.2-W4A8C8-0713-MTP"
# 请求目的 IP
HOST_IP = "141.xx.xx.xx"
# 请求目的端口
HOST_PORT = "8004"

# 模型baseUrl, 配置后会忽略HOST_IP、HOST_PORT配置
URL=""

## 鉴权信息
API_KEY = ""

# 如果使用稳态测试请将该字段设置为 "stable_stage"
DEFAULT_PERFORMANCE_TEST = "default_perf"

# aisbench输出日志保存路径
OUTPUT_DIR = str(Path(__file__).resolve().parent.parent / "outputs" / "default")

# 各节点信息，格式为 ["{ip}:{port}"]
# 用于查询vllm metrics计算各个dp域的prefix cache命中率，不配置默认为HOST_IP:HOST_PORT
# PD分离场景请填写各个P节点的IP和对应dp域的port
# POD_INFO = ["141.xx.xx.11:8000","141.xx.xx.12:8000"]
POD_INFO = []
