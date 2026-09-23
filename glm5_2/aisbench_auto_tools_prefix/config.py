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
# PD 分离部署：客户端统一访问 proxy（默认放在 P 节点 80.5.17.110 容器内，9000 端口）
# 若 proxy 改放其他机器（如本机），请同步修改此处
HOST_IP = "80.5.17.110"
# 请求目的端口
HOST_PORT = "9000"

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
# 本部署：单个 P 节点 80.5.17.110（PP2xTP8 单机），metrics 端口 = vllm serve 端口 18080
# POD_INFO = ["141.xx.xx.11:8000","141.xx.xx.12:8000"]
POD_INFO = ["80.5.17.110:18080"]
