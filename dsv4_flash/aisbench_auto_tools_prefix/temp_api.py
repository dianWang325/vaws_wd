from ais_bench.benchmark.models import VLLMCustomAPIChatStream

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChatStream,
        abbr='vllm-api-stream-chat',
        path="/mnt/weight/DeepSeek-V4-Flash-DSpark-w4a8-scope-ci",
        model="dsv4",
        api_key="",
        request_rate=0,
        retry=2,
        host_ip="80.5.9.138",
        host_port=9000,
        url="",
        max_out_len=3000,
        batch_size=64,
        generation_kwargs=dict(
            temperature=1.0,
            top_p = 0.95
        )
    )
]
