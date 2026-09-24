from ais_bench.benchmark.models import VLLMCustomAPIChatStream

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChatStream,
        abbr='vllm-api-stream-chat',
        path="/mnt/weight/GLM-5.2-W4A8C8-0713-MTP",
        model="glm-52",
        api_key="",
        request_rate=0,
        retry=2,
        host_ip="80.5.9.138",
        host_port=9000,
        url="",
        max_out_len=10000,
        batch_size=2048,
        generation_kwargs=dict(
            temperature=1.0,
            top_p = 0.95
        )
    )
]
