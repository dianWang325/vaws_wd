from ais_bench.benchmark.models import VLLMCustomAPIStream

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIStream,
        abbr='vllm-api-stream-completions',
        path="/mnt/weight/DeepSeek-V4-Flash-DSpark-w4a8-scope-ci",
        model="dsv4",
        api_key="",
        request_rate=0,
        retry=2,
        host_ip="80.5.9.129",
        host_port=9000,
        url="",
        max_out_len=2560,
        batch_size=2,
        generation_kwargs=dict(
            temperature=0,
            ignore_eos=True,
            add_special_tokens=False
        )
    )
]
