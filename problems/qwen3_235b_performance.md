# 服务拉起脚本

```shell
export VLLM_USE_V1=1
export HCCL_OP_EXPANSION_MODE="AIV"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ENGINE_READY_TIMEOUT_S=1800

vllm serve /data/weights/Qwen3-235B-A22B-w8a8-rot --served-model-name qwen3_235b --host 0.0.0.0 --port 8003 --async-scheduling --tensor-parallel-size 4 --data-parallel-size 4 --data-parallel-size-local 4 --data-parallel-start-rank 0 --data-parallel-address 141.61.81.133 --data-parallel-rpc-port 2345 --max-num-seqs 120 --max-model-len 131072 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.9 --enable-expert-parallel --no-enable-prefix-caching --quantization "ascend" --trust-remote-code --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' --hf-overrides '{"rope_parameters": {"rope_type":"yarn","rope_theta": 1000000.0,"factor":4.3,"original_max_position_embeddings":32768}}' --additional-config '{"ascend_scheduler_config":{"enabled":false},"pa_shape_list": [4,8,16,32,48,64,96,128,160,192]}'
```

# aisbench 脚本

```python
from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr="vllm-api-stream-chat",
        path="/data/weights/Qwen3-235B-A22B-w8a8-rot",
        model="qwen3_235b",
        stream=True,
        request_rate = 0,
        use_timestamp=False,
        retry=2,
        api_key="",
        host_ip = "141.61.81.133",
        host_port = 8004,
        url="",
        max_out_len = 4096,
        batch_size = 15,
        trust_remote_code = False,
        generation_kwargs=dict(
            temperature = 0,
            ignore_eos = True,
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]
```

aisbench 命令
```shell
ais_bench --models vllm_api_stream_chat --datasets gsm8k_gen_0_shot_cot_str_perf --debug --num-prompt 60 -m perf
```
