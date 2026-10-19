# AntAngelMed

## 1 Introduction

[AntAngelMed](https://huggingface.co/MedAIBase/AntAngelMed) is a medical-domain mixture-of-experts (MoE) language model based on Ling-flash-2.0. This tutorial deploys the BF16 model on a single node with eight Ascend 910B3 NPUs and verifies structured output through the OpenAI-compatible API.

The configuration below was validated with **vLLM Ascend v0.23.0**. The version is pinned to reproduce that validation; results do not establish compatibility with other releases.

## 2 Validated Features

| Feature | Validated configuration |
| --- | --- |
| Precision | BF16, without quantization |
| Parallelism | Tensor parallelism (TP) 4, data parallelism (DP) 2, expert parallelism (EP) enabled |
| Graph execution | `FULL_AND_PIECEWISE`: piecewise prefill/decode and full decode graphs |
| Scheduling | Asynchronous scheduling and chunked prefill |
| Prefix caching | Enabled, with cache hits verified on both DP engines |
| Structured output | XGrammar: JSON Schema, JSON object, choice, and regular expression |

See the [Feature Guide](../../user_guide/feature_guide/index.md) for configuration details.

## 3 Environment Preparation

| Model weights | Tested hardware | Download links |
| --- | --- | --- |
| AntAngelMed (BF16) | One node with 8 × Ascend 910B3, 64 GB HBM per NPU | [Hugging Face](https://huggingface.co/MedAIBase/AntAngelMed) / [ModelScope](https://modelscope.cn/models/MedAIBase/AntAngelMed) |

Download the complete model, including its configuration, tokenizer, and chat template, to `/home/weights/AntAngelMed` on the host. If you use a different directory, update the Docker mount below. The container path remains `/home/weights/AntAngelMed`.

Check that the host driver is compatible with the container's CANN runtime and that all eight NPUs are available. See [Installation](../../getting_started/installation.md) for environment requirements.

```bash
npu-smi info
```

## 4 Installation

### 4.1 Pull the image

```bash
docker pull quay.io/ascend/vllm-ascend:v0.23.0
```

The tested image contained these packages; no dependency upgrades or source patches were applied:

| Package | Version |
| --- | --- |
| vLLM | `0.23.0+empty` |
| vLLM Ascend | `0.23.0` |
| XGrammar | `0.2.3` |

### 4.2 Start the container

Run on the host:

```bash
docker run --rm -it \
    --name antangelmed \
    --ipc=host \
    --net=host \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci4 \
    --device /dev/davinci5 \
    --device /dev/davinci6 \
    --device /dev/davinci7 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /home/weights/AntAngelMed:/home/weights/AntAngelMed:ro \
    quay.io/ascend/vllm-ascend:v0.23.0 bash
```

Inside the container, check the installed versions:

```bash
python3 -m pip show vllm vllm-ascend xgrammar
```

## 5 Deployment

Run inside the container:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

taskset -c 0-23 python3 -m vllm.entrypoints.openai.api_server \
    --model /home/weights/AntAngelMed \
    --max-num-seqs 200 \
    --tensor-parallel-size 4 \
    --data-parallel-size 2 \
    --enable-expert-parallel \
    --gpu-memory-utilization 0.97 \
    --served-model-name AntAngelMed \
    --max-model-len 32768 \
    --port 8080 \
    --enable-prefix-caching \
    --block-size 128 \
    --async-scheduling \
    --trust-remote-code \
    --structured-outputs-config.backend xgrammar
```

- TP4 and DP2 use all eight NPUs; EP distributes the MoE experts across them.
- `--max-num-seqs 200` is the per-engine scheduling limit. Client concurrency is distributed between the two DP engines.
- `--max-model-len 32768` is the input-plus-output limit, not the length of every test request.
- `taskset -c 0-23` reproduces the tested CPU affinity. Adjust the CPU set to the CPUs available on your host.
- `--gpu-memory-utilization 0.97` reproduces the tested memory budget. It requires available device memory; reduce it if your environment encounters out-of-memory errors.

In this image, the command enables chunked prefill and `FULL_AND_PIECEWISE` graph mode by default. Do not add `--enforce-eager` when reproducing the graph-mode test. Check startup logs for graph capture completion and inference logs for `Replaying aclgraph` on both DP engines.

## 6 Functional Verification

From another terminal on the host, wait for the health endpoint to return HTTP 200:

```bash
curl --fail http://localhost:8080/health
curl --fail http://localhost:8080/v1/models
```

### 6.1 Structured output request

Create `request.json` on the client:

```json
{
  "model": "AntAngelMed",
  "messages": [
    {
      "role": "user",
      "content": "Record 0: name is Alice, age is 30, symptoms are cough and fever. Extract name, age, symptoms as JSON. Return only JSON."
    }
  ],
  "temperature": 0,
  "max_tokens": 128,
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "patient",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "age": {"type": "integer"},
          "symptoms": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["name", "age", "symptoms"],
        "additionalProperties": false
      }
    }
  }
}
```

Send the request:

```bash
curl --fail http://localhost:8080/v1/chat/completions \
    -H 'Content-Type: application/json' \
    --data-binary @request.json
```

The response's `choices[0].message.content` should contain a JSON object satisfying the schema, for example:

```json
{"name":"Alice","age":30,"symptoms":["cough","fever"]}
```

This exercises XGrammar constraints during generation. Asking for JSON only in the prompt does not exercise the same constrained-decoding path.

### 6.2 Concurrent verification

For a client-side smoke test, install `aiohttp` and `jsonschema`, then save the following as `check_concurrency.py` beside `request.json`. It sends both all-structured and mixed ordinary/structured requests at each concurrency level. It checks the returned schema and writes each complete response to `responses.jsonl`.

```bash
python3 -m pip install aiohttp jsonschema
```

```python
import asyncio
import copy
import json
from pathlib import Path

import aiohttp
import jsonschema


async def main():
    template = json.loads(Path("request.json").read_text())
    schema = template["response_format"]["json_schema"]["schema"]
    timeout = aiohttp.ClientTimeout(total=900)
    connector = aiohttp.TCPConnector(limit=0)
    with Path("responses.jsonl").open("w") as output:
        async with aiohttp.ClientSession(
            timeout=timeout, connector=connector
        ) as session:
            async def send(index, mixed, gate):
                body = copy.deepcopy(template)
                structured = not mixed or index % 2 == 0
                if not structured:
                    body.pop("response_format")
                    body["max_tokens"] = 64
                await gate.wait()
                async with session.post(
                    "http://localhost:8080/v1/chat/completions", json=body
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                output.write(json.dumps(result) + "\n")
                choice = result["choices"][0]
                content = choice["message"]["content"]
                if structured:
                    assert choice["finish_reason"] == "stop", choice
                    jsonschema.validate(json.loads(content), schema)
                else:
                    assert content and choice["finish_reason"] in ("stop", "length")

            for concurrency in (8, 32, 64, 200):
                for mixed in (False, True):
                    gate = asyncio.Event()
                    tasks = [
                        asyncio.create_task(send(i, mixed, gate))
                        for i in range(concurrency)
                    ]
                    gate.set()
                    await asyncio.gather(*tasks)
                    print(f"PASS: concurrency={concurrency}, mixed={mixed}")


asyncio.run(main())
```

```bash
python3 check_concurrency.py
```

## 7 Accuracy Evaluation

### 7.1 Using AISBench

Use [AISBench](../../developer_guide/evaluation/using_ais_bench.md) to evaluate model accuracy through the running OpenAI-compatible service. Install AISBench and the selected dataset as described in that guide, then update `ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py` for AntAngelMed:

```python
from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.model_postprocessors import (
    extract_non_reasoning_content,
)


models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr="antangelmed-vllm-api",
        path="/home/weights/AntAngelMed",
        model="AntAngelMed",
        request_rate=0,
        retry=2,
        host_ip="localhost",
        host_port=8080,
        max_out_len=4096,
        batch_size=8,
        trust_remote_code=True,
        generation_kwargs=dict(
            temperature=0.0,
            top_p=1.0,
            seed=0,
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]
```

For example, after preparing the C-Eval dataset, run:

```bash
ais_bench \
    --models vllm_api_general_chat \
    --datasets ceval_gen_0_shot_cot_chat_prompt.py \
    --mode all \
    --dump-eval-details \
    --merge-ds
```

AISBench writes the final scores under `outputs/default/<timestamp>/summary/`. Keep the same prompt template, decoding parameters, dataset revision, and vLLM Ascend image when comparing results across environments. The functional results in Section 6 validate constrained decoding only; they are not accuracy scores.

## 8 Performance Evaluation

Run performance evaluation against the service started in Section 5. Record the hardware, image, service command, workload shape, request rate, and concurrency together with the results so that measurements remain comparable.

### 8.1 Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for setup details. After configuring the model endpoint as shown in Section 7.1, a C-Eval performance run can be started with:

```bash
ais_bench \
    --models vllm_api_general_chat \
    --datasets ceval_gen_0_shot_cot_chat_prompt.py \
    --summarizer default_perf \
    --mode perf
```

Use a dataset and input/output lengths representative of the target workload. AISBench reports latency and throughput statistics in its generated output directory.

### 8.2 Using vLLM Benchmark

The `vllm bench serve` command measures online serving performance. The following example sends 200 random requests to the AntAngelMed service with up to 200 concurrent requests:

```bash
vllm bench serve \
    --base-url http://localhost:8080 \
    --endpoint /v1/chat/completions \
    --model AntAngelMed \
    --tokenizer /home/weights/AntAngelMed \
    --trust-remote-code \
    --dataset-name random \
    --random-input 200 \
    --random-output-len 128 \
    --num-prompts 200 \
    --request-rate inf \
    --max-concurrency 200 \
    --save-result \
    --result-dir ./antangelmed-benchmark-results
```

`--random-input 200` and `--random-output-len 128` define a short synthetic workload. Change them to match production prompt and response lengths. Reduce `--max-concurrency` to measure lower-load points such as 8, 32, and 64. Run each workload multiple times after warm-up and compare successful request count, request throughput, output-token throughput, time to first token, and time per output token.

The concurrency script in Section 6 is a correctness check with short requests. Its elapsed time should not be used as a benchmark result because the test was not designed for controlled performance measurement.

## 9 Conclusion

AntAngelMed BF16 was validated on eight Ascend 910B3 NPUs with vLLM Ascend v0.23.0 using TP4, DP2, expert parallelism, graph execution, asynchronous scheduling, prefix caching, and XGrammar structured output. Use the accuracy and performance procedures above to evaluate the model with datasets and workload shapes appropriate to your application.
