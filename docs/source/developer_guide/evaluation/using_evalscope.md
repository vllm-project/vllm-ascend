# Using EvalScope

This document will guide you through model inference stress testing and accuracy testing using [EvalScope](https://github.com/modelscope/evalscope).

## 1. Online server

You can run docker container to start the vLLM server on a single NPU:

```bash
# Update DEVICE according to your device (/dev/davinci[0-7])
export DEVICE=/dev/davinci7
# Update the vllm-ascend image
export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
docker run --rm \
--shm-size=1g \
--name vllm-ascend \
--device $DEVICE \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /root/.cache:/root/.cache \
-p 8000:8000 \
-e VLLM_USE_MODELSCOPE=True \
-e PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256 \
-it $IMAGE \
vllm serve Qwen/Qwen2.5-7B-Instruct --max_model_len 26240
```

If the vLLM server is started successfully, you can see information shown below:

```shell
INFO:     Started server process [6873]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

Once your server is started, you can query the model with input prompts in a new terminal:

```shell
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "prompt": "The future of AI is",
        "max_completion_tokens": 7,
        "temperature": 0
    }'
```

## 2. Install EvalScope using pip

You can install EvalScope as follows:

```bash
python3 -m venv .venv-evalscope
source .venv-evalscope/bin/activate
pip install gradio plotly evalscope
```

## 3. Run GSM8K using EvalScope for accuracy testing

You can use `evalscope eval` to run GSM8K (a grade-school math benchmark dataset) for accuracy testing:

```shell
evalscope eval \
 --model Qwen/Qwen2.5-7B-Instruct \
 --api-url http://localhost:8000/v1 \
 --api-key EMPTY \
 --eval-type server \
 --datasets gsm8k \
 --limit 10
```

After 1 to 2 minutes, the output is shown below:

```shell
+---------------------+-----------+-----------------+----------+-------+---------+---------+
| Model               | Dataset   | Metric          | Subset   |   Num |   Score | Cat.0   |
+=====================+===========+=================+==========+=======+=========+=========+
| Qwen2.5-7B-Instruct | gsm8k     | AverageAccuracy | main     |    10 |     0.8 | default |
+---------------------+-----------+-----------------+----------+-------+---------+---------+
```

See more details in [EvalScope doc - Model API Service Evaluation](https://evalscope.readthedocs.io/en/latest/get_started/basic_usage.html#model-api-service-evaluation).

## 4. Run model inference stress testing using EvalScope

### Install EvalScope[perf] using pip

```shell
pip install evalscope[perf] -U
```

### Basic usage

You can use `evalscope perf` to run perf testing:

```shell
evalscope perf \
    --url "http://localhost:8000/v1/chat/completions" \
    --parallel 5 \
    --model Qwen/Qwen2.5-7B-Instruct \
    --number 20 \
    --api openai \
    --dataset openqa \
    --stream
```

### Output results

After 1 to 2 minutes, the output is shown below:

```shell
Benchmarking summary:
+-----------------------------------+---------------------------------------------------------------+
| Key                               | Value                                                         |
+===================================+===============================================================+
| Time taken for tests (s)          | 38.3744                                                       |
+-----------------------------------+---------------------------------------------------------------+
| Number of concurrency             | 5                                                             |
+-----------------------------------+---------------------------------------------------------------+
| Total requests                    | 20                                                            |
+-----------------------------------+---------------------------------------------------------------+
| Succeed requests                  | 20                                                            |
+-----------------------------------+---------------------------------------------------------------+
| Failed requests                   | 0                                                             |
+-----------------------------------+---------------------------------------------------------------+
| Output token throughput (tok/s)   | 132.6926                                                      |
+-----------------------------------+---------------------------------------------------------------+
| Total token throughput (tok/s)    | 158.8819                                                      |
+-----------------------------------+---------------------------------------------------------------+
| Request throughput (req/s)        | 0.5212                                                        |
+-----------------------------------+---------------------------------------------------------------+
| Average latency (s)               | 8.3612                                                        |
+-----------------------------------+---------------------------------------------------------------+
| Average time to first token (s)   | 0.1035                                                        |
+-----------------------------------+---------------------------------------------------------------+
| Average time per output token (s) | 0.0329                                                        |
+-----------------------------------+---------------------------------------------------------------+
| Average input tokens per request  | 50.25                                                         |
+-----------------------------------+---------------------------------------------------------------+
| Average output tokens per request | 254.6                                                         |
+-----------------------------------+---------------------------------------------------------------+
| Average package latency (s)       | 0.0324                                                        |
+-----------------------------------+---------------------------------------------------------------+
| Average package per request       | 254.6                                                         |
+-----------------------------------+---------------------------------------------------------------+
| Expected number of requests       | 20                                                            |
+-----------------------------------+---------------------------------------------------------------+
| Result DB path                    | outputs/20250423_002442/Qwen2.5-7B-Instruct/benchmark_data.db |
+-----------------------------------+---------------------------------------------------------------+

Percentile results:
+------------+----------+---------+-------------+--------------+---------------+----------------------+
| Percentile | TTFT (s) | ITL (s) | Latency (s) | Input tokens | Output tokens | Throughput(tokens/s) |
+------------+----------+---------+-------------+--------------+---------------+----------------------+
|    10%     |  0.0962  |  0.031  |   4.4571    |      42      |      135      |       29.9767        |
|    25%     |  0.0971  | 0.0318  |   6.3509    |      47      |      193      |       30.2157        |
|    50%     |  0.0987  | 0.0321  |   9.3387    |      49      |      285      |       30.3969        |
|    66%     |  0.1017  | 0.0324  |   9.8519    |      52      |      302      |       30.5182        |
|    75%     |  0.107   | 0.0328  |   10.2391   |      55      |      313      |       30.6124        |
|    80%     |  0.1221  | 0.0329  |   10.8257   |      58      |      330      |       30.6759        |
|    90%     |  0.1245  | 0.0333  |   13.0472   |      62      |      404      |       30.9644        |
|    95%     |  0.1247  | 0.0336  |   14.2936   |      66      |      432      |       31.6691        |
|    98%     |  0.1247  | 0.0353  |   14.2936   |      66      |      432      |       31.6691        |
|    99%     |  0.1247  | 0.0627  |   14.2936   |      66      |      432      |       31.6691        |
+------------+----------+---------+-------------+--------------+---------------+----------------------+
```

See more detail in [EvalScope doc - Model Inference Stress Testing](https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/quick_start.html#basic-usage).

## 5. Troubleshooting: KV cache hit rate is always 0 under PD

If a perf run against a Prefill-Decode (PD) deployment reports a KV cache hit
rate of `0` even though the prefiller logs show prefix-cache hits, check how the
prefiller was started.

### Why this happens

The PD proxy
([load_balance_proxy_server_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py))
returns the prefiller's `usage.prompt_tokens_details.cached_tokens` to the
client. In PD the decoder receives the prompt KV cache from the prefiller and
would count it as cached tokens, so its own value is not a prefiller cache hit;
the proxy replaces it with the prefiller count for this reason.

vLLM only fills `usage.prompt_tokens_details` when the server is started with
`--enable-prompt-tokens-details`, and it also omits the field when the
prefiller had no cache hits. Both cases look the same to the proxy, so it
reports `cached_tokens: 0` and EvalScope shows a 0% hit rate. Before v0.27 the
proxy left the decoder's own value untouched, which is why older versions could
show a non-zero hit rate even without the flag.

### Fix

Start the prefiller with `--enable-prompt-tokens-details`:

```shell
vllm serve <MODEL_PATH> ... --enable-prompt-tokens-details
```

Add the same flag to the other engines if you also call them directly outside
the proxy. For streaming requests, ensure the decode side emits a usage chunk by
sending `"stream_options": {"include_usage": true}` (EvalScope does this) or by
starting the decode engine with `--enable-force-include-usage`.

Verify the flag with two identical requests:

```shell
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"glm","messages":[{"role":"user","content":"hi"}],
       "max_tokens":1,"stream":true,
       "stream_options":{"include_usage":true}}'
```

The second response should contain a non-zero
`"usage": {"prompt_tokens_details": {"cached_tokens": ...}}`. If it still
contains `0` while the prefiller logs report hits, make sure every endpoint in
`--prefiller-hosts`/`--prefiller-ports` was restarted with the flag. If the flag
is missing, the proxy logs a warning when it first encounters a prefiller
response without prompt-token details.
