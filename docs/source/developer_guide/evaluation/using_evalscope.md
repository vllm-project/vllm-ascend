# Using EvalScope

This document will guide you through model inference stress testing and accuracy testing using [EvalScope](https://github.com/modelscope/evalscope).

## 1. Online server

You can run docker container to start the vLLM server on a single NPU:

```{code-block} bash
   :substitutions:
# Update DEVICE according to your device (/dev/davinci[0-7])
export DEVICE=/dev/davinci7
# Update the vllm-ascend image
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
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

You can use `evalscope eval` to run GSM8K for accuracy testing:

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

See more detail in [EvalScope doc - Model API Service Evaluation](https://evalscope.readthedocs.io/en/latest/get_started/basic_usage.html#model-api-service-evaluation).

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
