# Using EvalScope

This document will guide you have model inference stress testing and accuracy testing using [EvalScope](https://github.com/modelscope/evalscope).

## 1. Online Serving

You can run docker container to start the vLLM server on a single NPU:

```{code-block} bash
   :substitutions:
# Update DEVICE according to your device (/dev/davinci[0-7])
export DEVICE=/dev/davinci7
# Update the vllm-ascend image
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
docker run --rm \
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

If your service start successfully, you can see the info shown below:

```
INFO:     Started server process [6873]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

Once your server is started, you can query the model with input prompts in new terminal:

```
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "prompt": "The future of AI is",
        "max_tokens": 7,
        "temperature": 0
    }'
```

## 2. Install EvalScope Using pip

We recommend using conda to manage your environment and installing dependencies with pip:

​	1.Create a conda environment (optional)

```shell
# It is recommended to use Python 3.10
conda create -n evalscope python=3.10
# Activate the conda environment
conda activate evalscope
```

​	2.Install dependencies using pip

```shell
pip install evalscope                # Install Native backend (default)
# Additional options
pip install 'evalscope[opencompass]'   # Install OpenCompass backend
pip install 'evalscope[vlmeval]'       # Install VLMEvalKit backend
pip install 'evalscope[rag]'           # Install RAGEval backend
pip install 'evalscope[perf]'          # Install dependencies for the model performance testing module
pip install 'evalscope[app]'           # Install dependencies for visualization
pip install 'evalscope[all]'           # Install all backends (Native, OpenCompass, VLMEvalKit, RAGEval)
```

## 3. Run model inference stress testing using EvalScope

### Environment Preparation

```shell
# Install additional dependencies
pip install evalscope[perf] -U
```

### Basic Usage

You should launch a model service using vllm ascend online serving.

You can start the model inference performance stress testing tool using the following two methods:

```
evalscope perf \
    --url "http://localhost:8000/v1/completions" \
    --parallel 5 \
    --model Qwen/Qwen2.5-7B-Instruct \
    --number 20 \
    --api openai \
    --dataset openqa \
    --stream
```

Please refer to this link for [parameters](https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/parameters.html).

### Output Results

After 1-2 mins, the output is as shown below, [metric descriptions](https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/quick_start.html#metric-descriptions) refer to this link: 

```shell
Benchmarking summary:
+-----------------------------------+-----------------------------------------------------------------+
| Key                               | Value                                                           |
+===================================+=================================================================+
| Time taken for tests (s)          | 3.882                                                           |
+-----------------------------------+-----------------------------------------------------------------+
| Number of concurrency             | 5                                                               |
+-----------------------------------+-----------------------------------------------------------------+
| Total requests                    | 20                                                              |
+-----------------------------------+-----------------------------------------------------------------+
| Succeed requests                  | 20                                                              |
+-----------------------------------+-----------------------------------------------------------------+
| Failed requests                   | 0                                                               |
+-----------------------------------+-----------------------------------------------------------------+
| Output token throughput (tok/s)   | 983.7757                                                        |
+-----------------------------------+-----------------------------------------------------------------+
| Total token throughput (tok/s)    | 1242.6641                                                       |
+-----------------------------------+-----------------------------------------------------------------+
| Request throughput (req/s)        | 5.152                                                           |
+-----------------------------------+-----------------------------------------------------------------+
| Average latency (s)               | 0.8416                                                          |
+-----------------------------------+-----------------------------------------------------------------+
| Average time to first token (s)   | 0.0247                                                          |
+-----------------------------------+-----------------------------------------------------------------+
| Average time per output token (s) | 0.0044                                                          |
+-----------------------------------+-----------------------------------------------------------------+
| Average input tokens per request  | 50.25                                                           |
+-----------------------------------+-----------------------------------------------------------------+
| Average output tokens per request | 190.95                                                          |
+-----------------------------------+-----------------------------------------------------------------+
| Average package latency (s)       | 0.0043                                                          |
+-----------------------------------+-----------------------------------------------------------------+
| Average package per request       | 190.95                                                          |
+-----------------------------------+-----------------------------------------------------------------+
| Expected number of requests       | 20                                                              |
+-----------------------------------+-----------------------------------------------------------------+
| Result DB path                    | outputs/20250410_133556/Qwen2.5-0.5B-Instruct/benchmark_data.db |
+-----------------------------------+-----------------------------------------------------------------+

Percentile results:
+------------+----------+---------+-------------+--------------+---------------+----------------------+
| Percentile | TTFT (s) | ITL (s) | Latency (s) | Input tokens | Output tokens | Throughput(tokens/s) |
+------------+----------+---------+-------------+--------------+---------------+----------------------+
|    10%     |  0.0118  |  0.004  |   0.3434    |      42      |      80       |       215.1057       |
|    25%     |  0.0122  | 0.0041  |   0.5027    |      47      |      113      |       223.166        |
|    50%     |  0.0157  | 0.0042  |   0.7857    |      49      |      194      |       226.2531       |
|    66%     |  0.0161  | 0.0043  |   1.0707    |      52      |      246      |       229.2028       |
|    75%     |  0.0293  | 0.0044  |   1.1826    |      55      |      268      |       229.7484       |
|    80%     |  0.0633  | 0.0044  |   1.2737    |      58      |      290      |       230.8304       |
|    90%     |  0.0654  | 0.0046  |   1.4551    |      62      |      328      |       232.939        |
|    95%     |  0.0655  | 0.0049  |   1.4913    |      66      |      335      |       250.1984       |
|    98%     |  0.0655  | 0.0065  |   1.4913    |      66      |      335      |       250.1984       |
|    99%     |  0.0655  | 0.0072  |   1.4913    |      66      |      335      |       250.1984       |
+------------+----------+---------+-------------+--------------+---------------+----------------------+
```

You can see more usage on [EvalScope Docs](https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/parameters.html).

## 4. Run gsm8k accuracy test using EvalScope

Specify the model API service address (api_url) and API Key (api_key) to evaluate the deployed model API service. In this case, the parameter must be specified as , for example:`eval-type``service`

You should launch a model service using vllm ascend online serving.

Then, you can use the following command to evaluate the model API service:

```
evalscope eval \
 --model Qwen/Qwen2.5-7B-Instruct \
 --api-url http://localhost:8000/v1/completions \
 --api-key EMPTY \
 --eval-type service \
 --datasets gsm8k \
 --limit 10
```

Please refer to this link for [parameters]( https://evalscope.readthedocs.io/en/latest/get_started/parameters.html#parameters).

After 1-2 mins, the output is as shown below:

```shell
+-----------------------+----------------+-----------------+-----------------+---------------+-------+---------+
| Model Name            | Dataset Name   | Metric Name     | Category Name   | Subset Name   |   Num |   Score |
+=======================+================+=================+=================+===============+=======+=========+
| Qwen2.5-7B-Instruct   |  gsm8k         | AverageAccuracy | default         | main          |    10 |    0.4  |
+-----------------------+----------------+-----------------+-----------------+---------------+-------+---------+
```

You can see more usage on [EvalScope Docs](https://evalscope.readthedocs.io/en/latest/get_started/basic_usage.html#).
