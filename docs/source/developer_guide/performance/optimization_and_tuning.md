# Optimization and Tuning

## Motivation

To achieve ultimate performance on vllm-asend `v0.7.3` with mindie-turbo `2.0rc1`, we have made efforts to optimize our compilation, environment variables, application configs, etc.

## Environment Preparation

```bash
# Install necessary dependencies
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install mindie-turbo pandas datasets gevent sacrebleu rouge_score pybind11 pytest

export HF_ENDPOINT="https://hf-mirror.com"
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export VLLM_USE_V1=1
```

## Optimizations

### 1. Compiler Optimization

Install compiled torch and torch-npu package:

```bash
cd /tmp/
wget https://repo.oepkgs.net/ascend/pytorch/vllm/lib/libomp.so
mv /tmp/*.so* /usr/local/lib

wget https://repo.oepkgs.net/ascend/pytorch/vllm/torch/torch-2.5.1-cp310-cp310-linux_aarch64.whl
pip install /tmp/torch-2.5.1*.whl --force-reinstall --no-deps

wget https://repo.oepkgs.net/ascend/pytorch/vllm/torch/torch_npu-2.5.1-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
pip install /tmp/torch_npu-*.whl --force-reinstall --no-deps

pip cache purge
rm -rf /tmp/*
```

:::{note}
You can also reproduce the torch build follow this [tutorial](https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/performance_tuning_0064.html) or reproduce the torch-npu build follow this [tutorial](https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/performance_tuning_0065.html).
:::

### 2. OS Optimization

Install `tcmalloc`:

```bash
sudo apt update
sudo apt install libgoogle-perftools4 libgoogle-perftools-dev
```

Get location of `libtcmalloc.so*`:

```bash
find /usr -name libtcmalloc.so*
```

Make the priority of `tcmalloc` higher:

```bash
export LD_PRELOAD="$LD_PRELOAD:<the location of libtcmalloc.so>"
# For example: export LD_PRELOAD="$LD_PRELOAD:/usr/lib/aarch64-linux-gnu/libtcmalloc.so"
```

Verify your configuration:

```bash
ldd `which python`
```

The path of `libtcmalloc.so` will be contained in the result if your configuration is enabled.

Find more details [here](https://www.hiascend.com/document/detail/zh/Pytorch/700/ptmoddevg/trainingmigrguide/performance_tuning_0068.html).

### 3. torch-npu Optimization

Memory optimization:

```bash
# Upper limit of memory block splitting allowed (MB), Setting this parameter can prevent large memory blocks from being split.
export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:250"

# When operators on the communication stream have dependencies, they all need to be ended before being released for reuse. The logic of multi-stream reuse is to release the memory on the communication stream in advance so that the computing stream can be reused.
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
```

Schedule optimization:

```bash
# Optimize operator delivery queue, this will affect the memory peak value, and may degrade if the memory is tight.
export TASK_QUEUE_ENABLE=2

# This will greatly improve the CPU bottleneck model and ensure the same performance for the NPU bottleneck model.
export CPU_AFFINITY_CONF=1
```

### 4. CANN Optimization

#### 4.1 HCCL Optimization

These environment variables will improve performance in certain scenarios:

- [HCCL_INTRA_ROCE_ENABLE](https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/performance_tuning_0044.html)
- [HCCL_RDMA_TC](https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/performance_tuning_0045.html)
- [HCCL_RDMA_SL](https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/performance_tuning_0046.html)
- [HCCL_BUFFSIZE](https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/performance_tuning_0047.html)

Find more details [here](https://www.hiascend.com).

#### 4.2 mindie-turbo Optimization

Some environment variables will improve performance in certain scenarios.

Find more details [here](https://www.hiascend.com/document/detail/zh/mindie/20RC1/AcceleratePlugin/turbodev/mindie-turbo-0010.html).

## Benchmark

### Usage

Launch vllm server:

```bash
python -m vllm.entrypoints.openai.api_server \
--model Qwen/Qwen2.5-7B-Instruct \
--tensor-parallel-size 1 \
--swap-space 16 \
--disable-log-stats \
--disable-log-requests \
--load-format dummy \
--additional-config '{"ascend_scheduler_config":{}}'
```

:::{note}
Set `load-format=dummy` for a lightweight test, we don't need real download weight.

You can pass `--additional-config '{"ascend_scheduler_config":{}}'` param to vllm when launch the server with ascend scheduler, which can accelerate the inference for V1 engine. Find more details [here](https://github.com/vllm-project/vllm-ascend/issues/788).
:::

Run benchmark for online serving (need wait for vllm serving ready):

```bash
cd /vllm-workspace/vllm/benchmarks
python benchmark_serving.py \
--model Qwen/Qwen2.5-7B-Instruct \
--dataset-name random \
--random-input-len 200 \
--num-prompts 200 \
--request-rate 1 \
--save-result --result-dir ./
```

### Result Comparison

Before optimization:

```bash
============ Serving Benchmark Result ============
Successful requests:                     200       
Benchmark duration (s):                  188.33    
Total input tokens:                      40000     
Total generated tokens:                  25600     
Request throughput (req/s):              1.06      
Output token throughput (tok/s):         135.93    
Total Token throughput (tok/s):          348.33    
---------------Time to First Token----------------
Mean TTFT (ms):                          51.93     
Median TTFT (ms):                        53.35     
P99 TTFT (ms):                           67.70     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          27.56     
Median TPOT (ms):                        27.51     
P99 TPOT (ms):                           28.86     
---------------Inter-token Latency----------------
Mean ITL (ms):                           27.56     
Median ITL (ms):                         27.45     
P99 ITL (ms):                            31.65     
==================================================
```

After optimization:

```bash
============ Serving Benchmark Result ============
Successful requests:                     200       
Benchmark duration (s):                  187.85    
Total input tokens:                      40000     
Total generated tokens:                  25600     
Request throughput (req/s):              1.06      
Output token throughput (tok/s):         136.28    
Total Token throughput (tok/s):          349.21    
---------------Time to First Token----------------
Mean TTFT (ms):                          44.92     
Median TTFT (ms):                        44.53     
P99 TTFT (ms):                           64.20     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          24.70     
Median TPOT (ms):                        24.80     
P99 TPOT (ms):                           26.03     
---------------Inter-token Latency----------------
Mean ITL (ms):                           24.70     
Median ITL (ms):                         24.04     
P99 ITL (ms):                            50.68     
==================================================
```

Summary: The TTFT is reduced by approximately **13.5%**, the TPOT and ITL is reduced by approximately **10.4%**.
