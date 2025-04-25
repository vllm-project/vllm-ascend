# Performance Benchmark
This document details the benchmark methodology for vllm-ascend, aimed at evaluating the performance under a variety of workloads. To maintain alignment with vLLM, we use the [benchmark](https://github.com/vllm-project/vllm/tree/main/benchmarks) script provided by the vllm project.

**Benchmark Coverage**: We measure offline e2e latency and throughput, and fixed-QPS online serving benchmarks, for more details see [vllm-ascend benchmark scripts](https://github.com/vllm-project/vllm-ascend/tree/main/benchmarks).

## 1. Run docker container
```{code-block} bash
   :substitutions:
# Update DEVICE according to your device (/dev/davinci[0-7])
export DEVICE=/dev/davinci7
export IMAGE=m.daocloud.io/quay.io/ascend/vllm-ascend:|vllm_ascend_version|
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
/bin/bash
```

## 2. Install dependencies
```bash
cd /workspace/vllm-ascend
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install -r benchmarks/requirements-bench.txt
```

## 3. (Optional)Prepare model weights
For faster running speed, we recommend downloading the model in advance：
```bash
modelscope download --model LLM-Research/Meta-Llama-3.1-8B-Instruct
```

You can also replace all model paths in the [json](https://github.com/vllm-project/vllm-ascend/tree/main/benchmarks/tests) files with your local paths:
```bash
[
  {
    "test_name": "latency_llama8B_tp1",
    "parameters": {
      "model": "your local model path",
      "tensor_parallel_size": 1,
      "load_format": "dummy",
      "num_iters_warmup": 5,
      "num_iters": 15
    }
  }
]
```

## 4. Run benchmark script
Run benchmark script:
```bash
bash benchmarks/scripts/run-performance-benchmarks.sh
```

After about 10 mins, the output is as shown below:
```bash
online serving:
qps 1:
============ Serving Benchmark Result ============
Successful requests:                     200       
Benchmark duration (s):                  212.77    
Total input tokens:                      42659     
Total generated tokens:                  43545     
Request throughput (req/s):              0.94      
Output token throughput (tok/s):         204.66    
Total Token throughput (tok/s):          405.16    
---------------Time to First Token----------------
Mean TTFT (ms):                          104.14    
Median TTFT (ms):                        102.22    
P99 TTFT (ms):                           153.82    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          38.78     
Median TPOT (ms):                        38.70     
P99 TPOT (ms):                           48.03     
---------------Inter-token Latency----------------
Mean ITL (ms):                           38.46     
Median ITL (ms):                         36.96     
P99 ITL (ms):                            75.03     
==================================================

qps 4:
============ Serving Benchmark Result ============
Successful requests:                     200       
Benchmark duration (s):                  72.55     
Total input tokens:                      42659     
Total generated tokens:                  43545     
Request throughput (req/s):              2.76      
Output token throughput (tok/s):         600.24    
Total Token throughput (tok/s):          1188.27   
---------------Time to First Token----------------
Mean TTFT (ms):                          115.62    
Median TTFT (ms):                        109.39    
P99 TTFT (ms):                           169.03    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          51.48     
Median TPOT (ms):                        52.40     
P99 TPOT (ms):                           69.41     
---------------Inter-token Latency----------------
Mean ITL (ms):                           50.47     
Median ITL (ms):                         43.95     
P99 ITL (ms):                            130.29    
==================================================

qps 16:
============ Serving Benchmark Result ============
Successful requests:                     200       
Benchmark duration (s):                  47.82     
Total input tokens:                      42659     
Total generated tokens:                  43545     
Request throughput (req/s):              4.18      
Output token throughput (tok/s):         910.62    
Total Token throughput (tok/s):          1802.70   
---------------Time to First Token----------------
Mean TTFT (ms):                          128.50    
Median TTFT (ms):                        128.36    
P99 TTFT (ms):                           187.87    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          83.60     
Median TPOT (ms):                        77.85     
P99 TPOT (ms):                           165.90    
---------------Inter-token Latency----------------
Mean ITL (ms):                           65.72     
Median ITL (ms):                         54.84     
P99 ITL (ms):                            289.63    
==================================================

qps inf:
============ Serving Benchmark Result ============
Successful requests:                     200       
Benchmark duration (s):                  41.26     
Total input tokens:                      42659     
Total generated tokens:                  43545     
Request throughput (req/s):              4.85      
Output token throughput (tok/s):         1055.44   
Total Token throughput (tok/s):          2089.40   
---------------Time to First Token----------------
Mean TTFT (ms):                          3394.37   
Median TTFT (ms):                        3359.93   
P99 TTFT (ms):                           3540.93   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          66.28     
Median TPOT (ms):                        64.19     
P99 TPOT (ms):                           97.66     
---------------Inter-token Latency----------------
Mean ITL (ms):                           56.62     
Median ITL (ms):                         55.69     
P99 ITL (ms):                            82.90     
==================================================

offline:
latency:
Avg latency: 4.944929537673791 seconds
10% percentile latency: 4.894104263186454 seconds
25% percentile latency: 4.909652255475521 seconds
50% percentile latency: 4.932477846741676 seconds
75% percentile latency: 4.9608619548380375 seconds
90% percentile latency: 5.035418218374252 seconds
99% percentile latency: 5.052476694583893 seconds

throughput:
Throughput: 4.64 requests/s, 2000.51 total tokens/s, 1010.54 output tokens/s
Total num prompt tokens:  42659
Total num output tokens:  43545
```
The result json files are generated into the path `benchmark/results`
These files contain detailed benchmarking results for further analysis.

```bash
.
|-- latency_llama8B_tp1.json
|-- serving_llama8B_tp1_qps_1.json
|-- serving_llama8B_tp1_qps_16.json
|-- serving_llama8B_tp1_qps_4.json
|-- serving_llama8B_tp1_qps_inf.json
`-- throughput_llama8B_tp1.json
```
