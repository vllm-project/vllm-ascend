# Long-Sequence Context Parallel (Deepseek)

## Getting Start

:::{note}
Context parallel feature currently is only supported on Atlas A3 device, and will be supported on Atlas A2 in the future.
:::

vLLM-Ascend now supports long sequence with context parallel options. This guide takes one-by-one steps to verify these features with constrained resources.

Take the Deepseek-V3.1-w8a8 model as an example, use 3 Atlas 800T A3 servers to deploy the “1P1D” architecture. Node p is deployed across multiple machines, while node d is deployed on a single machine. Assume the ip of the prefiller server is 192.0.0.1 (prefill 1) and 192.0.0.2 (prefill 2), and the decoder servers are 192.0.0.3 (decoder 1). On each server, use 8 NPUs 16 chips to deploy one service instance.In the current example, we will enable the context parallel feature on node p to improve TTFT. Although enabling the DCP feature on node d can reduce memory usage, it would introduce additional communication and small operator overhead. Therefore, we will not enable the DCP feature on node d.

## Environment Preparation

### Model Weight

- `DeepSeek-V3.1_w8a8mix_mtp`(Quantized version with mix mtp): [Download model weight](https://www.modelscope.cn/models/Eco-Tech/DeepSeek-V3.1-w8a8). Please modify `torch_dtype` from `float16` to `bfloat16` in `config.json`.

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`

### Verify Multi-node Communication

Refer to [verify multi-node communication environment](../installation.md#verify-multi-node-communication) to verify multi-node communication.

### Installation

You can using our official docker image to run `DeepSeek-V3.1` directly.

Select an image based on your machine type and start the docker image on your node, refer to [using docker](../installation.md#set-up-using-docker).

```{code-block} bash
   :substitutions:
# Update the vllm-ascend image according to your environment.
export IMAGE=m.daocloud.io/quay.io/ascend/vllm-ascend:|vllm_ascend_version|
export NAME=vllm-ascend

# Run the container using the defined variables
# Note: If you are running bridge network with docker, please expose available ports for multiple nodes communication in advance
docker run --rm \
--name $NAME \
--net=host \
--shm-size=1g \
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
-v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /root/.cache:/root/.cache \
-it $IMAGE bash
```

You need to set up environment on each node.

## Prefiller/Decoder Deployment

We can run the following scripts to launch a server on the prefiller/decoder node, respectively. Please note that each P/D node will occupy ports ranging from kv_port to kv_port + num_chips to initialize socket listeners. To avoid any issues, port conflicts should be prevented. Additionally, ensure that each node's engine_id is uniquely assigned to avoid conflicts.

1. Run the following script to execute online 128k inference on three nodes respectively.

:::::{tab-set}
:sync-group: nodes

::::{tab-item} Prefiller node 1
:sync: prefill node1

```shell
nic_name="eth0"  # network card name
local_ip="192.0.0.1"
master_addr="192.0.0.1"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export HCCL_BUFFSIZE=768
export OMP_PROC_BIND=false
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_USE_V1=1
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL=1

vllm serve /path_to_weight/DeepSeek-V3.1_w8a8mix_mtp \
  --host 0.0.0.0 \
  --port 8004 \
  --decode-context-parallel-size 8 \
  --prefill-context-parallel-size 2 \
  --cp-kv-cache-interleave-size 128 \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --quantization ascend \
  --enforce-eager \
  --served-model-name deepseek_v3 \
  --seed 1024 \
  --no-enable-chunked-prefill \
  --no-enable-prefix-caching \
  --max-num-seqs 1 \
  --max-model-len 136000 \
  --max-num-batched-tokens 136000 \
  --block-size 128 \
  --trust-remote-code \
  --gpu-memory-utilization 0.8 \
  --nnodes 2 \
  --node-rank 0 \
  --master-addr $master_addr \
  --master-port 7001 \
  --speculative-config '{"num_speculative_tokens": 3, "method":"deepseek_mtp"}' \
  --kv-transfer-config \
  '{"kv_connector": "MooncakeConnectorV1",
  "kv_role": "kv_producer",
  "kv_port": "30000",
  "engine_id": "0",
  "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector",
  "kv_connector_extra_config": {
            "use_ascend_direct": true,
            "prefill": {
                    "dp_size": 1,
                    "tp_size": 16
             },
             "decode": {
                    "dp_size": 1,
                    "tp_size": 16
             }
      }
  }'
```

::::

::::{tab-item} Prefiller node 2
:sync: prefill node2

```shell
nic_name="eth0"  # network card name
local_ip="192.0.0.2"
master_addr="192.0.0.1"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export HCCL_BUFFSIZE=768
export OMP_PROC_BIND=false
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_USE_V1=1
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL=1

vllm serve /path_to_weight/DeepSeek-V3.1_w8a8mix_mtp \
  --host 0.0.0.0 \
  --port 8004 \
  --decode-context-parallel-size 8 \
  --prefill-context-parallel-size 2 \
  --cp-kv-cache-interleave-size 128 \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --quantization ascend \
  --enforce-eager \
  --served-model-name deepseek_v3 \
  --seed 1024 \
  --no-enable-chunked-prefill \
  --no-enable-prefix-caching \
  --max-num-seqs 1 \
  --max-model-len 136000 \
  --max-num-batched-tokens 136000 \
  --block-size 128 \
  --trust-remote-code \
  --gpu-memory-utilization 0.8 \
  --nnodes 2 \
  --node-rank 1 \
  --headless \
  --master-addr $master_addr \
  --master-port 7001 \
  --speculative-config '{"num_speculative_tokens": 3, "method":"deepseek_mtp"}' \
  --kv-transfer-config \
  '{"kv_connector": "MooncakeConnectorV1",
  "kv_role": "kv_producer",
  "kv_port": "30000",
  "engine_id": "1",
  "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector",
  "kv_connector_extra_config": {
            "use_ascend_direct": true,
            "prefill": {
                    "dp_size": 1,
                    "tp_size": 16
             },
             "decode": {
                    "dp_size": 1,
                    "tp_size": 16
             }
      }
  }'
```

::::

::::{tab-item} Decoder node 1
:sync: decoder node1

```shell
nic_name="eth0"  # network card name
local_ip="192.0.0.3"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export HCCL_BUFFSIZE=768
export OMP_PROC_BIND=false
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_USE_V1=1
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_MLAPO="1"
export VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL=1

vllm serve /path_to_weight/DeepSeek-V3.1_w8a8mix_mtp \
  --host 0.0.0.0 \
  --port 8004 \
  --api-server-count 1 \
  --data-parallel-size 1 \
  --data-parallel-size-local 1 \
  --data-parallel-start-rank 0 \
  --data-parallel-address $local_ip \
  --data-parallel-rpc-port 5980  \
  --decode-context-parallel-size 1 \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --quantization ascend \
  --no-enable-prefix-caching \
  --distributed-executor-backend mp \
  --served-model-name deepseek_v3 \
  --seed 1024 \
  --max-model-len 136000 \
  --max-num-batched-tokens 128 \
  --enable-chunked-prefill \
  --max-num-seqs 4 \
  --trust-remote-code \
  --gpu-memory-utilization 0.96 \
  --speculative-config '{"num_speculative_tokens": 3, "method":"deepseek_mtp"}' \
  --compilation_config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes":[1,2,4]}' \
  --kv-transfer-config \
  '{"kv_connector": "MooncakeConnectorV1",
  "kv_role": "kv_consumer",
  "kv_port": "30200",
  "engine_id": "3",
  "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector",
  "kv_connector_extra_config": {
            "prefill": {
                    "dp_size": 1,
                    "tp_size": 16
             },
             "decode": {
                    "dp_size": 1,
                    "tp_size": 16
             }
      }
  }'
```

::::

:::::

2. Prefill master node `proxy.sh` scripts

```shell
python load_balance_proxy_server_example.py \
  --port 8005 \
  --host 192.0.0.1 \
  --prefiller-hosts \
    192.0.0.1 \
  --prefiller-ports \
    8004 \
  --decoder-hosts \
    192.0.0.3 \
  --decoder-ports \
    8004
```

3. run proxy

Run a proxy server on the same node with the prefiller service instance. You can get the proxy program in the repository's examples: [load\_balance\_proxy\_server\_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py)

```shell
cd vllm-ascend/examples/disaggregated_prefill_v1/
bash proxy.sh
```

**Notice:**
The parameters are explained as follows:
- `--tensor-parallel-size` 16 are common settings for tensor parallelism (TP) sizes.
- `--prefill-context-parallel-size` 2 are common settings for prefill context parallelism (PCP) sizes.
- `--decode-context-parallel-size` 8 are common settings for decode context parallelism (DCP) sizes.
- `--max-model-len` represents the context length, which is the maximum value of the input plus output for a single request.
- `--max-num-seqs` indicates the maximum number of requests that each DP group is allowed to process. If the number of requests sent to the service exceeds this limit, the excess requests will remain in a waiting state and will not be scheduled. Note that the time spent in the waiting state is also counted in metrics such as TTFT and TPOT. Therefore, when testing performance, it is generally recommended that `--max-num-seqs` * `--data-parallel-size` >= the actual total concurrency.
- `--max-num-batched-tokens` represents the maximum number of tokens that the model can process in a single step. Currently, vLLM v1 scheduling enables ChunkPrefill/SplitFuse by default, which means:
  - (1) If the input length of a request is greater than `--max-num-batched-tokens`, it will be divided into multiple rounds of computation according to `--max-num-batched-tokens`;
  - (2) Decode requests are prioritized for scheduling, and prefill requests are scheduled only if there is available capacity.
  - Generally, if `--max-num-batched-tokens` is set to a larger value, the overall latency will be lower, but the pressure on GPU memory (activation value usage) will be greater.
- `--gpu-memory-utilization` represents the proportion of HBM that vLLM will use for actual inference. Its essential function is to calculate the available kv_cache size. During the warm-up phase (referred to as profile run in vLLM), vLLM records the peak GPU memory usage during an inference process with an input size of `--max-num-batched-tokens`. The available kv_cache size is then calculated as: `--gpu-memory-utilization` * HBM size - peak GPU memory usage. Therefore, the larger the value of `--gpu-memory-utilization`, the more kv_cache can be used. However, since the GPU memory usage during the warm-up phase may differ from that during actual inference (e.g., due to uneven EP load), setting `--gpu-memory-utilization` too high may lead to OOM (Out of Memory) issues during actual inference. The default value is `0.9`.
- `--enable-expert-parallel` indicates that EP is enabled. Note that vLLM does not support a mixed approach of ETP and EP; that is, MoE can either use pure EP or pure TP.
- `--no-enable-prefix-caching` indicates that prefix caching is disabled. To enable it, remove this option.
- `--quantization` "ascend" indicates that quantization is used. To disable quantization, remove this option.
- `--compilation-config` contains configurations related to the aclgraph graph mode. The most significant configurations are "cudagraph_mode" and "cudagraph_capture_sizes", which have the following meanings:
"cudagraph_mode": represents the specific graph mode. Currently, "PIECEWISE" and "FULL_DECODE_ONLY" are supported. The graph mode is mainly used to reduce the cost of operator dispatch. Currently, "FULL_DECODE_ONLY" is recommended.
- "cudagraph_capture_sizes": represents different levels of graph modes. The default value is [1, 2, 4, 8, 16, 24, 32, 40,..., `--max-num-seqs`]. In the graph mode, the input for graphs at different levels is fixed, and inputs between levels are automatically padded to the next level. Currently, the default setting is recommended. Only in some scenarios is it necessary to set this separately to achieve optimal performance.
- `export VLLM_ASCEND_ENABLE_FLASHCOMM1=1` indicates that Flashcomm1 optimization is enabled. Currently, this optimization is only supported for MoE in scenarios where tensor-parallel-size > 1.
- `export VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL=1` indicates that context parallel is enabled. This environment variable is required in the PD architecture but not needed in the pd co-locate deployment scenario. It will be removed in the future.

**Notice:**
- tensor-parallel-size needs to be divisible by decode-context-parallel-size.
- decode-context-parallel-size must less than or equal to tensor-parallel-size.

## Accuracy Evaluation

Here are two accuracy evaluation methods.

### Using AISBench

1. Refer to [Using AISBench](../developer_guide/evaluation/using_ais_bench.md) for details.

2. After execution, you can get the result, here is the result of `DeepSeek-V3.1-w8a8` for reference only.

| dataset  | version | metric | mode | vllm-api-general-chat |
|----------| ----- | ----- | ----- |-----------------------|
| aime2024 | - | accuracy | gen | 86.67 |

## Performance

### Using AISBench

Refer to [Using AISBench for performance evaluation](../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### Using vLLM Benchmark

Run performance evaluation of `DeepSeek-V3.1-w8a8` as an example.

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/contributing/benchmarks.html) for more details.

There are three `vllm bench` subcommand:
- `latency`: Benchmark the latency of a single batch of requests.
- `serve`: Benchmark the online serving throughput.
- `throughput`: Benchmark offline inference throughput.

Take the `serve` as an example. Run the code as follows.

```shell
export VLLM_USE_MODELSCOPE=true
vllm bench serve --model /path_to_weight/DeepSeek-V3.1_w8a8mix_mtp  --dataset-name random --random-input 131072 --num-prompt 20 --request-rate 0 --save-result --result-dir ./
```

After about several minutes, you can get the performance evaluation result.

| dataset | version | metric      | mode | ttft   |
|---------| ----- |-------------|------|--------|
| random  | - | performance | perf | 20.7s |
