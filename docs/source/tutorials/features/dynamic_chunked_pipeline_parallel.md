# Dynamic Chunked Pipeline Parallel (DeepSeek-V3.1)

## Getting Started

vLLM-Ascend supports Dynamic Chunked Pipeline Parallel (CPP) for optimizing prefill performance in Pipeline Parallelism scenarios. This feature is supported starting from version `v0.19.1rc1`.

!!! important

    **CPP is designed to be used on the P (Prefiller) node in Prefill-Decode (PD) disaggregation deployments.** The D (Decoder) node does not require CPP configuration. By dynamically calculating the optimal chunk size based on profiling data, CPP significantly reduces Time-To-First-Token (TTFT) for long sequences on P nodes.

    This guide demonstrates PD disaggregation deployment with DeepSeek-V3.1 on 3 Atlas 800T A3 servers (64GB × 16): one server acts as the Prefiller (P node, with CPP enabled) and two servers act as the Decoder (D nodes, without CPP, DP32 standard configuration).

For configuration details, see the [Feature Guide](../../user_guide/feature_guide/dynamic_chunk_pipeline_parallel.md).

For design details, see the [Design Document](../../developer_guide/Design_Documents/dynamic_chunked_pipeline_parallel.md).

For complete PD disaggregation setup instructions (environment verification, Mooncake installation, proxy deployment), see:

- [PD Disaggregation Single Node (Qwen2.5-VL)](pd_disaggregation_mooncake_single_node.md)
- [PD Disaggregation Multi Node (DeepSeek)](pd_disaggregation_mooncake_multi_node.md)

## Environment Preparation

### Model Weight

- `DeepSeek-V3.1-W8A8` (Quantized version): 1 Atlas 800T A3 (64GB × 16) node

Download to shared directory such as `/mnt/weight/`

### Run with Docker

Start a Docker container on each node.

```bash
export IMAGE=m.daocloud.io/quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
export NAME=vllm-ascend

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
--device /dev/davinci8 \
--device /dev/davinci9 \
--device /dev/davinci10 \
--device /dev/davinci11 \
--device /dev/davinci12 \
--device /dev/davinci13 \
--device /dev/davinci14 \
--device /dev/davinci15 \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /etc/hccn.conf:/etc/hccn.conf \
-v /mnt/weight:/mnt/weight \
-it $IMAGE bash
```

### Install Mooncake

Mooncake is required for PD disaggregation KV cache transfer. Refer to [PD Disaggregation Multi Node - Install Mooncake](pd_disaggregation_mooncake_multi_node.md#install-mooncake) for installation and compilation steps.

## Deployment

!!! important

    In a PD disaggregation setup, enable CPP **only on the P (Prefiller) node**. The D (Decoder) node runs without pipeline parallelism and focuses on low-latency token-by-token decoding.

    - It is recommended to use `MooncakeConnectorV1` as the `kv_connector`, as it provides more comprehensive support for PP.
    - It is recommended **not** to enable `async-scheduling` on P nodes of PP, as it may cause performance degradation in the prefill stage.

Assume the P server IP is `192.0.0.1`, and the D server IPs are `192.0.0.3` (decoder 1) and `192.0.0.4` (decoder 2).

=== "P Node (Prefiller — with CPP)"

    ```shell
    #!/bin/sh
    unset https_proxy
    unset http_proxy

    # For nic_name, run the `ifconfig` command to check the network adapter whose IP address is the same as that of the local host.
    nic_name="eth0"
    local_ip="192.0.0.1"
    export HCCL_IF_IP=$local_ip
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export HCCL_SOCKET_IFNAME=$nic_name

    export OMP_PROC_BIND=false
    export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
    export OMP_NUM_THREADS=1
    export HCCL_BUFFSIZE=2048
    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
    export HCCL_OP_EXPANSION_MODE="AIV"
    export VLLM_USE_V1=1
    export TASK_QUEUE_ENABLE=1
    export ASCEND_LAUNCH_BLOCKING=0
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
    export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
    export VLLM_RPC_TIMEOUT=3600000
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
    export HCCL_EXEC_TIMEOUT=204
    export HCCL_CONNECT_TIMEOUT=120

    vllm serve /mnt/weight/DeepSeek-V3.1-w8a8 \
      --host 0.0.0.0 \
      --port 8003 \
      --served-model-name model \
      --tensor-parallel-size 8 \
      --pipeline-parallel-size 2 \
      --enable-expert-parallel \
      --max-num-seqs 32 \
      --max-model-len 131072 \
      --max-num-batched-tokens 32768 \
      --gpu-memory-utilization 0.9 \
      --enable-chunked-prefill \
      --enable-prefix-caching \
      --no-async-scheduling \
      --trust-remote-code \
      --quantization ascend \
      --additional-config '{
        "scheduler_config": {
          "profiling_chunk_config": {"enabled":true, "smooth_factor":1.0, "min_chunk":4096}
        }
      }' \
      --kv-transfer-config \
      '{
        "kv_connector": "MooncakeConnectorV1",
        "kv_role": "kv_producer",
        "kv_port": "36000",
        "engine_id": "0",
        "kv_connector_extra_config": {
          "prefill": {
            "pp_size": 2,
            "dp_size": 1,
            "tp_size": 8
          },
          "decode": {
            "dp_size": 32,
            "tp_size": 1
          }
        }
      }'
    ```

    The server has started successfully if the log outputs: `vLLM API server started on 0.0.0.0:8003`.

=== "D Node (Decoder — without CPP)"

    The D nodes use `launch_online_dp.py` to start multiple DP instances per node. Obtain the launcher and template scripts from the repository:

    - [launch_online_dp.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/external_online_dp/launch_online_dp.py)
    - [run_dp_template.sh](https://github.com/vllm-project/vllm-ascend/blob/main/examples/external_online_dp/run_dp_template.sh)

    Modify `run_dp_template.sh` on each D node. The launcher passes seven positional arguments (`$1`–`$7`) to the template. For a full explanation of each parameter, refer to [PD Disaggregation Multi Node](pd_disaggregation_mooncake_multi_node.md#run_dp_template-sh).

    === "Decoder node 1 (192.0.0.3)"

        ```shell
        #!/bin/sh
        unset https_proxy
        unset http_proxy

        nic_name="eth0"  # network card name
        local_ip="192.0.0.3"
        export HCCL_IF_IP=$local_ip
        export GLOO_SOCKET_IFNAME=$nic_name
        export TP_SOCKET_IFNAME=$nic_name
        export HCCL_SOCKET_IFNAME=$nic_name
        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=10
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export HCCL_BUFFSIZE=600
        export TASK_QUEUE_ENABLE=1
        export HCCL_OP_EXPANSION_MODE="AIV"
        export VLLM_USE_V1=1
        export ASCEND_RT_VISIBLE_DEVICES=$1
        vllm serve /mnt/weight/DeepSeek-V3.1-w8a8 \
          --host 0.0.0.0 \
          --port $2 \
          --data-parallel-size $3 \
          --data-parallel-rank $4 \
          --data-parallel-address $5 \
          --data-parallel-rpc-port $6 \
          --tensor-parallel-size $7 \
          --enable-expert-parallel \
          --seed 1024 \
          --served-model-name model \
          --max-model-len 131072 \
          --max-num-batched-tokens 256 \
          --max-num-seqs 40 \
          --trust-remote-code \
          --gpu-memory-utilization 0.94 \
          --quantization ascend \
          --no-enable-prefix-caching \
          --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
          --additional-config '{
            "scheduler_config": {"recompute_scheduler_enable": true},
            "multistream_overlap_shared_expert": true,
            "finegrained_tp_config": {"lmhead_tensor_parallel_size": 16}
          }' \
          --kv-transfer-config \
          '{
            "kv_connector": "MooncakeConnectorV1",
            "kv_role": "kv_consumer",
            "kv_port": "36000",
            "engine_id": "0",
            "kv_connector_extra_config": {
              "prefill": {
                "pp_size": 2,
                "dp_size": 1,
                "tp_size": 8
              },
              "decode": {
                "dp_size": 32,
                "tp_size": 1
              }
            }
          }'
        ```

    === "Decoder node 2 (192.0.0.4)"

        ```shell
        #!/bin/sh
        unset https_proxy
        unset http_proxy

        nic_name="eth0"  # network card name
        local_ip="192.0.0.4"
        export HCCL_IF_IP=$local_ip
        export GLOO_SOCKET_IFNAME=$nic_name
        export TP_SOCKET_IFNAME=$nic_name
        export HCCL_SOCKET_IFNAME=$nic_name
        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=10
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export HCCL_BUFFSIZE=600
        export TASK_QUEUE_ENABLE=1
        export HCCL_OP_EXPANSION_MODE="AIV"
        export VLLM_USE_V1=1
        export ASCEND_RT_VISIBLE_DEVICES=$1
        vllm serve /mnt/weight/DeepSeek-V3.1-w8a8 \
          --host 0.0.0.0 \
          --port $2 \
          --data-parallel-size $3 \
          --data-parallel-rank $4 \
          --data-parallel-address $5 \
          --data-parallel-rpc-port $6 \
          --tensor-parallel-size $7 \
          --enable-expert-parallel \
          --seed 1024 \
          --served-model-name model \
          --max-model-len 131072 \
          --max-num-batched-tokens 256 \
          --max-num-seqs 40 \
          --trust-remote-code \
          --gpu-memory-utilization 0.94 \
          --quantization ascend \
          --no-enable-prefix-caching \
          --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
          --additional-config '{
            "scheduler_config": {"recompute_scheduler_enable": true},
            "multistream_overlap_shared_expert": true,
            "finegrained_tp_config": {"lmhead_tensor_parallel_size": 16}
          }' \
          --kv-transfer-config \
          '{
            "kv_connector": "MooncakeConnectorV1",
            "kv_role": "kv_consumer",
            "kv_port": "36000",
            "engine_id": "0",
            "kv_connector_extra_config": {
              "prefill": {
                "pp_size": 2,
                "dp_size": 1,
                "tp_size": 8
              },
              "decode": {
                "dp_size": 32,
                "tp_size": 1
              }
            }
          }'
        ```

    Start the service on each D node:

    ```bash
    # on 192.0.0.3 (decoder 1, DP rank 0–15)
    python launch_online_dp.py --dp-size 32 --tp-size 1 --dp-size-local 16 --dp-rank-start 0 --dp-address 192.0.0.3 --dp-rpc-port 12321 --vllm-start-port 7100
    # on 192.0.0.4 (decoder 2, DP rank 16–31)
    python launch_online_dp.py --dp-size 32 --tp-size 1 --dp-size-local 16 --dp-rank-start 16 --dp-address 192.0.0.3 --dp-rpc-port 12321 --vllm-start-port 7100
    ```

=== "Example Proxy for Deployment"

    Run a proxy server on the same node with the prefiller service instance. You can get the proxy program in the repository's examples: [load\_balance\_proxy\_server\_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py)

    ```shell
    python load_balance_proxy_server_example.py \
        --host 192.0.0.1 \
        --port 8080 \
        --prefiller-hosts 192.0.0.1 \
        --prefiller-port 8003 \
        --decoder-hosts \
          192.0.0.3 192.0.0.3 192.0.0.3 192.0.0.3 192.0.0.3 192.0.0.3 192.0.0.3 192.0.0.3 \
          192.0.0.3 192.0.0.3 192.0.0.3 192.0.0.3 192.0.0.3 192.0.0.3 192.0.0.3 192.0.0.3 \
          192.0.0.4 192.0.0.4 192.0.0.4 192.0.0.4 192.0.0.4 192.0.0.4 192.0.0.4 192.0.0.4 \
          192.0.0.4 192.0.0.4 192.0.0.4 192.0.0.4 192.0.0.4 192.0.0.4 192.0.0.4 192.0.0.4 \
        --decoder-ports \
          7100 7101 7102 7103 7104 7105 7106 7107 7108 7109 7110 7111 7112 7113 7114 7115 \
          7100 7101 7102 7103 7104 7105 7106 7107 7108 7109 7110 7111 7112 7113 7114 7115
    ```

    | Parameter | Meaning |
    | --- | --- |
    | --port | Port of proxy |
    | --prefiller-port | All ports of prefill |
    | --decoder-ports | All ports of decoder |

=== "Verification"

    Check service health using the proxy server endpoint.

    ```shell
    curl http://192.0.0.1:8080/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "model",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a useful AI assistant."
                },
                {
                    "role": "user",
                    "content": "Question: Janet'\''s ducks lay 16 eggs per day. She eats three for breakfast and bakes muffins with four. She sells the remainder for $2 each. How much does she make?\nAnswer:"
                }
            ],
            "max_completion_tokens": 100,
            "temperature": 0
        }'
    ```

    To verify CPP is active on the P node, check the startup logs for profiling-related messages such as `profiling_chunk_config` enabled and chunk size calculation outputs.

> **Key Parameters**
>
> - `--pipeline-parallel-size 2`: Enables Pipeline Parallelism (required, P node only)
> - `--enable-chunked-prefill`: Enables Chunked Prefill (required, P node only)
> - `--max-num-batched-tokens 32768`: Initial chunk size (recommended for 128K sequences)
> - `profiling_chunk_config.enabled`: Enables Dynamic Chunked Pipeline Parallel
> - `profiling_chunk_config.smooth_factor`:  Smoothing factor (0 < x ≤ 1.0). Higher values trust dynamic prediction more
> - `profiling_chunk_config.min_chunk`: Minimum chunk size for dynamic calculation. Should be smaller than `max-num-batched-tokens`
> - `profiling_chunk_config.need_timing`: Enable/disable Online Calibration
> - `profiling_chunk_config.max_fit_chunk`: Number of chunk-time data for Online Calibration. Should be more when profiling failed
>
> **Key points for PD disaggregation with CPP:**
>
> - CPP (`profiling_chunk_config.enabled`, `--pipeline-parallel-size > 1`) is configured **only on the P node**.
> - The D nodes run without pipeline parallelism (DP32 TP1 standard configuration) — they focus on low-latency token-by-token decoding.
> - For complete PD disaggregation setup instructions (environment verification, Mooncake installation, proxy deployment), see:
>     - [PD Disaggregation Single Node](pd_disaggregation_mooncake_single_node.md)
>     - [PD Disaggregation Multi Node](pd_disaggregation_mooncake_multi_node.md)

For configuration details, see the [Feature Guide](../../user_guide/feature_guide/dynamic_chunk_pipeline_parallel.md).

## Online Calibration

For optimal performance, online calibrate with real data before production:

You can use aisbench to generate fixed-length random datasets. Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

1. Modify `<YOUR_AISBENCH_PATH>/benchmark/ais_bench/datasets/synthetic/synthetic_config.py`:

    ```python
    synthetic_config = {
        "Type": "string",
        "RequestCount": 5,
        "TrustRemoteCode": False,
        "StringConfig": {
            "Input": {
                "Method": "uniform",
                "Params": {"MinValue": 131072, "MaxValue": 131072}  # Your max sequence length, max-model-len
            },
            "Output": {
                "Method": "uniform",
                "Params": {"MinValue": 1, "MaxValue": 1}
            }
        },
    }
    ```

2. Run for online calibration:

    ```bash
    ais_bench --models vllm_api_stream_chat --datasets synthetic_gen --mode perf --debug
    ```

Configure online calibration data length to match your `max-model-len`. Use `batch_size=1` and ensure data differs to avoid cache hits if prefix caching is enabled.

## Accuracy Evaluation

Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details.

| dataset | accuracy |
|---------|----------|
| gsm8k   | 95.83    |

## Performance Benchmark

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

To evaluate the effectiveness of Dynamic Chunked Pipeline Parallel in PD disaggregation long sequence LLM inference scenarios, we use **DeepSeek-V3.1-W8A8** and **Qwen3-235B**, deploy prefill instance in Ascend Atlas 800T A3 server (64GB × 16), the configuration and performance data are as follows.

**Fixed-length requests, concurrency=1**:

- DeepSeek-V3.1-W8A8:

    | Configuration | CPP <br> (Dynamic Chunk, <br> chunksize=32k) | PP <br>(Static Chunk, <br> chunksize=32k) |
    | ----------------------------- | ------------------------- | ------------------------- |
    | Input length 128k | TTFT: 22.5s | TTFT: 27.0s |

- Qwen3-235B:

    | Configuration | CPP <br> (Dynamic Chunk, <br> chunksize=32k) | PP <br>(Static Chunk, <br> chunksize=32k) |
    | ----------------------------- | ------------------------- | ------------------------- |
    | Input length 256k | TTFT: 53.5s | TTFT: 61.4s |

**Variable-length requests, concurrency=4**:

- DeepSeek-V3.1-W8A8:

    | Configuration | 4k~64k Input, mean=32k, std=32k <br> prefix hit rate=99% |
    | ----------------------------- | ------------------------- |
    | CPP2TP8 | Input throughput: 22424 tps/card |
    | DP2TP8 | Input throughput: 16150 tps/card |
    | TP16 | Input throughput: 18875 tps/card |
