# DeepSeek-V4

## Introduction

DeepSeek-V4 is introducing several key upgrades over DeepSeek-V3. (Currently, vllm-ascend temporarily only supports DeepSeek-V4-FLASH)

- The Manifold-Constrained Hyper-Connections (mHC) to strengthen conventional residual connections;
- A hybrid attention architecture, which greatly improves long-context efficiency through Compress-4-Attention and Compress-128-Attention. For the Mixture-of Experts (MoE) components, it still adopt the DeepSeekMoE architecture, with only minor adjustments.

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, single-node and multi-node deployment, accuracy and performance evaluation.

## Environment Preparation

### Model Weight
- `DeepSeek-V4-Flash-w8a8-mtp`(Quantized version): require 1 Atlas 800 A3 (128G × 8) node or 1 Atlas 800 A2 (64G × 8) node. [Download model weight](https://www.modelscope.cn/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp)

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`

### Verify Multi-node Communication(Optional)

If you want to deploy multi-node environment, you need to verify multi-node communication according to [verify multi-node communication environment](../installation.md#verify-multi-node-communication).

### Installation

You can using our official docker image to run `DeepSeek-V4` directly. Currently, `DeepSeek-V4` is integrated in image `v0.13.0rc3`.

:::::{tab-set}
:sync-group: install

::::{tab-item} A2 series
:sync: A2

Start the docker image on your each node.

```{code-block} bash
   :substitutions:

export IMAGE=quay.io/ascend/vllm-ascend:v0.13.0rc3
export NAME=vllm-ascend
docker run --rm \
    --name $NAME \
    --net=host \
    --shm-size=1g \
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
    -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /etc/hccn.conf:/etc/hccn.conf \
    -v /mnt/sfs_turbo/.cache:/root/.cache \
    -it $IMAGE bash
```

::::

::::{tab-item} A3 series
:sync: A3

Start the docker image on your each node.

```{code-block} bash
   :substitutions:

export IMAGE=quay.io/ascend/vllm-ascend:v0.13.0rc3-a3
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
    -v /mnt/sfs_turbo/.cache:/root/.cache \
    -it $IMAGE bash
```

::::

:::::

In addition, if you don't want to use the docker image as above, you can also build all from source:

- Install `vllm-ascend` from source, refer to [installation](../installation.md).
If you want to deploy multi-node environment, you need to set up environment on each node.

:::{note}
Please use the v0.13.0rc3 code to install vllm-ascend.
:::

## Deployment

:::{note}
In this tutorial, we suppose you downloaded the model weight to `/root/.cache/`. Feel free to change it to your own path.
:::

### Single-node Deployment

- `DeepSeek-V4-Flash-w8a8-mtp`: can be deployed on 1 Atlas 800 A3 (128G × 8) or 1 Atlas 800 A2 (64G × 8).

Run the following scripts on each node respectively.

:::::{tab-set}
:sync-group: install

::::{tab-item} A2 series
:sync: A2

Run the following script to execute online inference.

```shell
export USE_MULTI_BLOCK_POOL=1
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ACL_OP_INIT_MODE=1
export TRITON_ALL_BLOCKS_PARALLEL=1

vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V4-Flash-w8a8-mtp \
  --host 0.0.0.0 \
  --max_model_len 65536 \
  --max-num-batched-tokens 8192 \
  --served-model-name ds \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 16 \
  --data-parallel-size 1 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --quantization ascend \
  --port 8006 \
  --block-size 128 \
  --chat-template /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V4-Flash-w8a8-mtp/chat_template.jinja \
  --async-scheduling \
  --additional-config '{"enable_cpu_binding": "true", "multistream_overlap_shared_expert": true}' \
  --speculative-config '{"num_speculative_tokens": 1,"method": "deepseek_mtp"}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
```

::::

::::{tab-item} A3 series
:sync: A3

Run the following script to execute online inference.

```shell
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10  
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True  
export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export USE_MULTI_BLOCK_POOL=1
export HCCL_BUFFSIZE=1024
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V4-Flash-w8a8-mtp \
    --host 0.0.0.0 \
    --max_model_len 65536 \
    --max-num-batched-tokens 8192 \
    --served-model-name deepseek_v4 \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 16 \
    --data-parallel-size 2 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --quantization ascend \
    --chat-template /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V4-Flash-w8a8-mtp/chat_template.jinja \
    --port 8005 \
    --block-size 128 \
    --async-scheduling \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'\
    --speculative-config '{"num_speculative_tokens": 1,"method": "deepseek_mtp"}' \
    --additional-config '{"enable_cpu_binding": "true","multistream_overlap_shared_expert": false}'
```

::::
:::::

### Prefill-Decode Disaggregation

We'd like to show the deployment guide of DeepSeek-V4 on Atlas 800 A3 (128G × 8) multi-node environment with 2P1D for better performance.

Before you start, please

1. prepare the script `launch_online_dp.py` on each node.

    ```python
    import argparse
    import multiprocessing
    import os
    import subprocess
    import sys

    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--dp-size",
            type=int,
            required=True,
            help="Data parallel size."
        )
        parser.add_argument(
            "--tp-size",
            type=int,
            default=1,
            help="Tensor parallel size."
        )
        parser.add_argument(
            "--dp-size-local",
            type=int,
            default=-1,
            help="Local data parallel size."
        )
        parser.add_argument(
            "--dp-rank-start",
            type=int,
            default=0,
            help="Starting rank for data parallel."
        )
        parser.add_argument(
            "--dp-address",
            type=str,
            required=True,
            help="IP address for data parallel master node."
        )
        parser.add_argument(
            "--dp-rpc-port",
            type=str,
            default=12345,
            help="Port for data parallel master node."
        )
        parser.add_argument(
            "--vllm-start-port",
            type=int,
            default=9000,
            help="Starting port for the engine."
        )
        return parser.parse_args()

    args = parse_args()
    dp_size = args.dp_size
    tp_size = args.tp_size
    dp_size_local = args.dp_size_local
    if dp_size_local == -1:
        dp_size_local = dp_size
    dp_rank_start = args.dp_rank_start
    dp_address = args.dp_address
    dp_rpc_port = args.dp_rpc_port
    vllm_start_port = args.vllm_start_port

    def run_command(visiable_devices, dp_rank, vllm_engine_port):
        command = [
            "bash",
            "./run_dp_template.sh",
            visiable_devices,
            str(vllm_engine_port),
            str(dp_size),
            str(dp_rank),
            dp_address,
            dp_rpc_port,
            str(tp_size),
        ]
        subprocess.run(command, check=True)

    if __name__ == "__main__":
        template_path = "./run_dp_template.sh"
        if not os.path.exists(template_path):
            print(f"Template file {template_path} does not exist.")
            sys.exit(1)

        processes = []
        num_cards = dp_size_local * tp_size
        for i in range(dp_size_local):
            dp_rank = dp_rank_start + i
            vllm_engine_port = vllm_start_port + i
            visiable_devices = ",".join(str(x) for x in range(i * tp_size, (i + 1) * tp_size))
            process = multiprocessing.Process(target=run_command,
                                            args=(visiable_devices, dp_rank,
                                                    vllm_engine_port))
            processes.append(process)
            process.start()

        for process in processes:
            process.join()

    ```

2. prepare the script `run_dp_template.sh` on each node.

    1. Prefill node 1

        ```shell
        nic_name="xxxx" # change to your own nic name
        local_ip=xx.xx.xx.1 # change to your own ip

        export HCCL_OP_EXPANSION_MODE="AIV"

        export HCCL_IF_IP=$local_ip
        export GLOO_SOCKET_IFNAME=$nic_name
        export TP_SOCKET_IFNAME=$nic_name
        export HCCL_SOCKET_IFNAME=$nic_name

        export VLLM_RPC_TIMEOUT=3600000
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_EXEC_TIMEOUT=204
        export HCCL_CONNECT_TIMEOUT=120

        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=10
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export HCCL_BUFFSIZE=2560
        export TASK_QUEUE_ENABLE=1

        export ASCEND_BUFFER_POOL=4:8
        export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
        export USE_MULTI_BLOCK_POOL=1

        export ASCEND_RT_VISIBLE_DEVICES=$1

        vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V4-Flash-w8a8-mtp \
            --host 0.0.0.0 \
            --port $2 \
            --data-parallel-size $3 \
            --data-parallel-rank $4 \
            --data-parallel-address $5 \
            --data-parallel-rpc-port $6 \
            --tensor-parallel-size $7 \
            --enable-expert-parallel \
            --seed 1024 \
            --served-model-name deepseek_v4 \
            --max-model-len 65536 \
            --max-num-batched-tokens 8192 \
            --max-num-seqs 4 \
            --no-disable-hybrid-kv-cache-manager \
            --no-enable-prefix-caching \
            --trust-remote-code \
            --gpu-memory-utilization 0.85 \
            --quantization ascend \
            --chat-template /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V4-Flash-w8a8-mtp/chat_template.jinja \
            --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp"}' \
            --enforce-eager \
            --additional_config '{"enable_cpu_binding": "true"}' \
            --kv-transfer-config \
            '{"kv_connector": "MooncakeConnectorV1",
            "kv_role": "kv_producer",
            "kv_port": "30000",
            "engine_id": "0",
            "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector",
            "kv_connector_extra_config": {
                        "prefill": {
                                "dp_size": 16,
                                "tp_size": 1
                        },
                        "decode": {
                                "dp_size": 32,
                                "tp_size": 1
                        }
                }
            }'
        ```

    2. Prefill node 2

        ```shell
        nic_name="xxxx" # change to your own nic name
        local_ip=xx.xx.xx.2 # change to your own ip

        export HCCL_OP_EXPANSION_MODE="AIV"

        export HCCL_IF_IP=$local_ip
        export GLOO_SOCKET_IFNAME=$nic_name
        export TP_SOCKET_IFNAME=$nic_name
        export HCCL_SOCKET_IFNAME=$nic_name

        export VLLM_RPC_TIMEOUT=3600000
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_EXEC_TIMEOUT=204
        export HCCL_CONNECT_TIMEOUT=120

        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=10
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export HCCL_BUFFSIZE=2560
        export TASK_QUEUE_ENABLE=1

        export ASCEND_BUFFER_POOL=4:8
        export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
        export USE_MULTI_BLOCK_POOL=1

        export ASCEND_RT_VISIBLE_DEVICES=$1

        vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V4-Flash-w8a8-mtp \
            --host 0.0.0.0 \
            --port $2 \
            --data-parallel-size $3 \
            --data-parallel-rank $4 \
            --data-parallel-address $5 \
            --data-parallel-rpc-port $6 \
            --tensor-parallel-size $7 \
            --enable-expert-parallel \
            --seed 1024 \
            --served-model-name deepseek_v4 \
            --max-model-len 65536 \
            --max-num-batched-tokens 8192 \
            --max-num-seqs 4 \
            --no-disable-hybrid-kv-cache-manager \
            --no-enable-prefix-caching \
            --trust-remote-code \
            --gpu-memory-utilization 0.85 \
            --quantization ascend \
            --chat-template /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V4-Flash-w8a8-mtp/chat_template.jinja \
            --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp"}' \
            --enforce-eager \
            --additional_config '{"enable_cpu_binding": "true"}' \
            --kv-transfer-config \
            '{"kv_connector": "MooncakeConnectorV1",
            "kv_role": "kv_producer",
            "kv_port": "30100",
            "engine_id": "1",
            "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector",
            "kv_connector_extra_config": {
                        "prefill": {
                                "dp_size": 16,
                                "tp_size": 1
                        },
                        "decode": {
                                "dp_size": 32,
                                "tp_size": 1
                        }
                }
            }'
        ```

    3. Decode node (Same as another D node)

        ```shell
        nic_name="xxxx" # change to your own nic name
        local_ip=xx.xx.xx.xx # change to your own ip

        export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
        export HCCL_OP_EXPANSION_MODE="AIV"
        export TASK_QUEUE_ENABLE=1

        export VLLM_RPC_TIMEOUT=3600000
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_EXEC_TIMEOUT=2000
        export HCCL_CONNECT_TIMEOUT=1200

        export HCCL_IF_IP=$local_ip
        export GLOO_SOCKET_IFNAME=$nic_name
        export TP_SOCKET_IFNAME=$nic_name
        export HCCL_SOCKET_IFNAME=$nic_name

        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=10
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export HCCL_BUFFSIZE=1024
        export ASCEND_BUFFER_POOL=4:8

        export USE_MULTI_BLOCK_POOL=1
        export VLLM_ASCEND_ENABLE_FUSED_MC2=1
        export ASCEND_RT_VISIBLE_DEVICES=$1

        vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V4-Flash-w8a8-mtp \
            --host 0.0.0.0 \
            --port $2 \
            --data-parallel-size $3 \
            --data-parallel-rank $4 \
            --data-parallel-address $5 \
            --data-parallel-rpc-port $6 \
            --tensor-parallel-size $7 \
            --enable-expert-parallel \
            --seed 1024 \
            --served-model-name deepseek_v4 \
            --max-model-len 65536 \
            --max-num-batched-tokens 144 \
            --max-num-seqs 48 \
            --async-scheduling \
            --no-disable-hybrid-kv-cache-manager \
            --no-enable-prefix-caching \
            --trust-remote-code \
            --gpu-memory-utilization 0.88 \
            --quantization ascend \
            --chat-template /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V4-Flash-w8a8-mtp/chat_template.jinja \
            --speculative-config '{"num_speculative_tokens": 2, "method":"deepseek_mtp"}' \
            --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY","cudagraph_capture_sizes":[144]}' \
            --kv-transfer-config \
            '{"kv_connector": "MooncakeConnectorV1",
            "kv_role": "kv_consumer",
            "kv_port": "30200",
            "engine_id": "2",
            "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector",
            "kv_connector_extra_config": {
                        "prefill": {
                                "dp_size": 16,
                                "tp_size": 1
                        },
                        "decode": {
                                "dp_size": 32,
                                "tp_size": 1
                        }
                }
            }' \
            --additional_config '{"enable_cpu_binding": "true", "multistream_overlap_shared_expert": false, "multistream_dsa_preprocess": false}'
        ```

Once the preparation is done, you can start the server with the following command on each node:

1. Prefill node 0

```shell
# change ip to your own
python launch_online_dp.py --dp-size 16 --tp-size 1 --dp-size-local 16 --dp-rank-start 0 --dp-address xx.xx.xx.1 --dp-rpc-port 12321 --vllm-start-port 7100
```

2. Prefill node 1

```shell
# change ip to your own
python launch_online_dp.py --dp-size 16 --tp-size 1 --dp-size-local 16 --dp-rank-start 0 --dp-address xx.xx.xx.2 --dp-rpc-port 12321 --vllm-start-port 7100
```

3. Decode node 0

```shell
# change ip to your own
python launch_online_dp.py --dp-size 32 --dp-size-local 16 --dp-rank-start 0 --dp-address xx.xx.xx.3 --dp-rpc-port 12321 --vllm-start-port 7100
```

4. Decode node 1

```shell
# change ip to your own
python launch_online_dp.py --dp-size 32 --dp-size-local 16 --dp-rank-start 16 --dp-address xx.xx.xx.3 --dp-rpc-port 12321 --vllm-start-port 7100
```

Finally, Refer to [Prefill-Decode Disaggregation (Deepseek)](./pd_disaggregation_mooncake_multi_node.md) to deploy the P-D disaggregation proxy.

## Functional Verification

Once your server is started, you can query the model with input prompts:

```shell
curl http://<node0_ip>:<port>/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "deepseek_v4",
        "messages": [
            {
                "role": "user",
                "content": "Who are you?"
            }
        ],
        "max_tokens": 256,
        "temperature": 0
    }'
```

## Accuracy Evaluation

Here are two accuracy evaluation methods.

### Using AISBench

1. Refer to [Using AISBench](../developer_guide/evaluation/using_ais_bench.md) for details.

2. After execution, you can get the result.

### Using Language Model Evaluation Harness

As an example, take the `gsm8k` dataset as a test dataset, and run accuracy evaluation of `DeepSeek-V4` in online mode.

1. Refer to [Using lm_eval](../developer_guide/evaluation/using_lm_eval.md) for `lm_eval` installation.

2. Run `lm_eval` to execute the accuracy evaluation.

```shell
lm_eval \
  --model local-completions \
  --model_args model=/root/.cache/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp,base_url=http://127.0.0.1:8006/v1/completions,tokenized_requests=False,trust_remote_code=True \
  --tasks gsm8k \
  --output_path ./
```

3. After execution, you can get the result.

## Performance

### Using AISBench

Refer to [Using AISBench for performance evaluation](../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### Using vLLM Benchmark

Run performance evaluation of `DeepSeek-V4-Flash-w8a8-mtp` as an example.

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/contributing/benchmarks.html) for more details.

There are three `vllm bench` subcommand:
- `latency`: Benchmark the latency of a single batch of requests.
- `serve`: Benchmark the online serving throughput.
- `throughput`: Benchmark offline inference throughput.

Take the `serve` as an example. Run the code as follows.

```shell
export VLLM_USE_MODELSCOPE=true
vllm bench serve --model /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V4-Flash-w8a8-mtp  --dataset-name random --random-input 200 --num-prompt 200 --request-rate 1 --save-result --result-dir ./
```
