# MiniMax-M3

## 1 Introduction

MiniMax-M3 is a multimodal large language model that supports text, image, and video inputs. On Ascend, it supports BF16 and W8A8 on A2/A3, Prefill-Decode disaggregation on Atlas 800 A3 (BF16) and 950DT products (MXFP8), thinking mode, reasoning parsing, tool-call parsing, and multimodal inputs.

This document covers supported features, environment and model preparation, single-node deployment, multi-node deployment, PD separation, thinking and parser configuration, functional verification, accuracy evaluation, and troubleshooting.

This document is written based on the vLLM-Ascend v0.27.1 release. This model is supported in this release.

## 2 Supported Features

Refer to [Supported Features List](../../user_guide/support_matrix/supported_models.md) for the model support matrix.

Refer to the [Feature Guide](../../user_guide/feature_guide/index.md) for feature configuration instructions.

## 3 Prerequisites

### 3.1 Model Weight

- `MiniMax-M3` (BF16): requires 16 × 64 GB NPU chips. Prefill-Decode disaggregation uses 2 Atlas 800 A3 (64GB × 16). [Download the model weights](https://www.modelscope.cn/collections/MiniMax/MiniMax-M3).
- `MiniMax-M3-w8a8` (W8A8): requires at least 8 × 64 GB NPU chips. Recommended for Atlas 800 A3 (64GB × 16) and Atlas 800 A2 (64GB × 8). [Download the model weights](https://www.modelscope.cn/models/Eco-Tech/MiniMax-M3-w8a8-0626).
- `MiniMax-M3-MXFP8` (MXFP8): used for 950DT products (96GB × 8) PD disaggregation (2 nodes, 1P1D). [Download the model weights](https://huggingface.co/MiniMaxAI/MiniMax-M3-MXFP8).
- `MiniMax-M3-EAGLE3`: EAGLE3 draft model used as the draft model for speculative decoding (eagle3 method) to accelerate generation. [Download the model weights](https://www.modelscope.cn/models/Inferact/MiniMax-M3-EAGLE3).

It is recommended to place the model weight in a shared cache directory.

### 3.2 Verify Multi-node Communication (Optional)

For multi-node deployment, verify the communication environment by following [Verify Multi-node Communication Environment](../../getting_started/installation.md#installation-multi-node-interconnect).

## 4 Installation

### 4.1 Docker Image Installation

You can use the official all-in-one Docker image. For the available image tags and published versions, refer to [Using Docker](../../getting_started/installation.md#installation-prebuilt-image).

- Step 1: Download the latest Docker image

  ```bash
  docker pull quay.io/ascend/vllm-ascend:{tag}
  ```

- Step 2: Start Docker container

  ```bash
  # Set the vLLM Ascend image name.
  export IMAGE=quay.io/ascend/vllm-ascend:{tag}
  export NAME=minimax-m3-dev

  # Start the container with the variables defined above.
  # Update --device for your hardware (Atlas A3: /dev/davinci[0-15]; Atlas A2: /dev/davinci[0-7]).
  # If you use a Docker bridge network, open the ports required for multi-node communication in advance.
  docker run --rm \
  --name $NAME \
  --net=host \
  --shm-size=100g \
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
  -v /root/.cache:/root/.cache \
  -it $IMAGE bash
  ```

  Expected result: The container is listed with status `Up`. You can also verify the vllm-ascend version inside the container:

  ```bash
  pip show vllm-ascend
  ```

  Expected result: The version information is displayed, matching the pulled image version.

## 5 Online Service Deployment {: #5-online-service-deployment }

Start the online serving service with the following command:

For descriptions of the standard `vllm serve` arguments used in the deployment examples, refer to the [vLLM Serving Arguments documentation](https://docs.vllm.ai/en/latest/cli/serve/#arguments). For Ascend-specific options passed through `--additional-config`, refer to [Additional Configuration](../../user_guide/configuration/additional_config.md). For Ascend-specific environment variables, refer to [Environment Variables](../../user_guide/configuration/env_vars.md).

### 5.1 Single-Node Deployment

Single-node deployment completes both Prefill and Decode within the same node. The bfloat(MiniMax-M3) model can be deployed on 1 Atlas 800 A3 (64GB × 16), but dual-node deployment is recommended for BF16 on A3 series; single-node is not recommended. The W8A8 quantized model is recommended for single-node deployment on 1 Atlas 800 A3 (64GB × 16) or 1 Atlas 800 A2 (64GB × 8). The MXFP8 quantized model can be deployed on 1 950DT products (96GB × 8).

=== "A3 series(BF16)"

    ```bash
    export HCCL_OP_EXPANSION_MODE="AIV"
    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    vllm serve ${WEIGHT_PATH} \
      --served-model-name minimax-m3 \
      --trust-remote-code \
      --max-model-len 43008 \
      --tensor-parallel-size 16 \
      --enable-expert-parallel \
      --max-num-seqs 16 \
      --distributed_executor_backend "mp" \
      --gpu-memory-utilization 0.92 \
      --reasoning-parser minimax_m3 \
      --limit-mm-per-prompt '{"image":1,"video":0}' \
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
      --additional-config '{
          "enable_cpu_binding": true,
          "ascend_compilation_config": {
          "enable_static_kernel": true,
          "fuse_norm_quant": false
          },
          "multistream_overlap_shared_expert": true,
          "weight_nz_mode": 2,
          "enable_flashcomm1": true,
          "enable_reduce_sample": true
      }' \
      --port 11223 > ${LOG_PATH} 2>&1 &
    ```

=== "A3 series(W8A8)"

    ```bash
    export HCCL_OP_EXPANSION_MODE="AIV"
    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    vllm serve ${WEIGHT_PATH} \
    --served-model-name minimax-m3 \
    --trust-remote-code \
    --max-model-len 131072 \
    --tensor-parallel-size 4 \
    --data-parallel-size 4 \
    --api-server-count 1 \
    --max-num-batched-tokens 32768 \
    --long-prefill-token-threshold 4096 \
    --enable-expert-parallel \
    --max-num-seqs 32 \
    --distributed_executor_backend "mp" \
    --gpu-memory-utilization 0.92 \
    --reasoning-parser minimax_m3 \
    --limit-mm-per-prompt '{"image":1,"video":0}' \
    --speculative-config '{"model":"${EAGLE3_WEIGHT_PATH}", "method":"eagle3", "num_speculative_tokens":3}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --additional-config '{
        "enable_cpu_binding": true,
        "ascend_compilation_config": {
          "enable_static_kernel": true,
          "fuse_norm_quant": false
            },
        "multistream_overlap_shared_expert": true,
        "enable_shared_expert_dp": true,
        "weight_nz_mode": 2,
        "enable_flashcomm1": true,
        "enable_reduce_sample": true
    }' \
    --port 11223 > ${LOG_PATH} 2>&1 &
    ```

=== "950DT products"

    ```bash
    nic_name="xxxx"  # NIC corresponding to local_ip
    export GLOO_SOCKET_IFNAME=$nic_name
    export HCCL_SOCKET_IFNAME=$nic_name
    export HCCL_BUFFSIZE=512
    export HCCL_BUFFSIZE_EP=512
    export HCCL_OP_EXPANSION_MODE=AIV
    export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.1.0/opp/vendors/experimental_950_transformer/op_api/lib/:${LD_LIBRARY_PATH}
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export VLLM_SERVER_DEV_MODE=1

    vllm serve ${WEIGHT_PATH} \
      --host 0.0.0.0 \
      --port 11223 \
      --served-model-name minimax-m3 \
      --trust-remote-code \
      --distributed-executor-backend mp \
      --tensor-parallel-size 4 \
      --data-parallel-size 2 \
      --enable-expert-parallel \
      --dtype bfloat16 \
      --quantization mxfp8 \
      --max-model-len 140000 \
      --max-num-batched-tokens 16384 \
      --kv-cache-dtype fp8 \
      --max-num-seqs 500 \
      --enable-prefix-caching \
      --async-scheduling \
      --reasoning-parser minimax_m3 \
      --limit-mm-per-prompt '{"image":1,"video":0}' \
      --gpu-memory-utilization 0.92 \
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
      --additional-config '{"enable_cpu_binding":true,"ascend_compilation_config":{"fuse_qknorm_rope":false,"fuse_norm_quant":false,"enable_static_kernel":false},"multistream_overlap_shared_expert":true,"enable_shared_expert_dp":true,"enable_reduce_sample":false}' \
      --speculative-config '{"method":"eagle3","model":"${EAGLE3_WEIGHT_PATH}","num_speculative_tokens":3,"kv_cache_dtype": "bfloat16"}' \
      --safetensors-load-strategy prefetch > ${LOG_PATH} 2>&1 &
    ```

**Note**: In the script above, `max-num-seqs` represents the maximum number of sequences the scheduler can process in a single iteration. Adjust the `max-num-seqs` parameter dynamically based on actual business.

For text-only deployment, `--limit-mm-per-prompt` can be omitted. For multimodal deployment, configure this parameter according to the actual request shape. For example, use `--limit-mm-per-prompt '{"image":2,"video":0}'` for two-image requests, and use `--limit-mm-per-prompt '{"image":0,"video":1}'` for one-video requests.

### 5.2 Multi-Node Deployment

Deploying the float model on Ascend A2 servers requires at least two nodes. Multi-node deployment on A3 servers without prefill–decode disaggregation is not recommended. Update `WEIGHT_PATH`, `EAGLE3_WEIGHT_PATH`, `LOG_PATH`, `local_ip`, `node0_ip`, and `IFNAME` based on the actual environment.

=== "A3 series(BF16)"

    Run the following command on node 0:

    ```bash
    local_ip="${NODE0_IP}"
    node0_ip="${NODE0_IP}"

    export HCCL_IF_IP=$local_ip
    export IFNAME="${NETWORK_INTERFACE}"
    export HCCL_OP_EXPANSION_MODE="AIV"
    export HCCL_SOCKET_IFNAME="$IFNAME"
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
    export GLOO_SOCKET_IFNAME="$IFNAME"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    vllm serve ${WEIGHT_PATH} \
      --host 0.0.0.0 \
      --served-model-name minimax-m3 \
      --trust-remote-code \
      --max-model-len 40960 \
      --tensor-parallel-size 8 \
      --enable-expert-parallel \
      --max-num-seqs 8 \
      --data-parallel-size 2 \
      --data-parallel-size-local 1 \
      --data-parallel-start-rank 0 \
      --data-parallel-address $node0_ip \
      --distributed_executor_backend "mp" \
      --gpu-memory-utilization 0.94 \
      --reasoning-parser minimax_m3 \
      --limit-mm-per-prompt '{"image":1,"video":0}' \
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
      --additional-config '{"enable_cpu_binding":true, "ascend_compilation_config":{"fuse_norm_quant":false}, "multistream_overlap_shared_expert": true, "weight_nz_mode": 2}' \
      --port 11223 > ${LOG_PATH} 2>&1 &
    ```

    Run the following command on node 1:

    ```bash
    local_ip="${NODE1_IP}"
    node0_ip="${NODE0_IP}"

    export HCCL_IF_IP=$local_ip
    export IFNAME="${NETWORK_INTERFACE}"
    export HCCL_OP_EXPANSION_MODE="AIV"
    export HCCL_SOCKET_IFNAME="$IFNAME"
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
    export GLOO_SOCKET_IFNAME="$IFNAME"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    vllm serve ${WEIGHT_PATH} \
      --host 0.0.0.0 \
      --served-model-name minimax-m3 \
      --trust-remote-code \
      --headless \
      --max-model-len 40960 \
      --tensor-parallel-size 8 \
      --enable-expert-parallel \
      --max-num-seqs 8 \
      --data-parallel-size 2 \
      --data-parallel-size-local 1 \
      --data-parallel-start-rank 1 \
      --data-parallel-address $node0_ip \
      --distributed_executor_backend "mp" \
      --gpu-memory-utilization 0.94 \
      --reasoning-parser minimax_m3 \
      --limit-mm-per-prompt '{"image":1,"video":0}' \
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
      --additional-config '{"enable_cpu_binding":true, "ascend_compilation_config":{"fuse_norm_quant":false}, "multistream_overlap_shared_expert": true, "weight_nz_mode": 2}' \
      --port 11223 > ${LOG_PATH} 2>&1 &
    ```

=== "A3 series(W8A8)"

    Run the following command on node 0:

    ```bash
    local_ip="${NODE0_IP}"
    node0_ip="${NODE0_IP}"

    export HCCL_IF_IP=$local_ip
    export IFNAME="${NETWORK_INTERFACE}"
    export HCCL_OP_EXPANSION_MODE="AIV"
    export HCCL_SOCKET_IFNAME="$IFNAME"
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
    export GLOO_SOCKET_IFNAME="$IFNAME"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    vllm serve ${WEIGHT_PATH} \
      --host 0.0.0.0 \
      --served-model-name minimax-m3 \
      --trust-remote-code \
      --max-model-len 131072 \
      --tensor-parallel-size 8 \
      --enable-expert-parallel \
      --max-num-seqs 8 \
      --data-parallel-size 2 \
      --data-parallel-size-local 1 \
      --data-parallel-start-rank 0 \
      --data-parallel-address $node0_ip \
      --distributed_executor_backend "mp" \
      --gpu-memory-utilization 0.92 \
      --reasoning-parser minimax_m3 \
      --limit-mm-per-prompt '{"image":1,"video":0}' \
      --speculative-config '{"model":"${EAGLE3_WEIGHT_PATH}", "method":"eagle3", "num_speculative_tokens":3}' \
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
      --additional-config '{"enable_cpu_binding":true, "ascend_compilation_config":{"fuse_norm_quant":false}, "multistream_overlap_shared_expert": false, "weight_nz_mode": 2, "enable_flashcomm1": true}' \
      --port 11223 > ${LOG_PATH} 2>&1 &
    ```

    Run the following command on node 1:

    ```bash
    local_ip="${NODE1_IP}"
    node0_ip="${NODE0_IP}"

    export HCCL_IF_IP=$local_ip
    export IFNAME="${NETWORK_INTERFACE}"
    export HCCL_OP_EXPANSION_MODE="AIV"
    export HCCL_SOCKET_IFNAME="$IFNAME"
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
    export GLOO_SOCKET_IFNAME="$IFNAME"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True


    vllm serve ${WEIGHT_PATH} \
      --host 0.0.0.0 \
      --served-model-name minimax-m3 \
      --trust-remote-code \
      --headless \
      --max-model-len 131072 \
      --tensor-parallel-size 8 \
      --enable-expert-parallel \
      --max-num-seqs 8 \
      --data-parallel-size 2 \
      --data-parallel-size-local 1 \
      --data-parallel-start-rank 1 \
      --data-parallel-address $node0_ip \
      --distributed_executor_backend "mp" \
      --gpu-memory-utilization 0.92 \
      --reasoning-parser minimax_m3 \
      --limit-mm-per-prompt '{"image":1,"video":0}' \
      --speculative-config '{"model":"${EAGLE3_WEIGHT_PATH}", "method":"eagle3", "num_speculative_tokens":3}' \
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
      --additional-config '{"enable_cpu_binding":true, "ascend_compilation_config":{"fuse_norm_quant":false}, "multistream_overlap_shared_expert": false, "weight_nz_mode": 2, "enable_flashcomm1": true}' \
      --port 11223 > ${LOG_PATH} 2>&1 &
    ```

### 5.3 Prefill-Decode Disaggregation

We'd like to show the deployment guide of MiniMax-M3 on a multi-node environment with 1P1D for better performance.

PD disaggregation separates Prefill and Decode into different service groups. Prefill nodes process large prompt chunks, Decode nodes serve token generation, and a proxy forwards requests between them. Use Mooncake for KV cache transfer. Refer to [Mooncake](../features/pd_disaggregation_mooncake_multi_node.md) for the general PD disaggregation workflow.

The launch pattern is: prepare `launch_online_dp.py` and a role-specific `run_dp_template.sh` on each node, then start a load-balance proxy after every engine prints `Application startup complete`. The launcher below extends the repository example with `--pp-size`: on A3, Prefill uses pipeline parallel (`PP=2`) with a `30,30` split of the 60 transformer layers, while the 950DT products MXFP8 launch uses `PP=1` with `DP=2` on both roles. Each DP rank occupies `tp_size * pp_size` NPUs.

**Common Issues Tip:** For PD disaggregation issues such as KV transfer timeouts or Mooncake connection errors, refer to the [Public FAQs](../../faqs.md). For MiniMax-specific issues, refer to [Chapter 10 FAQ](#10-faq).

Before you start, prepare the script `launch_online_dp.py` on each node:

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
        "--pp-size",
        type=int,
        default=1,
        help="Pipeline parallel size."
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
        default="12321",
        help="Port for data parallel master node."
    )
    parser.add_argument(
        "--vllm-start-port",
        type=int,
        default=8000,
        help="Starting port for the engine."
    )
    return parser.parse_args()


args = parse_args()
dp_size = args.dp_size
tp_size = args.tp_size
pp_size = args.pp_size
dp_size_local = args.dp_size_local
if dp_size_local == -1:
    dp_size_local = dp_size
dp_rank_start = args.dp_rank_start
dp_address = args.dp_address
dp_rpc_port = args.dp_rpc_port
vllm_start_port = args.vllm_start_port
gpus_per_dp_rank = tp_size * pp_size


def run_command(visible_devices, dp_rank, vllm_engine_port):
    command = [
        "bash",
        "./run_dp_template.sh",
        visible_devices,
        str(vllm_engine_port),
        str(dp_size),
        str(dp_rank),
        dp_address,
        dp_rpc_port,
        str(tp_size),
        str(pp_size),
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    template_path = "./run_dp_template.sh"
    if not os.path.exists(template_path):
        print(f"Template file {template_path} does not exist.")
        sys.exit(1)

    processes = []
    for i in range(dp_size_local):
        dp_rank = dp_rank_start + i
        vllm_engine_port = vllm_start_port + i
        visible_devices = ",".join(
            str(x) for x in range(i * gpus_per_dp_rank, (i + 1) * gpus_per_dp_rank)
        )
        process = multiprocessing.Process(
            target=run_command,
            args=(visible_devices, dp_rank, vllm_engine_port),
        )
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
```

`launch_online_dp.py` passes the visible devices, port, DP size, DP rank, DP address, DP RPC port, TP size, and PP size as `$1` through `$8`.

Then prepare `run_dp_template.sh` on each node and start the engines.

=== "A3 series"

    Prefill-Decode disaggregation can be deployed on 2 Atlas 800 A3 (64GB × 16) for `MiniMax-M3` (BF16) with EAGLE3.

    **Deployment topology:**

    | Node group | Nodes | Parallelism | Engine ports |
    | ---------- | ----- | ----------- | ------------ |
    | Prefill | 1 | `DP2 TP4 PP2` (2 ranks, 8 NPUs each) | 31050/31051 |
    | Decode | 1 | `DP4 TP4 PP1` (4 ranks, 4 NPUs each) | 31060-31063 |

    1. Prefill node

    ```bash
    unset http_proxy https_proxy ftp_proxy

    nic_name="xxxx"                 # NIC corresponding to local_ip
    local_ip="xxxx"                 # Prefill node IP
    model_path="xxxx"               # MiniMax-M3 model path
    draft_model_path="xxxx"         # MiniMax-M3-EAGLE3 path

    export VLLM_PP_LAYER_PARTITION="30,30"
    export HCCL_BUFFSIZE=1024
    export HCCL_IF_IP=$local_ip
    export HCCL_OP_EXPANSION_MODE="AIV"
    export ASCEND_RT_VISIBLE_DEVICES=$1
    export HCCL_SOCKET_IFNAME=$nic_name
    export GLOO_SOCKET_IFNAME=$nic_name
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export PYTHONHASHSEED=0

    # If Mooncake is installed in a non-standard path, set this before startup.
    if [ -n "${MOONCAKE_LIB_DIRS:-}" ]; then
        export LD_LIBRARY_PATH="${MOONCAKE_LIB_DIRS}:${LD_LIBRARY_PATH:-}"
    fi

    vllm serve "$model_path" \
        --host 0.0.0.0 \
        --port $2 \
        --data-parallel-size $3 \
        --data-parallel-rank $4 \
        --data-parallel-address $5 \
        --data-parallel-rpc-port $6 \
        --tensor-parallel-size $7 \
        --pipeline-parallel-size $8 \
        --enforce-eager \
        --distributed-executor-backend mp \
        --served-model-name minimax-m3 \
        --enable-expert-parallel \
        --seed 1024 \
        --max-model-len 133000 \
        --max-num-seqs 32 \
        --max-num-batched-tokens 32768 \
        --long-prefill-token-threshold 2048 \
        --trust-remote-code \
        --gpu-memory-utilization 0.85 \
        --reasoning-parser minimax_m3 \
        --limit-mm-per-prompt '{"image":1,"video":0}' \
        --additional-config '{"enable_cpu_binding":true,"ascend_compilation_config":{"fuse_norm_quant":false},"multistream_overlap_shared_expert":true,"weight_nz_mode":2,"enable_shared_expert_dp":true}' \
        --speculative-config '{"method":"eagle3","model":"'"$draft_model_path"'","num_speculative_tokens":3}' \
        --kv-transfer-config \
        '{
            "kv_connector":"MooncakeConnectorV1",
            "kv_role":"kv_producer",
            "kv_port":"36000",
            "engine_id":"0",
            "kv_connector_extra_config":{
                "use_ascend_direct":true,
                "prefill":{"dp_size":2,"tp_size":4,"pp_size":2,"pp_layer_partition":"30,30"},
                "decode":{"dp_size":4,"tp_size":4,"pp_size":1}
            }
        }'
    ```

    2. Decode node

    ```bash
    unset http_proxy https_proxy ftp_proxy

    nic_name="xxxx"                 # NIC corresponding to local_ip
    local_ip="xxxx"                 # Decode node IP
    model_path="xxxx"               # MiniMax-M3 model path
    draft_model_path="xxxx"         # MiniMax-M3-EAGLE3 path

    export HCCL_BUFFSIZE=2048
    export HCCL_IF_IP=$local_ip
    export HCCL_OP_EXPANSION_MODE="AIV"
    export ASCEND_RT_VISIBLE_DEVICES=$1
    export HCCL_SOCKET_IFNAME=$nic_name
    export GLOO_SOCKET_IFNAME=$nic_name
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export PYTHONHASHSEED=0

    # If Mooncake is installed in a non-standard path, set this before startup.
    if [ -n "${MOONCAKE_LIB_DIRS:-}" ]; then
        export LD_LIBRARY_PATH="${MOONCAKE_LIB_DIRS}:${LD_LIBRARY_PATH:-}"
    fi

    vllm serve "$model_path" \
        --host 0.0.0.0 \
        --port $2 \
        --data-parallel-size $3 \
        --data-parallel-rank $4 \
        --data-parallel-address $5 \
        --data-parallel-rpc-port $6 \
        --tensor-parallel-size $7 \
        --pipeline-parallel-size $8 \
        --enable-expert-parallel \
        --seed 1024 \
        --served-model-name minimax-m3 \
        --reasoning-parser minimax_m3 \
        --distributed-executor-backend mp \
        --max-model-len 133000 \
        --max-num-batched-tokens 32768 \
        --trust-remote-code \
        --no-enable-prefix-caching \
        --max-num-seqs 64 \
        --gpu-memory-utilization 0.92 \
        --limit-mm-per-prompt '{"image":1,"video":0}' \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
        --additional-config '{"enable_cpu_binding":true,"ascend_compilation_config":{"fuse_norm_quant":false},"multistream_overlap_shared_expert":true,"weight_nz_mode":2,"enable_shared_expert_dp":true}' \
        --speculative-config '{"method":"eagle3","model":"'"$draft_model_path"'","num_speculative_tokens":3}' \
        --kv-transfer-config \
        '{
            "kv_connector":"MooncakeConnectorV1",
            "kv_role":"kv_consumer",
            "kv_port":"36100",
            "engine_id":"1",
            "kv_connector_extra_config":{
                "use_ascend_direct":true,
                "prefill":{"dp_size":2,"tp_size":4,"pp_size":2,"pp_layer_partition":"30,30"},
                "decode":{"dp_size":4,"tp_size":4,"pp_size":1}
            }
        }'
    ```

    Once the preparation is done, start the server with the following command on each node:

    1. Prefill node

    ```bash
    python launch_online_dp.py \
        --dp-size 2 --tp-size 4 --pp-size 2 \
        --dp-size-local 2 --dp-rank-start 0 \
        --dp-address $node_p_ip --dp-rpc-port 6884 \
        --vllm-start-port 31050
    ```

    This starts two Prefill API servers on ports `31050` and `31051`. Wait until both ranks print `Application startup complete`.

    2. Decode node

    ```bash
    python launch_online_dp.py \
        --dp-size 4 --tp-size 4 --pp-size 1 \
        --dp-size-local 4 --dp-rank-start 0 \
        --dp-address $node_d_ip --dp-rpc-port 5964 \
        --vllm-start-port 31060
    ```

    This starts four Decode API servers on ports `31060` through `31063`.

    To set up request forwarding, run the following script on a node that can reach every Prefill and Decode API port. You can get the proxy program in the repository's examples: [load_balance_proxy_server_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py). For A3 1P1D, the proxy forwards requests to 2 Prefill ranks and 4 Decode ranks.

    ```bash
    unset http_proxy
    unset https_proxy
    unset ftp_proxy

    python load_balance_proxy_server_example.py \
    --port 8009 \
    --host $node_p_ip \
    --prefiller-hosts \
        $node_p_ip $node_p_ip \
    --prefiller-ports \
        31050 31051 \
    --decoder-hosts \
        $node_d_ip $node_d_ip $node_d_ip $node_d_ip \
    --decoder-ports \
        31060 31061 31062 31063 \
    --max-retries 3
    ```

    The service is then accessible at `http://<proxy_ip>:8009`. For PD disaggregation, use this proxy endpoint in Section 7.

=== "950DT products"

    Prefill-Decode disaggregation can be deployed on 2 950DT products (96GB × 8) for `MiniMax-M3-MXFP8` with EAGLE3. Mount `/etc/hixlep/` in the container for UBOE / Ascend direct KV transfer.

    **Deployment topology:**

    | Node group | Nodes | Parallelism | Engine ports |
    | ---------- | ----- | ----------- | ------------ |
    | Prefill | 1 | `DP2 TP4 PP1` (2 ranks, 4 NPUs each) | 31050/31051 |
    | Decode | 1 | `DP2 TP4 PP1` (2 ranks, 4 NPUs each) | 31060/31061 |

    Each node launches one API process per DP rank: 2 Prefill ranks on ports 31050/31051 and 2 Decode ranks on ports 31060/31061. Both roles use `PP=1` (no pipeline parallel), so no `VLLM_PP_LAYER_PARTITION` setting is required. Both sides must declare the same topology in `kv_connector_extra_config`:

    ```json
    {
      "prefill": {"dp_size": 2, "tp_size": 4, "pp_size": 1},
      "decode": {"dp_size": 2, "tp_size": 4, "pp_size": 1}
    }
    ```

    1. Prefill node

    ```bash
    unset ftp_proxy https_proxy http_proxy all_proxy
    unset FTP_PROXY HTTPS_PROXY HTTP_PROXY ALL_PROXY

    nic_name="xxxx"                 # NIC corresponding to local_ip
    local_ip="xxxx"                 # Prefill node IP
    model_path="xxxx"               # MiniMax-M3-MXFP8 model path
    draft_model_path="xxxx"         # MiniMax-M3-EAGLE3 path

    export HCCL_BUFFSIZE=256
    export HCCL_IF_IP=$local_ip
    export HCCL_OP_EXPANSION_MODE=AIV
    export ASCEND_RT_VISIBLE_DEVICES=$1
    export HCCL_SOCKET_IFNAME=$nic_name
    export GLOO_SOCKET_IFNAME=$nic_name
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export PYTHONHASHSEED=0
    export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH

    vllm serve "$model_path" \
        --host 0.0.0.0 \
        --port $2 \
        --data-parallel-size $3 \
        --data-parallel-rank $4 \
        --data-parallel-address $5 \
        --data-parallel-rpc-port $6 \
        --tensor-parallel-size $7 \
        --pipeline-parallel-size $8 \
        --served-model-name minimax-m3 \
        --trust-remote-code \
        --dtype bfloat16 \
        --max-num-seqs 128 \
        --max-num-batched-tokens 32768 \
        --max-model-len 133000 \
        --enable-expert-parallel \
        --quantization mxfp8 \
        --gpu-memory-utilization 0.92 \
        --distributed-executor-backend mp \
        --kv-cache-dtype fp8 \
        --reasoning-parser minimax_m3 \
        --safetensors-load-strategy prefetch \
        --speculative-config '{"method":"eagle3","model":"'"$draft_model_path"'","num_speculative_tokens":3,"kv_cache_dtype":"bfloat16"}' \
        --enforce-eager \
        --no-async-scheduling \
        --additional-config '{"enable_cpu_binding":true,"ascend_compilation_config":{"fuse_qknorm_rope":false,"fuse_norm_quant":false,"enable_static_kernel":false},"multistream_overlap_shared_expert":true,"enable_shared_expert_dp":true,"enable_reduce_sample":false}' \
        --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_producer","kv_port":"30000","engine_id":"0","kv_connector_extra_config":{"use_ascend_direct":true,"ascend_local_comm_res_path":"/etc/hixlep","prefill":{"dp_size":2,"tp_size":4,"pp_size":1},"decode":{"dp_size":2,"tp_size":4,"pp_size":1}}}'
    ```

    2. Decode node

    ```bash
    unset ftp_proxy https_proxy http_proxy all_proxy
    unset FTP_PROXY HTTPS_PROXY HTTP_PROXY ALL_PROXY

    nic_name="xxxx"                 # NIC corresponding to local_ip
    local_ip="xxxx"                 # Decode node IP
    model_path="xxxx"               # MiniMax-M3-MXFP8 model path
    draft_model_path="xxxx"         # MiniMax-M3-EAGLE3 path

    export HCCL_BUFFSIZE=2048
    export HCCL_IF_IP=$local_ip
    export HCCL_OP_EXPANSION_MODE=AIV
    export HCCL_SOCKET_IFNAME=$nic_name
    export GLOO_SOCKET_IFNAME=$nic_name
    export ASCEND_RT_VISIBLE_DEVICES=$1
    export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export PYTHONHASHSEED=0

    vllm serve "$model_path" \
        --host 0.0.0.0 \
        --port $2 \
        --data-parallel-size $3 \
        --data-parallel-rank $4 \
        --data-parallel-address $5 \
        --data-parallel-rpc-port $6 \
        --tensor-parallel-size $7 \
        --pipeline-parallel-size $8 \
        --enable-expert-parallel \
        --seed 1024 \
        --served-model-name minimax-m3 \
        --reasoning-parser minimax_m3 \
        --distributed-executor-backend mp \
        --max-model-len 133000 \
        --max-num-batched-tokens 32768 \
        --trust-remote-code \
        --max-num-seqs 256 \
        --gpu-memory-utilization 0.92 \
        --dtype bfloat16 \
        --quantization mxfp8 \
        --kv-cache-dtype fp8 \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
        --speculative-config '{"method":"eagle3","model":"'"$draft_model_path"'","num_speculative_tokens":3,"kv_cache_dtype":"bfloat16"}' \
        --additional-config '{"enable_cpu_binding":true,"ascend_compilation_config":{"enable_static_kernel":false,"fuse_norm_quant":false},"multistream_overlap_shared_expert":true,"enable_shared_expert_dp":true,"enable_reduce_sample":false}' \
        --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_consumer","kv_port":"26900","engine_id":"1","kv_connector_extra_config":{"use_ascend_direct":true,"ascend_local_comm_res_path":"/etc/hixlep","prefill":{"dp_size":2,"tp_size":4,"pp_size":1},"decode":{"dp_size":2,"tp_size":4,"pp_size":1}}}'
    ```

    Once the preparation is done, start the server with the following command on each node:

    1. Prefill node

    ```bash
    python launch_online_dp.py \
        --dp-size 2 --tp-size 4 --pp-size 1 \
        --dp-size-local 2 --dp-rank-start 0 \
        --dp-address $node_p_ip --dp-rpc-port 6884 \
        --vllm-start-port 31050
    ```

    This starts two Prefill API servers on ports `31050` and `31051`. Wait until both ranks print `Application startup complete`.

    2. Decode node

    ```bash
    python launch_online_dp.py \
        --dp-size 2 --tp-size 4 --pp-size 1 \
        --dp-size-local 2 --dp-rank-start 0 \
        --dp-address $node_d_ip --dp-rpc-port 5964 \
        --vllm-start-port 31060
    ```

    This starts two Decode API servers on ports `31060` and `31061`.

    To set up request forwarding, run the following script on a node that can reach every Prefill and Decode API port. You can get the proxy program in the repository's examples: [load_balance_proxy_server_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py). For 950DT products 1P1D, the proxy forwards requests to 2 Prefill ranks and 2 Decode ranks.

    ```bash
    unset ftp_proxy
    unset https_proxy
    unset http_proxy

    python load_balance_proxy_server_example.py \
    --port 8009 \
    --host $node_p_ip \
    --prefiller-hosts \
        $node_p_ip $node_p_ip \
    --prefiller-ports \
        31050 31051 \
    --decoder-hosts \
        $node_d_ip $node_d_ip \
    --decoder-ports \
        31060 31061
    ```

    The service is then accessible at `http://<proxy_ip>:8009`. For PD disaggregation, use this proxy endpoint in Section 7.

Key Parameter Descriptions:

**`launch_online_dp.py` parameters:**

| Parameter | Type | Required | Default | Description |
| --------- | ---- | -------- | ------- | ----------- |
| `--dp-size` | int | Yes | - | Data parallel size (total number of DP ranks across all nodes). |
| `--tp-size` | int | No | 1 | Tensor parallel size within each DP rank. |
| `--pp-size` | int | No | 1 | Pipeline parallel size within each DP rank. Each rank occupies `tp_size * pp_size` NPUs. |
| `--dp-size-local` | int | No | (same as `--dp-size`) | Number of DP ranks on the current node. |
| `--dp-rank-start` | int | No | 0 | Starting rank offset for data parallel ranks on this node. |
| `--dp-address` | str | Yes | - | IP address of the data parallel master node. |
| `--dp-rpc-port` | str | No | 12321 | RPC port for data parallel master communication. |
| `--vllm-start-port` | int | No | 8000 | Starting port for each vLLM engine instance on this node. Each DP rank's engine port = `vllm_start_port` + local rank index. |

**Prefill node-specific configurations:**

- `--pipeline-parallel-size` (A3 Prefill: `2`): Splits the 60 MiniMax-M3 layers across two pipeline stages. A3 sets `VLLM_PP_LAYER_PARTITION=30,30` and also writes `pp_layer_partition` into the Mooncake extra config. The 950DT products launch uses `--pp-size 1` on both Prefill and Decode (no pipeline parallel), so no layer partition is needed.
- `--enforce-eager`: Prefill nodes do not capture CUDA/ACL graphs.
- `--speculative-config '{"method":"eagle3", ...}'`: Enables the MiniMax-M3 EAGLE3 draft model. Do not replace this with GLM MTP options.
- `--no-async-scheduling` (950DT products): Used by the verified MXFP8 Prefill launch.

**Decode node-specific configurations:**

- `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`: Graph capture for the decode phase only.
- `--no-enable-prefix-caching` (A3 Decode): Disables prefix caching on the Decode node to avoid the D-node prefix-cache known issue tracked in [#7944](https://github.com/vllm-project/vllm-ascend/issues/7944). The 950DT products launch does not set this flag and keeps prefix caching enabled.
- `--max-num-seqs 256`: Decode concurrency used by the verified 950DT products 1P1D launch. A3 uses `64`.

**Mooncake KV transfer configuration (`--kv-transfer-config`):**

- `"kv_connector": "MooncakeConnectorV1"`: Uses Mooncake as the KV cache transfer connector between prefill and decode nodes.
- `"kv_role": "kv_producer"` / `"kv_consumer"`: `kv_producer` on prefill nodes, `kv_consumer` on decode nodes.
- `"kv_port"`: Port for Mooncake KV transfer. Use different ports for prefill and decode. The verified values are A3 `36000`/`36100` and 950DT products `30000`/`26900`.
- `"use_ascend_direct": true`: Enables Ascend direct transfer for KV cache.
- `"ascend_local_comm_res_path": "/etc/hixlep"` (950DT products only): Required for UBOE / Ascend direct communication on 950DT products.
- `"prefill"` / `"decode"` sections: `dp_size`, `tp_size`, and `pp_size` must match the actual global layout on both nodes. A3 uses `prefill: dp2 tp4 pp2` and `decode: dp4 tp4 pp1`. 950DT products uses `prefill: dp2 tp4 pp1` and `decode: dp2 tp4 pp1`.

**Request forwarding (proxy):**

- Wait until every Prefill and Decode rank prints `Application startup complete` before starting the proxy.
- The proxy maps every prefill engine endpoint and every decode engine endpoint to a single entry point on port `8009`.
- If requests reach the proxy but no output is returned, check that the proxy host list includes every healthy Prefill and Decode port, and that both nodes still have free NPU memory after the previous run.

Please refer to [envs.py](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/envs.py) for further explanation and restrictions of the environment variables above.

### 5.4 Prefill-Decode Disaggregation with KV Cache Pool

This section reuses the Section 5.3 topology, `launch_online_dp.py`, proxy mapping, and most `vllm serve` flags. The difference is that Prefill and Decode use `MultiConnector` so that live P→D KV transfer and Mooncake KV Cache Pool work together.

Use Mooncake for both paths:

- `MooncakeConnectorV1` transfers KV from Prefill to Decode in real time.
- `AscendStoreConnector` stores KV in the Mooncake KV Cache Pool.
- By default, the Prefill node looks up, loads, and writes the pool.

Reuse the `launch_online_dp.py` from Section 5.3. The launcher still passes visible devices, port, DP size, DP rank, DP address, DP RPC port, TP size, and PP size as positional arguments. Refer to the [KV Cache Pool Guide](../../user_guide/feature_guide/kv_pool.md) for backend, eviction, and tenant options. For MiniMax-specific issues, refer to [Chapter 10 FAQ](#10-faq).

Before startup, mount the host HCCN config into every container that participates in pooling:

```bash
-v /etc/hccn.conf:/etc/hccn.conf:ro
```

Place `mooncake.json` next to `run_dp_template.sh` on each node. Prefill contributes pool memory; Decode does not. Replace `xxxx` with the Prefill node IP. A non-zero `global_segment_size` must be aligned to `1GB`.

Prefill `mooncake.json`:

```json
{"metadata_server":"P2PHANDSHAKE","protocol":"ascend","device_name":"","master_server_address":"xxxx:50088","global_segment_size":"16GB","preferred_segment":true,"prefer_alloc_in_same_node":true,"enable_ssd_offload":false,"tenant_id":"default"}
```

Decode `mooncake.json` (same Master and `tenant_id`, no donated segment):

```json
{"metadata_server":"P2PHANDSHAKE","protocol":"ascend","device_name":"","master_server_address":"xxxx:50088","global_segment_size":0,"preferred_segment":false,"prefer_alloc_in_same_node":true,"enable_ssd_offload":false,"tenant_id":"default"}
```

Both nodes must use the same `PYTHONHASHSEED`, Mooncake Master address, and `tenant_id`. Set `MOONCAKE_CONFIG_PATH` to the local `mooncake.json`. Prefill pooling RPC ports start at `37000`; Decode pooling RPC ports start at `37100`; add the DP rank so every rank is unique.

Start the services in this order: Mooncake Master → Decode → Prefill → Proxy.

#### A3 series

Prefill-Decode disaggregation with KV Cache Pool can be deployed on 2 Atlas 800 A3 (64GB × 16) for `MiniMax-M3` (BF16) with EAGLE3. Keep expert dual-stream, Shared Expert DP, EAGLE3, Prefill eager, Decode `FULL_DECODE_ONLY`, and `max-model-len=133000` from Section 5.3.

**Deployment topology:**

| Node group | Nodes | Parallelism | Engine ports | GPU Memory Utilization |
| ---------- | ----- | ----------- | ------------ | ---------------------- |
| Prefill | 1 | `DP2 TP4 PP2` (2 ranks, 8 NPUs each) | 31050/31051 | 0.85 |
| Decode | 1 | `DP4 TP4 PP1` (4 ranks, 4 NPUs each) | 31060-31063 | 0.92 |

Model and serving limits stay the same as Section 5.3: Prefill `max-num-seqs=32`, Decode `max-num-seqs=64`, `max-model-len=133000`.

1. Prefill node `run_dp_template.sh`

```bash
unset http_proxy https_proxy ftp_proxy

nic_name="xxxx"                 # NIC corresponding to local_ip
local_ip="xxxx"                 # Prefill node IP
model_path="xxxx"               # MiniMax-M3 model path
draft_model_path="xxxx"         # MiniMax-M3-EAGLE3 path

export VLLM_PP_LAYER_PARTITION="30,30"
export HCCL_BUFFSIZE=1024
export HCCL_IF_IP=$local_ip
export HCCL_OP_EXPANSION_MODE="AIV"
export ASCEND_RT_VISIBLE_DEVICES=$1
export HCCL_SOCKET_IFNAME=$nic_name
export GLOO_SOCKET_IFNAME=$nic_name
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export PYTHONHASHSEED=0
export MOONCAKE_CONFIG_PATH=./mooncake.json

# If Mooncake is installed in a non-standard path, set this before startup.
if [ -n "${MOONCAKE_LIB_DIRS:-}" ]; then
    export LD_LIBRARY_PATH="${MOONCAKE_LIB_DIRS}:${LD_LIBRARY_PATH:-}"
fi

vllm serve "$model_path" \
    --host 0.0.0.0 \
    --port $2 \
    --data-parallel-size $3 \
    --data-parallel-rank $4 \
    --data-parallel-address $5 \
    --data-parallel-rpc-port $6 \
    --tensor-parallel-size $7 \
    --pipeline-parallel-size $8 \
    --enforce-eager \
    --distributed-executor-backend mp \
    --served-model-name minimax-m3 \
    --enable-expert-parallel \
    --seed 1024 \
    --max-model-len 133000 \
    --max-num-seqs 32 \
    --max-num-batched-tokens 32768 \
    --long-prefill-token-threshold 2048 \
    --trust-remote-code \
    --gpu-memory-utilization 0.85 \
    --reasoning-parser minimax_m3 \
    --limit-mm-per-prompt '{"image":1,"video":0}' \
    --additional-config '{"enable_cpu_binding":true,"ascend_compilation_config":{"fuse_norm_quant":false},"multistream_overlap_shared_expert":true,"weight_nz_mode":2,"enable_shared_expert_dp":true}' \
    --speculative-config '{"method":"eagle3","model":"'"$draft_model_path"'","num_speculative_tokens":3}' \
    --kv-transfer-config '{"kv_connector":"MultiConnector","kv_role":"kv_producer","engine_id":"0","kv_connector_extra_config":{"connectors":[{"kv_connector":"MooncakeConnectorV1","kv_buffer_device":"npu","kv_role":"kv_producer","kv_port":"36000","kv_connector_extra_config":{"use_ascend_direct":true,"prefill":{"dp_size":2,"tp_size":4,"pp_size":2,"pp_layer_partition":"30,30"},"decode":{"dp_size":4,"tp_size":4,"pp_size":1}}},{"kv_connector":"AscendStoreConnector","kv_role":"kv_producer","kv_connector_extra_config":{"backend":"mooncake","lookup_rpc_port":"37000"}}]}}'
```

2. Decode node `run_dp_template.sh`

```bash
unset http_proxy https_proxy ftp_proxy

nic_name="xxxx"                 # NIC corresponding to local_ip
local_ip="xxxx"                 # Decode node IP
model_path="xxxx"               # MiniMax-M3 model path
draft_model_path="xxxx"         # MiniMax-M3-EAGLE3 path

export HCCL_BUFFSIZE=2048
export HCCL_IF_IP=$local_ip
export HCCL_OP_EXPANSION_MODE="AIV"
export ASCEND_RT_VISIBLE_DEVICES=$1
export HCCL_SOCKET_IFNAME=$nic_name
export GLOO_SOCKET_IFNAME=$nic_name
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export PYTHONHASHSEED=0
export MOONCAKE_CONFIG_PATH=./mooncake.json

# If Mooncake is installed in a non-standard path, set this before startup.
if [ -n "${MOONCAKE_LIB_DIRS:-}" ]; then
    export LD_LIBRARY_PATH="${MOONCAKE_LIB_DIRS}:${LD_LIBRARY_PATH:-}"
fi

vllm serve "$model_path" \
    --host 0.0.0.0 \
    --port $2 \
    --data-parallel-size $3 \
    --data-parallel-rank $4 \
    --data-parallel-address $5 \
    --data-parallel-rpc-port $6 \
    --tensor-parallel-size $7 \
    --pipeline-parallel-size $8 \
    --enable-expert-parallel \
    --seed 1024 \
    --served-model-name minimax-m3 \
    --reasoning-parser minimax_m3 \
    --distributed-executor-backend mp \
    --max-model-len 133000 \
    --max-num-batched-tokens 32768 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --max-num-seqs 64 \
    --gpu-memory-utilization 0.92 \
    --limit-mm-per-prompt '{"image":1,"video":0}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_cpu_binding":true,"ascend_compilation_config":{"fuse_norm_quant":false},"multistream_overlap_shared_expert":true,"weight_nz_mode":2,"enable_shared_expert_dp":true}' \
    --speculative-config '{"method":"eagle3","model":"'"$draft_model_path"'","num_speculative_tokens":3}' \
    --kv-transfer-config '{"kv_connector":"MultiConnector","kv_role":"kv_consumer","engine_id":"1","kv_connector_extra_config":{"connectors":[{"kv_connector":"MooncakeConnectorV1","kv_buffer_device":"npu","kv_role":"kv_consumer","kv_port":"36100","kv_connector_extra_config":{"use_ascend_direct":true,"prefill":{"dp_size":2,"tp_size":4,"pp_size":2,"pp_layer_partition":"30,30"},"decode":{"dp_size":4,"tp_size":4,"pp_size":1}}},{"kv_connector":"AscendStoreConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"backend":"mooncake","lookup_rpc_port":"37100"}}]}}'
```

3. Mooncake Master on the Prefill node

```bash
mooncake_master \
  --port 50088 \
  --eviction_high_watermark_ratio 0.9 \
  --eviction_ratio 0.1 \
  --default_kv_lease_ttl 11000 \
  --enable_offload=false \
  --client_ttl=120
```

4. Decode node

```bash
python launch_online_dp.py \
    --dp-size 4 --tp-size 4 --pp-size 1 \
    --dp-size-local 4 --dp-rank-start 0 \
    --dp-address $node_d_ip --dp-rpc-port 5964 \
    --vllm-start-port 31060
```

This starts four Decode API servers on ports `31060` through `31063`. Wait until all ranks print `Application startup complete`.

5. Prefill node

```bash
python launch_online_dp.py \
    --dp-size 2 --tp-size 4 --pp-size 2 \
    --dp-size-local 2 --dp-rank-start 0 \
    --dp-address $node_p_ip --dp-rpc-port 6884 \
    --vllm-start-port 31050
```

This starts two Prefill API servers on ports `31050` and `31051`. Wait until both ranks print `Application startup complete`.

6. Proxy

Use the same A3 proxy command as Section 5.3. The service is then accessible at `http://<proxy_ip>:8009`.

```bash
unset http_proxy
unset https_proxy
unset ftp_proxy

python load_balance_proxy_server_example.py \
--port 8009 \
--host $node_p_ip \
--prefiller-hosts \
    $node_p_ip $node_p_ip \
--prefiller-ports \
    31050 31051 \
--decoder-hosts \
    $node_d_ip $node_d_ip $node_d_ip $node_d_ip \
--decoder-ports \
    31060 31061 31062 31063 \
--max-retries 3
```

#### 950DT products

Prefill-Decode disaggregation with KV Cache Pool can be deployed on 2 950DT products (96GB × 8) for `MiniMax-M3-MXFP8` with EAGLE3. Keep the Section 5.3 MXFP8 serving flags, including `quantization mxfp8`, Prefill eager, Decode `FULL_DECODE_ONLY`, and `max-model-len=133000`. Mount `/etc/hixlep/` for UBOE / Ascend direct KV transfer, and also mount `/etc/hccn.conf` for the KV Pool.

**Deployment topology:**

| Node group | Nodes | Parallelism | Engine ports | GPU Memory Utilization |
| ---------- | ----- | ----------- | ------------ | ---------------------- |
| Prefill | 1 | `DP2 TP4 PP1` (2 ranks, 4 NPUs each) | 31050/31051 | 0.92 |
| Decode | 1 | `DP2 TP4 PP1` (2 ranks, 4 NPUs each) | 31060/31061 | 0.92 |

Model and serving limits stay the same as Section 5.3: Prefill `max-num-seqs=128`, Decode `max-num-seqs=256`, `max-model-len=133000`. Both roles use `PP=1`, so no `VLLM_PP_LAYER_PARTITION` setting is required. Both sides must declare the same topology in the Mooncake child connector: Prefill `dp_size=2, tp_size=4, pp_size=1` and Decode `dp_size=2, tp_size=4, pp_size=1`.

1. Prefill node `run_dp_template.sh`

```bash
unset ftp_proxy https_proxy http_proxy all_proxy
unset FTP_PROXY HTTPS_PROXY HTTP_PROXY ALL_PROXY

nic_name="xxxx"                 # NIC corresponding to local_ip
local_ip="xxxx"                 # Prefill node IP
model_path="xxxx"               # MiniMax-M3-MXFP8 model path
draft_model_path="xxxx"         # MiniMax-M3-EAGLE3 path

export HCCL_BUFFSIZE=256
export HCCL_IF_IP=$local_ip
export HCCL_OP_EXPANSION_MODE=AIV
export ASCEND_RT_VISIBLE_DEVICES=$1
export HCCL_SOCKET_IFNAME=$nic_name
export GLOO_SOCKET_IFNAME=$nic_name
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export PYTHONHASHSEED=0
export MOONCAKE_CONFIG_PATH=./mooncake.json
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH

vllm serve "$model_path" \
    --host 0.0.0.0 \
    --port $2 \
    --data-parallel-size $3 \
    --data-parallel-rank $4 \
    --data-parallel-address $5 \
    --data-parallel-rpc-port $6 \
    --tensor-parallel-size $7 \
    --pipeline-parallel-size $8 \
    --served-model-name minimax-m3 \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-num-seqs 128 \
    --max-num-batched-tokens 32768 \
    --max-model-len 133000 \
    --enable-expert-parallel \
    --quantization mxfp8 \
    --gpu-memory-utilization 0.92 \
    --distributed-executor-backend mp \
    --kv-cache-dtype fp8 \
    --reasoning-parser minimax_m3 \
    --safetensors-load-strategy prefetch \
    --speculative-config '{"method":"eagle3","model":"'"$draft_model_path"'","num_speculative_tokens":3,"kv_cache_dtype":"bfloat16"}' \
    --enforce-eager \
    --no-async-scheduling \
    --additional-config '{"enable_cpu_binding":true,"ascend_compilation_config":{"fuse_qknorm_rope":false,"fuse_norm_quant":false,"enable_static_kernel":false},"multistream_overlap_shared_expert":true,"enable_shared_expert_dp":true,"enable_reduce_sample":false}' \
    --kv-transfer-config '{"kv_connector":"MultiConnector","kv_role":"kv_producer","engine_id":"0","kv_connector_extra_config":{"connectors":[{"kv_connector":"MooncakeConnectorV1","kv_buffer_device":"npu","kv_role":"kv_producer","kv_port":"30000","kv_connector_extra_config":{"use_ascend_direct":true,"ascend_local_comm_res_path":"/etc/hixlep","prefill":{"dp_size":2,"tp_size":4,"pp_size":1},"decode":{"dp_size":2,"tp_size":4,"pp_size":1}}},{"kv_connector":"AscendStoreConnector","kv_role":"kv_producer","kv_connector_extra_config":{"backend":"mooncake","lookup_rpc_port":"37000"}}]}}'
```

2. Decode node `run_dp_template.sh`

```bash
unset ftp_proxy https_proxy http_proxy all_proxy
unset FTP_PROXY HTTPS_PROXY HTTP_PROXY ALL_PROXY

nic_name="xxxx"                 # NIC corresponding to local_ip
local_ip="xxxx"                 # Decode node IP
model_path="xxxx"               # MiniMax-M3-MXFP8 model path
draft_model_path="xxxx"         # MiniMax-M3-EAGLE3 path

export HCCL_BUFFSIZE=2048
export HCCL_IF_IP=$local_ip
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_SOCKET_IFNAME=$nic_name
export GLOO_SOCKET_IFNAME=$nic_name
export ASCEND_RT_VISIBLE_DEVICES=$1
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export PYTHONHASHSEED=0
export MOONCAKE_CONFIG_PATH=./mooncake.json

vllm serve "$model_path" \
    --host 0.0.0.0 \
    --port $2 \
    --data-parallel-size $3 \
    --data-parallel-rank $4 \
    --data-parallel-address $5 \
    --data-parallel-rpc-port $6 \
    --tensor-parallel-size $7 \
    --pipeline-parallel-size $8 \
    --enable-expert-parallel \
    --seed 1024 \
    --served-model-name minimax-m3 \
    --reasoning-parser minimax_m3 \
    --distributed-executor-backend mp \
    --max-model-len 133000 \
    --max-num-batched-tokens 32768 \
    --trust-remote-code \
    --max-num-seqs 256 \
    --gpu-memory-utilization 0.92 \
    --dtype bfloat16 \
    --quantization mxfp8 \
    --kv-cache-dtype fp8 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --speculative-config '{"method":"eagle3","model":"'"$draft_model_path"'","num_speculative_tokens":3,"kv_cache_dtype":"bfloat16"}' \
    --additional-config '{"enable_cpu_binding":true,"ascend_compilation_config":{"enable_static_kernel":false,"fuse_norm_quant":false},"multistream_overlap_shared_expert":true,"enable_shared_expert_dp":true,"enable_reduce_sample":false}' \
    --kv-transfer-config '{"kv_connector":"MultiConnector","kv_role":"kv_consumer","engine_id":"1","kv_connector_extra_config":{"connectors":[{"kv_connector":"MooncakeConnectorV1","kv_buffer_device":"npu","kv_role":"kv_consumer","kv_port":"26900","kv_connector_extra_config":{"use_ascend_direct":true,"ascend_local_comm_res_path":"/etc/hixlep","prefill":{"dp_size":2,"tp_size":4,"pp_size":1},"decode":{"dp_size":2,"tp_size":4,"pp_size":1}}},{"kv_connector":"AscendStoreConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"backend":"mooncake","lookup_rpc_port":"37100"}}]}}'
```

3. Mooncake Master on the Prefill node

```bash
mooncake_master \
  --port 50088 \
  --eviction_high_watermark_ratio 0.9 \
  --eviction_ratio 0.1 \
  --default_kv_lease_ttl 11000 \
  --enable_offload=false \
  --client_ttl=120
```

4. Decode node

```bash
python launch_online_dp.py \
    --dp-size 2 --tp-size 4 --pp-size 1 \
    --dp-size-local 2 --dp-rank-start 0 \
    --dp-address $node_d_ip --dp-rpc-port 5964 \
    --vllm-start-port 31060
```

This starts two Decode API servers on ports `31060` and `31061`. Wait until both ranks print `Application startup complete`.

5. Prefill node

```bash
python launch_online_dp.py \
    --dp-size 2 --tp-size 4 --pp-size 1 \
    --dp-size-local 2 --dp-rank-start 0 \
    --dp-address $node_p_ip --dp-rpc-port 6884 \
    --vllm-start-port 31050
```

This starts two Prefill API servers on ports `31050` and `31051`. Wait until both ranks print `Application startup complete`.

6. Proxy

Use the same 950DT products proxy command as Section 5.3. The service is then accessible at `http://<proxy_ip>:8009`.

```bash
unset ftp_proxy
unset https_proxy
unset http_proxy

python load_balance_proxy_server_example.py \
--port 8009 \
--host $node_p_ip \
--prefiller-hosts \
    $node_p_ip $node_p_ip \
--prefiller-ports \
    31050 31051 \
--decoder-hosts \
    $node_d_ip $node_d_ip \
--decoder-ports \
    31060 31061
```

**Parameters to add on top of Section 5.3:**

| Item | Where | What to set |
| ---- | ----- | ----------- |
| `PYTHONHASHSEED=0` | Prefill and Decode | Keep the same value on every node so block hashes match. |
| `MOONCAKE_CONFIG_PATH` | Prefill and Decode | Point to the local `mooncake.json`. |
| Mooncake `LD_LIBRARY_PATH` | Prefill and Decode | Add the Mooncake package directory. |
| `--kv-transfer-config` | Prefill and Decode | Replace `MooncakeConnectorV1` with `MultiConnector`. |
| `MooncakeConnectorV1` child | Extra config | Keep the Section 5.3 P→D transfer settings (`kv_role`, `kv_port`, `prefill` / `decode` topology). |
| `AscendStoreConnector` child | Extra config | Set `backend: mooncake` and a unique `lookup_rpc_port` per DP rank. |
| `engine_id` | Prefill and Decode | Use a unique ID per DP rank, for example `0` / `1`. |
| Prefill `mooncake.json` | Prefill node | `global_segment_size=16GB`, `preferred_segment=true`, Master `xxxx:50088`. |
| Decode `mooncake.json` | Decode node | Same Master and `tenant_id`, but `global_segment_size=0`. |
| `/etc/hccn.conf` | Container | Mount read-only. 950DT products still also need `/etc/hixlep/`. |

**Steps to enable PD pooling:**

1. Add `-v /etc/hccn.conf:/etc/hccn.conf:ro` to the container. On 950DT products, keep the `/etc/hixlep/` mount from Section 5.3.
2. Write `mooncake.json` on both nodes. Prefill donates `16GB` per NPU; Decode donates `0`. Use the same Master address and `tenant_id`.
3. Start `mooncake_master` on the Prefill node and confirm port `50088` is reachable.
4. In each `run_dp_template.sh`, export `PYTHONHASHSEED`, `MOONCAKE_CONFIG_PATH`, and the Mooncake library path, then replace `--kv-transfer-config` with the `MultiConnector` JSON above.
5. Assign unique pooling RPC ports: Prefill `37000` plus DP rank, Decode `37100` plus DP rank.
6. Start Decode, wait until every Decode rank is ready, then start Prefill, then start the proxy on port `8009`.
7. Send requests to the proxy. After a repeated-prefix warmup, check Prefill logs for KV Pool lookup/get/put and hit information, and check Decode logs for a normal Mooncake P→D KV transfer.

The `launch_online_dp.py` arguments, proxy host/port lists, and remaining serving flags stay the same as Section 5.3. Requests still go to the proxy endpoint in Section 7.

> The pooling combination on MiniMax-M3 + EAGLE3 + A3 `PP2` still needs a live bring-up to confirm. 950DT products use `PP=1` on both roles.

### 5.5 Multimodal and ViT DP (Optional)

MiniMax-M3 supports image and video inputs on Ascend. The deployment examples above keep `--limit-mm-per-prompt '{"image":1,"video":0}'` as the default multimodal capacity assumption because the other serving parameters are tuned for the single-image path.

MiniMax-M3 image and video inputs share the same Vision Tower. If a service only needs one modality, explicitly set the unused modality to `0`; for example, use `{"image":1,"video":0}` for image-only serving and `{"image":0,"video":1}` for video-only serving. As long as either image or video remains enabled, the shared Vision Tower is retained. Setting an unused modality to `0` is clearer than omitting it, because omitted modalities may still participate in multimodal capacity and profiling planning.

For the ViT / multimodal encoder part, data parallel execution is supported and can be enabled with:

```bash
--mm-encoder-tp-mode data
```

This option is not enabled in the default deployment examples because it can increase per-card memory usage. When enabling ViT DP, re-evaluate memory-related parameters such as `--max-model-len`, `--max-num-seqs`, and `--gpu-memory-utilization` for the target workload.

For video or mixed image-video requests, adjust the multimodal limit according to the actual request shape instead of changing the default template blindly:

```bash
# one video
--limit-mm-per-prompt '{"image":0,"video":1}'

# one image and one video
--limit-mm-per-prompt '{"image":1,"video":1}'
```

When using local media paths in requests, such as `file:///path/to/video.mp4`, add an explicit allowlist path:

```bash
--allowed-local-media-path /
```

If the number of sampled video frames is not specified, vLLM uses its default video sampling policy, which samples 32 frames by default. For quick functional smoke tests, a smaller frame count such as 8 or 16 can be set in the request or evaluation config. For benchmark runs, follow the dataset protocol.

FLASHCOMM1 and language-model-only mode should not be enabled at the same time for MiniMax-M3 serving. FLASHCOMM1 is enabled through `additional_config.enable_flashcomm1`, while language-model-only mode is enabled with `--language-model-only`.

```bash
# Enable FLASHCOMM1.
--additional-config '{"enable_flashcomm1": true}'

# Enable language-model-only mode.
--language-model-only
```

`VLLM_ASCEND_ENABLE_FLASHCOMM1=1` is kept for compatibility, but `additional_config.enable_flashcomm1` is preferred.

## 6 Thinking and Parser Configuration

### 6.1 Thinking Mode

MiniMax-M3 supports three thinking modes, controlled via `thinking_mode` in `chat_template_kwargs`:

| Mode | Behavior | Use Case |
|------|----------|----------|
| `enabled` | The model thinks before every response, including after tool results | Complex reasoning, agents |
| `disabled` | No thinking; the model answers directly | Latency-sensitive turns |
| `adaptive` | The model decides whether to think based on the task (default when unset) | General use |

#### 6.1.1 Request Examples

**With thinking disabled (curl):**

```bash
curl http://{ip}:{port}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minimax-m3",
    "messages": [{"role": "user", "content": "who are you?"}],
    "max_tokens": 100,
    "stream": false,
    "top_p": 0.95,
    "top_k": 40,
    "temperature": 1.0,
    "chat_template_kwargs": {"thinking_mode": "disabled"}
  }'
```

Change `"thinking_mode"` to `"enabled"` or `"adaptive"` as needed. The deprecated `enable_thinking` parameter (equivalent to `thinking_mode: "enabled"`) is also supported.

**With thinking enabled (Python SDK):**

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

response = client.chat.completions.create(
    model="minimax-m3",
    messages=[{"role": "user", "content": "Prove there are infinitely many primes."}],
    extra_body={"chat_template_kwargs": {"thinking_mode": "enabled"}},
)
msg = response.choices[0].message
print(getattr(msg, "reasoning", None))  # the <mm:think> block
print(msg.content)                       # the final answer
```

### 6.2 Reasoning Parser

The MiniMax-M3 reasoning parser (`--reasoning-parser minimax_m3`) extracts the thinking block `<mm:think>...</mm:think>` from model output and exposes it as the `reasoning` field. The remaining text is returned as `content`.

#### 6.2.1 Server Configuration

The `--reasoning-parser minimax_m3` flag enables the MiniMax-M3 reasoning parser, which splits model output into reasoning and content using `<mm:think>...</mm:think>` delimiters:

```bash
vllm serve ${WEIGHT_PATH} \
  --reasoning-parser minimax_m3 \
  ...
```

#### 6.2.2 Output Format

MiniMax-M3 uses explicit thinking delimiters:

```text
<mm:think>reasoning process...</mm:think>final answer
```

#### 6.2.3 Parser Behavior

- **`thinking_mode="enabled"`**: The chat template pre-fills `<mm:think>` in the prompt. Generated text starts inside the reasoning block and transitions to content after `</mm:think>`.
- **`thinking_mode="disabled"` or default**: Model output is treated as plain content. If `<mm:think>` appears, the parser splits on the delimiters.
- **Streaming**: Reasoning and content are streamed incrementally via `DeltaMessage.reasoning` and `DeltaMessage.content` token-by-token.
- **Token counting**: Reasoning tokens inside `<mm:think>` blocks are correctly counted.

### 6.3 Tool Call Parser

MiniMax-M3 uses a namespace-delimited XML format for tool calls. Enable it with `--tool-parser minimax_m3`.

#### 6.3.1 Server Configuration

When both `--reasoning-parser minimax_m3` and `--tool-call-parser minimax_m3` are specified, the parsers work together automatically to handle responses that contain both reasoning blocks and tool calls:

```bash
vllm serve ${WEIGHT_PATH} \
  --reasoning-parser minimax_m3 \
  --enable-auto-tool-choice \
  --tool-call-parser minimax_m3 \
  ...
```

#### 6.3.2 Tool Call Format

Each structural tag is preceded by the `]<]minimax[>[` namespace marker:

```xml
]<]minimax[>[<tool_call>
]<]minimax[>[<invoke name="create_order">
]<]minimax[>[<user_id>42]<]minimax[>[</user_id>
]<]minimax[>[<shipping>
]<]minimax[>[<city>Singapore]<]minimax[>[</city>
]<]minimax[>[<zip>018956]<]minimax[>[</zip>
]<]minimax[>[</shipping>
]<]minimax[>[</invoke>
]<]minimax[>[</tool_call>
```

#### 6.3.3 Key Features

- **Recursive parameter parsing**: Supports nested objects and arrays (e.g., `shipping` containing `city`/`zip`).
- **Schema-aware type coercion**: String parameter values are automatically converted to the correct types (integer, boolean, object, array) based on the function's JSON Schema definition.
- **Multiple invocations**: A single `<tool_call>` block can contain multiple `<invoke>` blocks.
- **Streaming**: Tool name and argument fragments are streamed incrementally as the `<invoke>` block is received.

#### 6.3.4 Request Example (curl)

```bash
curl http://{ip}:{port}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minimax-m3",
    "messages": [{"role": "user", "content": "What's the weather like in Shanghai?"}],
    "max_tokens": 300,
    "stream": false,
    "tool_choice": "auto",
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City or country name"
                        }
                    },
                    "required": ["location"],
                    "additionalProperties": false
                }
            }
        }
    ],
    "chat_template_kwargs": {"thinking_mode": "disabled"}
  }'
```

## 7 Functional Verification

### 7.1 Text

  ```bash
  curl http://{ip}:{port}/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d @- <<EOF
  {
    "model": "minimax-m3",
    "messages": [
      {
        "role": "user",
        "content": "Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\nA student regrets that he fell asleep during a lecture in electrochemistry, facing the following incomplete statement in a test:\nThermodynamically, oxygen is a …oxidant in basic solutions. Kinetically, oxygen reacts …in acidic solutions.\nWhich combination of weaker/stronger and faster/slower is correct?\n\nA) weaker —faster\nB) stronger —faster\nC) weaker - slower\nD) stronger —slower"
      }
    ],
    "max_tokens": 8000,
    "temperature": 1.0
  }
  EOF
  ```

  Expected result: the answer is C.

### 7.2 Single Image

  Start the service with image input enabled, for example `--limit-mm-per-prompt '{"image":1,"video":0}'`. Replace `${IMAGE_PATH}` with a local image path on the client side.

  ```bash
  IMAGE_PATH=/path/to/image.jpg
  IMAGE_BASE64="$(base64 -w 0 "${IMAGE_PATH}")"

  curl http://{ip}:{port}/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d @- <<EOF
  {
    "model": "minimax-m3",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,${IMAGE_BASE64}"}},
          {"type": "text", "text": "Briefly describe this image."}
        ]
      }
    ],
    "max_tokens": 512,
    "temperature": 0
  }
  EOF
  ```

  Expected result: HTTP 200 response with a JSON body containing non-empty `choices` and generated text describing the image.

### 7.3 Single Video

  Start the service with video input enabled, for example `--limit-mm-per-prompt '{"image":0,"video":1}'`. If the request uses `file://` local video paths, also add `--allowed-local-media-path /` or a narrower allowed directory. If `media_io_kwargs.video.num_frames` is not specified, vLLM samples 32 frames by default.

  ```bash
  curl http://{ip}:{port}/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "minimax-m3",
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "video_url",
              "video_url": {
                "url": "file:///path/to/video.mp4"
              }
            },
            {
              "type": "text",
              "text": "Briefly describe the main content of this video."
            }
          ]
        }
      ],
      "max_tokens": 512,
      "temperature": 0
    }'
  ```

  Expected result: HTTP 200 response with a JSON body containing non-empty `choices` and generated text describing the video content.

### 7.4 Mixed Image and Video Request

  Start the service with both image and video input enabled. For the following request, use `--limit-mm-per-prompt '{"image":1,"video":1}'`. If the request uses `file://` local video paths, also add `--allowed-local-media-path /` or a narrower allowed directory.

  ```bash
  IMAGE_BASE64="$(base64 -w 0 /path/to/image.jpg)"

  curl http://{ip}:{port}/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d @- <<EOF
  {
    "model": "minimax-m3",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,${IMAGE_BASE64}"}},
          {"type": "video_url", "video_url": {"url": "file:///path/to/video.mp4"}},
          {"type": "text", "text": "Describe the image and video separately, and explain whether they are related."}
        ]
      }
    ],
    "max_tokens": 512,
    "temperature": 0
  }
  EOF
  ```

  Expected result: HTTP 200 response with a JSON body containing non-empty `choices` and generated text that describes the image and video separately and explains whether they are related.

## 8 Accuracy Evaluation

### 8.1 Using AISBench

For detailed instructions, refer to [Using AISBench for accuracy evaluation](../../developer_guide/evaluation/using_ais_bench.md).

### 8.2 Text Evaluation

| Dataset | Hardware | Score | max-model-len | max-num-seqs | max_out_len | batch_size | generation_kwargs |
|---------|----------|-------|---------------|--------------|-------------|------------|-------------------|
| GSM8K   | 8 H20 (96G × 8)     | 96.72 | 65536         | 16           | 49152       | 16         | temperature=1.0, top_p=0.95 |
| GSM8K   | 8 Atlas 800 A3 (64GB × 16)      | 96.36 | 10240         | 16           | 9500        | 20         | temperature=1.0, top_p=0.95 |
| AIME2025 | 8 H20 (96G × 8)     | 95@repeat4 | -        | -            | -           | -          | -                 |
| AIME2025 | 8 Atlas 800 A3 (64GB × 16)      | 93.3@repeat2    | 131072        | 32         | 65536           | 8         | temperature=1.0, top_p=0.95 |
| GPQA-Diamond | 8 H20 (96G × 8)     | 92.42    | 81920      | 64        | 75776       | 8       | temperature=0.6, top_p=0.95 |
| GPQA-Diamond | 8 Atlas 800 A3 (64GB × 16)      | 92.42    | 131072      | 32        | 65536       | 8       | temperature=0.6, top_p=0.95 |
| GPQA-Diamond | 8 950DT products (96GB × 8)      | 92.9    | 133000      | 128       | 131072       | 128       | temperature=0.6, top_p=0.95 |
| MMMU-pro | 8 950DT products (96GB × 8)      | 78.9    | 133000      | 128       | 131072       | 50       | temperature=0.6, top_p=0.95 |

### 8.3 Multimodal Evaluation

MiniMax-M3 multimodal accuracy is evaluated with AISBench. The ViT DP path is optional and can be enabled by adding `--mm-encoder-tp-mode data` to the serving command, but it is not required for all multimodal accuracy runs. For video evaluation, if no frame count is specified in the request or evaluation config, vLLM samples 32 frames by default.

The Video-MME results below are measured on chunk1 and chunk2, not the full dataset.

For Video-MME evaluation, run the vLLM OpenAI-compatible service with video input enabled and use AISBench to send the Video-MME requests. The official AISBench guide may not list Video-MME as a built-in example, so the key MiniMax-M3 settings used here are:

- serve with `--limit-mm-per-prompt '{"image":0,"video":1}'`;
- do not set `media_io_kwargs.video.num_frames`, so vLLM uses the default 32 sampled frames;
- use `max-model-len=90112` and `max_out_len=8192`;
- evaluate Video-MME chunk1 and chunk2, not the full dataset.

The AISBench command used for the Video-MME chunk1+chunk2 evaluation is:

```bash
ais_bench \
  --models vllm_api_general_chat \
  --datasets videomme_subset_1_2.py \
  --mode all \
  --dump-eval-details \
  --merge-ds
```

`videomme_subset_1_2.py` is a local AISBench dataset config derived from the original Video-MME config, such as `videomme_gen.py`. It points `path` to the parquet file filtered from the full Video-MME metadata by the locally available chunk1/chunk2 videos, and points `video_path` to the extracted chunk1/chunk2 `.mp4` directory. This keeps the evaluation lightweight while preserving the standard Video-MME request and scoring flow.

| Dataset | Modality | Tool | Hardware | ViT DP | max-model-len | max_out_len | Input Config | generation_kwargs | Score |
|---------|----------|------|----------|--------|---------------|-------------|--------------|-------------------|-------|
| TextVQA | Image | AISBench | GPU | disabled | 65536 | 512 | `--limit-mm-per-prompt '{"image":1,"video":0}'` | temperature=1.0, top_p=0.95 | 70.82 |
| TextVQA | Image | AISBench | NPU | disabled | 65536 | 512 | `--limit-mm-per-prompt '{"image":1,"video":0}'` | temperature=1.0, top_p=0.95 | 72.75 |
| Video-MME chunk1+chunk2 | Video | AISBench | GPU | - | 90112 | 8192 | `--limit-mm-per-prompt '{"image":0,"video":1}'`, default 32 frames | temperature=1.0, top_p=0.95 | 73.41 |
| Video-MME chunk1+chunk2 | Video | AISBench | NPU | - | 90112 | 8192 | `--limit-mm-per-prompt '{"image":0,"video":1}'`, default 32 frames | temperature=1.0, top_p=0.95 | 74.21 |

## 9 Performance Tuning

> **Note**: The following configurations are validated in specific test environments and are for reference only. The optimal configuration depends on factors such as maximum input/output length, prefix cache hit rate, precision requirements, and deployment machine ratios. It is recommended to refer to Section 9.2 for tuning based on actual conditions.

### 9.1 Recommended Configurations

The recommended configurations are the same as those specified in Chapter 5, "Online Service Deployment."

### 9.2 Tuning Guidelines

#### 9.2.1 General Tuning Reference

Please refer to the [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md) for general tuning methods.

Please refer to the [Feature Matrix](../../user_guide/support_matrix/feature_matrix.md) for detailed feature descriptions.

## 10 FAQ

- **Q: How can I reinstall vLLM Ascend?**

  A: Use the following command to reinstall vLLM Ascend and build it with the dependencies from the current Python environment:

  ```bash
  pip install -v --no-build-isolation -e . -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
  ```

- **Q: What should I do if a video request is slow or times out when `media_io_kwargs.video.num_frames` is not set?**

  A: By default, vLLM samples 32 frames when reading a video. MiniMax-M3 produces many visual tokens per frame, so a 32-frame video significantly increases prefill computation. If the request is slow or times out, explicitly set `media_io_kwargs.video.num_frames` to a smaller value, such as 8 or 16 frames:

  ```json
  {
    "media_io_kwargs": {
      "video": {
        "num_frames": 8
      }
    }
  }
  ```
