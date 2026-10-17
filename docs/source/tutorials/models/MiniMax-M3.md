# MiniMax-M3

## 1 Introduction

!!! note

    The "PD Separation Deployment" mentioned in this document is provided only as an example/reference design to demonstrate the basic process of deploying Prefill and Decode in separated mode. It is not recommended for use in the production environment.

    **For the production environment on the customer side, [MindIE Motor](https://www.hiascend.com/document/detail/en/mindie/latest/quickstart/docs/en/user_guide/quick_start.md) + vLLM-Ascend is recommended.**

MiniMax-M3 is a multimodal large language model that supports text, image, and video inputs. On Ascend, it supports BF16 and W8A8 on A2/A3, Prefill-Decode disaggregation on Atlas 800I A3 (BF16) and 950DT products (MXFP8), thinking mode, reasoning parsing, tool-call parsing, and multimodal inputs.

This document covers supported features, environment and model preparation, single-node deployment, multi-node deployment, PD separation, thinking and parser configuration, functional verification, accuracy evaluation, and troubleshooting.

This document is written based on the vLLM-Ascend main branch. This model is supported on this branch.

## 2 Supported Features

Refer to [Supported Features List](../../user_guide/support_matrix/supported_models.md) for the model support matrix.

Refer to the [Feature Guide](../../user_guide/feature_guide/index.md) for feature configuration instructions.

## 3 Prerequisites

### 3.1 Model Weight

| Weight Version | Hardware Requirements | Download Links | Notes |
| --- | --- | --- | --- |
| `MiniMax-M3` (BF16) | 2 Atlas 800I A3 nodes (64GB × 16 per node) | [ModelScope](https://www.modelscope.cn/collections/MiniMax/MiniMax-M3) | / |
| `MiniMax-M3-w8a8` (W8A8) | 1 Atlas 800I A3 node (64GB × 16) or 1 Atlas 800 A2 node (64GB × 8) | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/MiniMax-M3-w8a8-0626) | / |
| `MiniMax-M3-MXFP8` (MXFP8) | 1 950DT node (96GB × 8) | [ModelScope](https://www.modelscope.cn/models/MiniMax/MiniMax-M3-MXFP8) | Supports a W8A8C8 configuration: C8 is dynamic KV-cache quantization. Set `--kv-cache-dtype fp8` in the serving command to enable FP8 KV cache. |
| `MiniMax-M3-EAGLE3-GQA` | / | [ModelScope](https://www.modelscope.cn/models/Inferact/MiniMax-M3-EAGLE3-GQA) | Draft model for EAGLE3 speculative decoding; uses Grouped Query Attention (GQA). |

It is recommended to place the model weight in a shared cache directory.

### 3.2 Verify Multi-node Communication (Optional)

For multi-node deployment, verify the communication environment by following [Verify Multi-node Communication Environment](../../getting_started/installation.md#installation-multi-node-interconnect).

## 4 Installation

### 4.1 Docker Image Installation

You can use the official all-in-one Docker image. For the available image tags and published versions, refer to [Using Docker](../../getting_started/installation.md#installation-prebuilt-image).

**Step 1:** Download the latest Docker image

```bash
docker pull quay.io/ascend/vllm-ascend:{tag}
```

**Step 2:** Start Docker container

Select the `docker run` command for your hardware platform:

=== "A3 series"

    ```bash
    # Set the vLLM Ascend image name.
    export IMAGE=quay.io/ascend/vllm-ascend:{tag}
    export NAME=minimax-m3-dev

    # Start the container with the variables defined above.
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

=== "950DT products"

    ```bash
    # Set the vLLM Ascend image name.
    export IMAGE=quay.io/ascend/vllm-ascend:{tag}
    export NAME=minimax-m3-dev

    # 950DT products have 8 NPUs and use Device UB.
    docker run --rm \
    --name $NAME \
    --net=host \
    --privileged=true \
    --shm-size=60g \
    --device /dev/davinci_manager \
    --device /dev/hisi_hdc \
    --device /dev/ummu \
    --device /dev/uburma \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci4 \
    --device /dev/davinci5 \
    --device /dev/davinci6 \
    --device /dev/davinci7 \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /etc/hixlep:/etc/hixlep \
    -v /etc/hccn.conf:/etc/hccn.conf \
    -v /var/log/npu/:/usr/slog \
    -v /root/.cache:/root/.cache \
    -it $IMAGE bash
    ```

Adjust data volume mounts (for example, model weights and datasets) according to your environment. On 950DT products, do not add Atlas A3 mounts such as `/dev/devmm_svm` that do not exist on the host.

Expected result: The container is listed with status `Up`. You can also verify the vllm-ascend version inside the container:

```bash
pip show vllm-ascend
```

Expected result: The version information is displayed, matching the pulled image version.

## 5 Online Service Deployment

Start the online serving service with the following command:

Select a platform or weight format using the tabs in each section to switch between deployment examples.

For descriptions of the standard `vllm serve` arguments used in the deployment examples, refer to the [vLLM Serving Arguments documentation](https://docs.vllm.ai/en/latest/cli/serve/#arguments). For Ascend-specific options passed through `--additional-config`, refer to [Additional Configuration](../../user_guide/configuration/additional_config.md). For Ascend-specific environment variables, refer to [Environment Variables](../../user_guide/configuration/env_vars.md).

The `--speculative-config` examples use single-quoted JSON. The `'"${EAGLE3_WEIGHT_PATH}"'` segment expands the draft-weight path within that JSON.

### 5.1 Single-Node Deployment

Single-node deployment completes both Prefill and Decode within the same node. The MiniMax-M3 (BF16) model can be deployed on 1 Atlas 800I A3 (64GB × 16), but dual-node deployment is recommended for BF16 on A3 series; single-node is not recommended. The MiniMax-M3-w8a8 (W8A8) quantized model is recommended for single-node deployment on 1 Atlas 800I A3 (64GB × 16) or 1 Atlas 800 A2 (64GB × 8). The MiniMax-M3-MXFP8 (MXFP8) quantized model can be deployed on 1 950DT products (96GB × 8).

!!! note

    The `--quantization` parameter is **only required for quantized weights** and must match the weight variant:

    - **BF16 weights** (`MiniMax-M3`): do **not** add `--quantization`; the model loads as float.
    - **W8A8 weights** (`MiniMax-M3-w8a8`, ModelSlim format): use `--quantization ascend`. vLLM Ascend can auto-detect this from the checkpoint files, but specifying it explicitly is recommended for clarity.
    - **MXFP8 weights** (`MiniMax-M3-MXFP8`): use `--quantization mxfp8`.

    Adding the wrong `--quantization` value (or adding it to BF16 weights) will cause the checkpoint to be misinterpreted and fail to load.

=== "A3 series (BF16)"

    ```bash
    export VLLM_USE_V2_MODEL_RUNNER=1
    export HCCL_OP_EXPANSION_MODE="AIV"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    vllm serve ${WEIGHT_PATH} \
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
      --host 0.0.0.0 \
      --port 11223 \
      --served-model-name minimax-m3 \
      --trust-remote-code \
      --max-model-len 133000 \
      --tensor-parallel-size 8 \
      --data-parallel-size 2 \
      --enable-expert-parallel \
      --max-num-seqs 16 \
      --distributed_executor_backend "mp" \
      --gpu-memory-utilization 0.92 \
      --reasoning-parser minimax_m3 \
      --limit-mm-per-prompt '{"image":1,"video":0}' \
      --enable-chunked-prefill \
      --enable-prefix-caching \
      --additional-config '{
          "enable_cpu_binding": true,
          "enable_flashcomm1": true,
          "ascend_compilation_config": {
          "fuse_norm_quant": false
          },
          "multistream_overlap_shared_expert": true,
          "weight_nz_mode": 2
      }' > ${LOG_PATH} 2>&1 &
    ```

=== "A3 series (W8A8)"

    ```bash
    export VLLM_USE_V2_MODEL_RUNNER=1
    export HCCL_OP_EXPANSION_MODE="AIV"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    vllm serve ${WEIGHT_PATH} \
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
      --host 0.0.0.0 \
      --port 11223 \
      --served-model-name minimax-m3 \
      --trust-remote-code \
      --quantization ascend \
      --max-model-len 133000 \
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
      --enable-chunked-prefill \
      --enable-prefix-caching \
      --speculative-config '{"model":"'"${EAGLE3_WEIGHT_PATH}"'", "method":"eagle3", "num_speculative_tokens":3}' \
      --additional-config '{
          "enable_cpu_binding": true,
          "enable_flashcomm1": true,
          "ascend_compilation_config": {
            "fuse_norm_quant": false
              },
          "multistream_overlap_shared_expert": true,
          "enable_shared_expert_dp": true,
          "weight_nz_mode": 2
      }' > ${LOG_PATH} 2>&1 &
    ```

=== "950DT products"

    ```bash
    export VLLM_USE_V2_MODEL_RUNNER=1
    export HCCL_BUFFSIZE=512
    export HCCL_OP_EXPANSION_MODE="AIV"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    vllm serve ${WEIGHT_PATH} \
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
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
      --max-model-len 133000 \
      --max-num-batched-tokens 16384 \
      --kv-cache-dtype fp8 \
      --max-num-seqs 500 \
      --enable-chunked-prefill \
      --enable-prefix-caching \
      --async-scheduling \
      --reasoning-parser minimax_m3 \
      --limit-mm-per-prompt '{"image":1,"video":0}' \
      --gpu-memory-utilization 0.92 \
      --additional-config '{"enable_cpu_binding":true,"ascend_compilation_config":{"fuse_qknorm_rope":false,"fuse_norm_quant":false},"multistream_overlap_shared_expert":true,"enable_shared_expert_dp":true,"enable_flashcomm1":true}' \
      --speculative-config '{"model":"'"${EAGLE3_WEIGHT_PATH}"'", "method":"eagle3", "num_speculative_tokens":3, "kv_cache_dtype":"bfloat16"}' \
      --safetensors-load-strategy prefetch > ${LOG_PATH} 2>&1 &
    ```

**Note**: In the script above, `max-num-seqs` represents the maximum number of sequences the scheduler can process in a single iteration. Adjust the `max-num-seqs` parameter dynamically based on actual business.

For text-only deployment, `--limit-mm-per-prompt` can be omitted. For multimodal deployment, configure this parameter according to the actual request shape. For example, use `--limit-mm-per-prompt '{"image":2,"video":0}'` for two-image requests, and use `--limit-mm-per-prompt '{"image":0,"video":1}'` for one-video requests.

### 5.2 Multi-Node Deployment

Deploying the float model on Ascend A2 servers requires at least two nodes. Multi-node deployment on A3 servers without prefill–decode disaggregation is not recommended. Update `WEIGHT_PATH`, `EAGLE3_WEIGHT_PATH`, `LOG_PATH`, `local_ip`, `node0_ip`, and `NETWORK_INTERFACE` based on the actual environment. Set `NETWORK_INTERFACE` to the local communication NIC name (for example, `eth0`); HCCL and Gloo use this value directly.

=== "A3 series (BF16)"

    Run the following command on node 0:

    ```bash
    local_ip="${NODE0_IP}"
    node0_ip="${NODE0_IP}"

    export VLLM_USE_V2_MODEL_RUNNER=1
    export HCCL_IF_IP=$local_ip
    export HCCL_OP_EXPANSION_MODE="AIV"
    export HCCL_SOCKET_IFNAME="${NETWORK_INTERFACE}"
    export GLOO_SOCKET_IFNAME="${NETWORK_INTERFACE}"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    vllm serve ${WEIGHT_PATH} \
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
      --host 0.0.0.0 \
      --port 11223 \
      --served-model-name minimax-m3 \
      --trust-remote-code \
      --max-model-len 133000 \
      --tensor-parallel-size 8 \
      --enable-expert-parallel \
      --max-num-seqs 32 \
      --data-parallel-size 4 \
      --data-parallel-size-local 2 \
      --data-parallel-start-rank 0 \
      --data-parallel-address $node0_ip \
      --distributed_executor_backend "mp" \
      --gpu-memory-utilization 0.92 \
      --reasoning-parser minimax_m3 \
      --limit-mm-per-prompt '{"image":1,"video":0}' \
      --speculative-config '{"model":"'"${EAGLE3_WEIGHT_PATH}"'", "method":"eagle3", "num_speculative_tokens":3}' \
      --enable-chunked-prefill \
      --enable-prefix-caching \
      --additional-config '{"enable_cpu_binding":true, "ascend_compilation_config":{"fuse_norm_quant":false}, "enable_shared_expert_dp":true,"multistream_overlap_shared_expert": true, "weight_nz_mode": 2,"enable_flashcomm1":true}' > ${LOG_PATH} 2>&1 &
    ```

    Run the following command on node 1:

    ```bash
    local_ip="${NODE1_IP}"
    node0_ip="${NODE0_IP}"

    export VLLM_USE_V2_MODEL_RUNNER=1
    export HCCL_IF_IP=$local_ip
    export HCCL_OP_EXPANSION_MODE="AIV"
    export HCCL_SOCKET_IFNAME="${NETWORK_INTERFACE}"
    export GLOO_SOCKET_IFNAME="${NETWORK_INTERFACE}"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    vllm serve ${WEIGHT_PATH} \
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
      --host 0.0.0.0 \
      --port 11223 \
      --served-model-name minimax-m3 \
      --trust-remote-code \
      --headless \
      --max-model-len 133000 \
      --tensor-parallel-size 8 \
      --enable-expert-parallel \
      --max-num-seqs 32 \
      --data-parallel-size 4 \
      --data-parallel-size-local 2 \
      --data-parallel-start-rank 2 \
      --data-parallel-address $node0_ip \
      --distributed_executor_backend "mp" \
      --gpu-memory-utilization 0.92 \
      --reasoning-parser minimax_m3 \
      --limit-mm-per-prompt '{"image":1,"video":0}' \
      --speculative-config '{"model":"'"${EAGLE3_WEIGHT_PATH}"'", "method":"eagle3", "num_speculative_tokens":3}' \
      --enable-chunked-prefill \
      --enable-prefix-caching \
      --additional-config '{"enable_cpu_binding":true, "ascend_compilation_config":{"fuse_norm_quant":false}, "enable_shared_expert_dp":true,"multistream_overlap_shared_expert": true, "weight_nz_mode": 2,"enable_flashcomm1":true}' > ${LOG_PATH} 2>&1 &
    ```

=== "A3 series (W8A8)"

    Run the following command on node 0:

    ```bash
    local_ip="${NODE0_IP}"
    node0_ip="${NODE0_IP}"

    export VLLM_USE_V2_MODEL_RUNNER=1
    export HCCL_IF_IP=$local_ip
    export HCCL_OP_EXPANSION_MODE="AIV"
    export HCCL_SOCKET_IFNAME="${NETWORK_INTERFACE}"
    export GLOO_SOCKET_IFNAME="${NETWORK_INTERFACE}"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    vllm serve ${WEIGHT_PATH} \
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
      --host 0.0.0.0 \
      --port 11223 \
      --served-model-name minimax-m3 \
      --trust-remote-code \
      --quantization ascend \
      --max-model-len 133000 \
      --tensor-parallel-size 4 \
      --enable-expert-parallel \
      --max-num-seqs 32 \
      --data-parallel-size 8 \
      --data-parallel-size-local 4 \
      --data-parallel-start-rank 0 \
      --data-parallel-address $node0_ip \
      --distributed_executor_backend "mp" \
      --gpu-memory-utilization 0.92 \
      --reasoning-parser minimax_m3 \
      --limit-mm-per-prompt '{"image":1,"video":0}' \
      --speculative-config '{"model":"'"${EAGLE3_WEIGHT_PATH}"'", "method":"eagle3", "num_speculative_tokens":3}' \
      --enable-chunked-prefill \
      --enable-prefix-caching \
      --additional-config '{"enable_cpu_binding":true, "ascend_compilation_config":{"fuse_norm_quant":false}, "enable_shared_expert_dp":true,"multistream_overlap_shared_expert": true, "weight_nz_mode": 2,"enable_flashcomm1":true}' > ${LOG_PATH} 2>&1 &
    ```

    Run the following command on node 1:

    ```bash
    local_ip="${NODE1_IP}"
    node0_ip="${NODE0_IP}"

    export VLLM_USE_V2_MODEL_RUNNER=1
    export HCCL_IF_IP=$local_ip
    export HCCL_OP_EXPANSION_MODE="AIV"
    export HCCL_SOCKET_IFNAME="${NETWORK_INTERFACE}"
    export GLOO_SOCKET_IFNAME="${NETWORK_INTERFACE}"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    vllm serve ${WEIGHT_PATH} \
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
      --host 0.0.0.0 \
      --port 11223 \
      --served-model-name minimax-m3 \
      --trust-remote-code \
      --quantization ascend \
      --headless \
      --max-model-len 133000 \
      --tensor-parallel-size 4 \
      --enable-expert-parallel \
      --max-num-seqs 32 \
      --data-parallel-size 8 \
      --data-parallel-size-local 4 \
      --data-parallel-start-rank 4 \
      --data-parallel-address $node0_ip \
      --distributed_executor_backend "mp" \
      --gpu-memory-utilization 0.92 \
      --reasoning-parser minimax_m3 \
      --limit-mm-per-prompt '{"image":1,"video":0}' \
      --speculative-config '{"model":"'"${EAGLE3_WEIGHT_PATH}"'", "method":"eagle3", "num_speculative_tokens":3}' \
      --enable-chunked-prefill \
      --enable-prefix-caching \
      --additional-config '{"enable_cpu_binding":true, "ascend_compilation_config":{"fuse_norm_quant":false}, "enable_shared_expert_dp":true,"multistream_overlap_shared_expert": true, "weight_nz_mode": 2,"enable_flashcomm1":true}' > ${LOG_PATH} 2>&1 &
    ```

### 5.3 Sub-1M-Token Context Deployment

The following PD examples cover contexts below 1M tokens and use `--max-model-len 133000`. Choose ordinary PD or add a KV cache pool with the same serving topology. For the separate 1M-context configurations, see [Section 5.4](#54-1m-token-context-deployment).

#### 5.3.1 Prefill-Decode Disaggregation

This section deploys MiniMax-M3 PD disaggregation as 1P1D: on Atlas 800I A3 it spans three nodes (one Prefill node and two Decode nodes); on 950DT it runs on two nodes, and the 950DT example below also describes scaling the same 1P1D deployment to three nodes. Here, 1P1D means one Prefill service group and one Decode service group. The Decode group can span two machines while remaining one global DP group.

PD disaggregation separates Prefill and Decode into different service groups. Prefill nodes process large prompt chunks, Decode nodes serve token generation, and a proxy forwards requests between them. Use Mooncake for KV cache transfer. Refer to [Mooncake](../features/pd_disaggregation_mooncake_multi_node.md) for the general PD disaggregation workflow.

The launch pattern is: prepare `launch_online_dp.py` and a role-specific `run_dp_template.sh` on each node, then start a load-balance proxy after every engine prints `Application startup complete`. The launcher below extends the repository example with `--pp-size`: on A3, Prefill uses `DP=1`, `TP=4`, and pipeline parallel (`PP=4`) with a `16,15,15,14` split of the 60 transformer layers, while the 950DT products MXFP8 launch uses `PP=1` with `DP=2` on both roles. Each DP rank occupies `tp_size * pp_size` NPUs.

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
    env = os.environ.copy()
    env.update(
        VISIBLE_DEVICES=visible_devices,
        SERVICE_PORT=str(vllm_engine_port),
        DP_SIZE=str(dp_size),
        DP_RANK=str(dp_rank),
        DP_ADDRESS=dp_address,
        DP_RPC_PORT=dp_rpc_port,
        TP_SIZE=str(tp_size),
        PP_SIZE=str(pp_size),
    )
    subprocess.run(["bash", "./run_dp_template.sh"], env=env, check=True)


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

The templates use named environment variables instead of positional shell arguments. Replace the `<...>` IP, NIC and weight-path placeholders in each template, or export those variables in the shell before launching. The numeric defaults below describe one engine; `launch_online_dp.py` overrides the per-rank values when it starts all local engines. You do not need to calculate a different port or device list for every rank. When using the launcher, edit `export DP_SIZE=...`, `export TP_SIZE=...`, `export PP_SIZE=...` and the other values in the node-specific launch block; these take precedence over the `${VARIABLE:-default}` values in the template.

| Variable | Meaning | How it is set |
| -------- | ------- | ------------- |
| `LOCAL_IP` | Current node's communication IP | Fill in `<CURRENT_NODE_IP>` separately on each node. |
| `NIC_NAME` | NIC associated with `LOCAL_IP` | Fill in `<CURRENT_NODE_NIC>`. |
| `MODEL_PATH` / `EAGLE3_WEIGHT_PATH` | Target / draft weight directories | Use the same mounted paths on every node. |
| `DP_SIZE` / `TP_SIZE` / `PP_SIZE` | Global DP size / TP and PP size within one rank | Numeric values are exported in each launch command below. |
| `LOCAL_DP_SIZE` | Number of engines on the current node | Passed to the launcher as `--dp-size-local`. |
| `DP_RANK_START` | First global DP rank on this node | Passed as `--dp-rank-start`; differs between the two Decode nodes. |
| `DP_ADDRESS` | DP coordinator IP for this role | Prefill's IP for P; the first Decode node's IP for every D node. |
| `DP_RPC_PORT` | DP coordinator port | All nodes in the same role use the same value. |
| `SERVICE_PORT` | First local API port in the launch shell | The launcher adds the local rank index for each engine. |
| `DP_RANK` | Engine's global DP rank | The launcher sets `DP_RANK_START + local_rank`. |
| `VISIBLE_DEVICES` | NPU IDs for one engine | The launcher assigns `TP_SIZE × PP_SIZE` consecutive IDs per local rank. |

Save the Prefill template as `run_dp_template.sh` on the P node and the Decode template under that name on each D node. Run the launcher from the directory containing the matching template. If using a shared filesystem, keep P and D templates in separate directories. Changing TP/DP/PP also requires updating the matching `prefill` and `decode` layouts in both roles' `--kv-transfer-config`.

The following values match the configurations in Sections 5.3.1 and 5.3.2. P/D node counts refer to machines; `DP_SIZE` is the global number of ranks for that role.

| Section / platform | Role / node | `DP_SIZE` | `TP_SIZE` | `PP_SIZE` | `LOCAL_DP_SIZE` | `DP_RANK_START` | First `SERVICE_PORT` |
| ------------------ | ----------- | --------- | --------- | --------- | --------------- | --------------- | -------------------- |
| 5.3.1 A3, three nodes | P | 1 | 4 | 4 | 1 | 0 | 31050 |
| 5.3.1 A3, three nodes | First D | 8 | 4 | 1 | 4 | 0 | 31260 |
| 5.3.1 A3, three nodes | Second D | 8 | 4 | 1 | 4 | 4 | 31260 |
| 5.3.1 950DT, two nodes | P | 2 | 4 | 1 | 2 | 0 | 31050 |
| 5.3.1 950DT, two nodes | D | 2 | 4 | 1 | 2 | 0 | 31060 |
| 5.3.1 950DT, three nodes | P | 2 | 4 | 1 | 2 | 0 | 31050 |
| 5.3.1 950DT, three nodes | First D | 4 | 4 | 1 | 2 | 0 | 31060 |
| 5.3.1 950DT, three nodes | Second D | 4 | 4 | 1 | 2 | 2 | 31062 |
| 5.3.2 A3, three nodes | P | 1 | 4 | 4 | 1 | 0 | 31050 |
| 5.3.2 A3, three nodes | First D | 8 | 4 | 1 | 4 | 0 | 31260 |
| 5.3.2 A3, three nodes | Second D | 8 | 4 | 1 | 4 | 4 | 31260 |
| 5.3.2 950DT, two nodes | P | 2 | 4 | 1 | 2 | 0 | 31050 |
| 5.3.2 950DT, two nodes | D | 2 | 4 | 1 | 2 | 0 | 31060 |

=== "A3 series (W8A8)"

    Prefill-Decode disaggregation can be deployed on 3 Atlas 800I A3 (64GB × 24) for `MiniMax-M3-w8a8` (W8A8) with `MiniMax-M3-EAGLE3-GQA`.


    1. Prefill node

    ```bash
    unset http_proxy https_proxy ftp_proxy

    # Replace IP, NIC and weight-path placeholders for this node.
    export LOCAL_IP="${LOCAL_IP:-<CURRENT_NODE_IP>}"
    export NIC_NAME="${NIC_NAME:-<CURRENT_NODE_NIC>}"
    export MODEL_PATH="${MODEL_PATH:-<MINIMAX_M3_W8A8_PATH>}"
    export EAGLE3_WEIGHT_PATH="${EAGLE3_WEIGHT_PATH:-<MINIMAX_M3_EAGLE3_GQA_PATH>}"

    # Role defaults; the launcher overrides these for each engine.
    export DP_SIZE=${DP_SIZE:-1}
    export TP_SIZE=${TP_SIZE:-4}
    export PP_SIZE=${PP_SIZE:-4}
    export DP_RANK=${DP_RANK:-0}
    export DP_ADDRESS="${DP_ADDRESS:-$LOCAL_IP}"
    export DP_RPC_PORT=${DP_RPC_PORT:-6884}
    export SERVICE_PORT=${SERVICE_PORT:-31050}
    export VISIBLE_DEVICES="${VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"

    export VLLM_PP_LAYER_PARTITION="16,15,15,14"
    export VLLM_USE_V2_MODEL_RUNNER=1
    export HCCL_BUFFSIZE=1024
    export HCCL_IF_IP=$LOCAL_IP
    export HCCL_OP_EXPANSION_MODE="AIV"
    export HCCL_SOCKET_IFNAME=$NIC_NAME
    export ASCEND_RT_VISIBLE_DEVICES=$VISIBLE_DEVICES
    # If Mooncake is installed in a non-standard path, set this before startup.
    if [ -n "${MOONCAKE_LIB_DIRS:-}" ]; then
        export LD_LIBRARY_PATH="${MOONCAKE_LIB_DIRS}:${LD_LIBRARY_PATH:-}"
    fi
    export GLOO_SOCKET_IFNAME=$NIC_NAME
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    # For better prefill performance, keep max_num_batched_tokens at 8192 for input
    # lengths >= 8192 tokens; for shorter inputs, set it to the input token length.
    vllm serve "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port "$SERVICE_PORT" \
        --data-parallel-size "$DP_SIZE" \
        --data-parallel-rank "$DP_RANK" \
        --data-parallel-address "$DP_ADDRESS" \
        --data-parallel-rpc-port "$DP_RPC_PORT" \
        --tensor-parallel-size "$TP_SIZE" \
        --pipeline-parallel-size "$PP_SIZE" \
        --enforce-eager \
        --distributed-executor-backend mp \
        --served-model-name minimax-m3 \
        --enable-expert-parallel \
        --seed 1024 \
        --max-model-len 133000 \
        --max-num-seqs 32 \
        --max-num-batched-tokens 8192 \
        --enable-chunked-prefill \
        --enable-prefix-caching \
        --trust-remote-code \
        --quantization ascend \
        --gpu-memory-utilization 0.92 \
        --limit-mm-per-prompt '{"image":1,"video":0}' \
        --reasoning-parser minimax_m3 \
        --additional-config '{"enable_cpu_binding":true,"ascend_compilation_config":{"fuse_norm_quant":false},"multistream_overlap_shared_expert":true,"weight_nz_mode":2,"enable_shared_expert_dp":true,"enable_flashcomm1":true}' \
        --speculative-config '{"model":"'"${EAGLE3_WEIGHT_PATH}"'", "method":"eagle3", "num_speculative_tokens":3}' \
        --kv-transfer-config \
        '{
            "kv_connector":"MooncakeConnectorV1",
            "kv_role":"kv_producer",
            "kv_port":"36000",
            "engine_id":"0",
            "kv_connector_extra_config":{
                "use_ascend_direct":true,
                "prefill":{"dp_size":1,"tp_size":4,"pp_size":4,"pp_layer_partition":"16,15,15,14"},
                "decode":{"dp_size":8,"tp_size":4,"pp_size":1}
            }
        }'
    ```

    2. Decode node

    ```bash
    unset http_proxy https_proxy ftp_proxy all_proxy HTTP_PROXY HTTPS_PROXY FTP_PROXY ALL_PROXY

    # Replace IP, NIC and weight-path placeholders for this node.
    export LOCAL_IP="${LOCAL_IP:-<CURRENT_NODE_IP>}"
    export NIC_NAME="${NIC_NAME:-<CURRENT_NODE_NIC>}"
    export MODEL_PATH="${MODEL_PATH:-<MINIMAX_M3_W8A8_PATH>}"
    export EAGLE3_WEIGHT_PATH="${EAGLE3_WEIGHT_PATH:-<MINIMAX_M3_EAGLE3_GQA_PATH>}"

    # Role defaults; the launcher overrides these for each engine.
    export DP_SIZE=${DP_SIZE:-8}
    export TP_SIZE=${TP_SIZE:-4}
    export PP_SIZE=${PP_SIZE:-1}
    export DP_RANK=${DP_RANK:-0}
    export DP_ADDRESS="${DP_ADDRESS:-<DECODE_COORDINATOR_IP>}"
    export DP_RPC_PORT=${DP_RPC_PORT:-31422}
    export SERVICE_PORT=${SERVICE_PORT:-31260}
    export VISIBLE_DEVICES="${VISIBLE_DEVICES:-0,1,2,3}"

    export VLLM_USE_V2_MODEL_RUNNER=1
    export HCCL_BUFFSIZE=1024
    export HCCL_IF_IP=$LOCAL_IP
    export HCCL_OP_EXPANSION_MODE="AIV"
    export HCCL_SOCKET_IFNAME=$NIC_NAME
    export ASCEND_RT_VISIBLE_DEVICES=$VISIBLE_DEVICES
    # If Mooncake is installed in a non-standard path, set this before startup.
    if [ -n "${MOONCAKE_LIB_DIRS:-}" ]; then
        export LD_LIBRARY_PATH="${MOONCAKE_LIB_DIRS}:${LD_LIBRARY_PATH:-}"
    fi
    export GLOO_SOCKET_IFNAME=$NIC_NAME
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    vllm serve "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port "$SERVICE_PORT" \
        --data-parallel-size "$DP_SIZE" \
        --data-parallel-rank "$DP_RANK" \
        --data-parallel-address "$DP_ADDRESS" \
        --data-parallel-rpc-port "$DP_RPC_PORT" \
        --tensor-parallel-size "$TP_SIZE" \
        --pipeline-parallel-size "$PP_SIZE" \
        --enable-expert-parallel \
        --enable-chunked-prefill \
        --enable-prefix-caching \
        --seed 1024 \
        --served-model-name minimax-m3 \
        --reasoning-parser minimax_m3 \
        --distributed-executor-backend mp \
        --max-model-len 133000 \
        --max-num-batched-tokens 32768 \
        --trust-remote-code \
        --quantization ascend \
        --max-num-seqs 128 \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
        --gpu-memory-utilization 0.92 \
        --limit-mm-per-prompt '{"image":1,"video":0}' \
        --additional-config '{"enable_cpu_binding":true,"ascend_compilation_config":{"fuse_norm_quant":false},"multistream_overlap_shared_expert":true,"weight_nz_mode":2,"enable_shared_expert_dp":true,"enable_fused_mc2":0,"enable_flashcomm1":true}' \
        --speculative-config '{"model":"'"${EAGLE3_WEIGHT_PATH}"'", "method":"eagle3", "num_speculative_tokens":3}' \
        --kv-transfer-config \
        '{
            "kv_connector":"MooncakeConnectorV1",
            "kv_buffer_device":"npu",
            "kv_role":"kv_consumer",
            "kv_port":"36100",
            "engine_id":"minimax-m3-decode-dp'"$DP_RANK"'",
            "kv_connector_extra_config":{
                "use_ascend_direct":true,
                "prefill":{"dp_size":1,"tp_size":4,"pp_size":4,"pp_layer_partition":"16,15,15,14"},
                "decode":{"dp_size":8,"tp_size":4,"pp_size":1}
            }
        }'
    ```

    For short-input workloads, tune `--max-num-batched-tokens` using [Section 9.2](#92-tuning-guidelines).
    Once the preparation is done, start the server with the following command on each node:

    1. Prefill node

    ```bash
    export PREFILL_IP="<PREFILL_NODE_IP>"
    export LOCAL_IP="$PREFILL_IP"
    export DP_SIZE=1
    export TP_SIZE=4
    export PP_SIZE=4
    export LOCAL_DP_SIZE=1
    export DP_RANK_START=0
    export DP_ADDRESS="$PREFILL_IP"
    export DP_RPC_PORT=6884
    export SERVICE_PORT=31050

    python launch_online_dp.py \
        --dp-size "$DP_SIZE" --tp-size "$TP_SIZE" --pp-size "$PP_SIZE" \
        --dp-size-local "$LOCAL_DP_SIZE" --dp-rank-start "$DP_RANK_START" \
        --dp-address "$DP_ADDRESS" --dp-rpc-port "$DP_RPC_PORT" \
        --vllm-start-port "$SERVICE_PORT"
    ```

    This starts one Prefill API server on port `31050`. Wait until it prints `Application startup complete`.

    2. First Decode node (`DECODE0_IP`, global ranks 0–3)

    ```bash
    export DECODE0_IP="<FIRST_DECODE_NODE_IP>"
    export LOCAL_IP="$DECODE0_IP"
    export DP_SIZE=8
    export TP_SIZE=4
    export PP_SIZE=1
    export LOCAL_DP_SIZE=4
    export DP_RANK_START=0
    export DP_ADDRESS="$DECODE0_IP"
    export DP_RPC_PORT=31422
    export SERVICE_PORT=31260

    python launch_online_dp.py \
        --dp-size "$DP_SIZE" --tp-size "$TP_SIZE" --pp-size "$PP_SIZE" \
        --dp-size-local "$LOCAL_DP_SIZE" --dp-rank-start "$DP_RANK_START" \
        --dp-address "$DP_ADDRESS" --dp-rpc-port "$DP_RPC_PORT" \
        --vllm-start-port "$SERVICE_PORT"
    ```


    3. Second Decode node (`DECODE1_IP`, global ranks 4–7)

    ```bash
    export DECODE0_IP="<FIRST_DECODE_NODE_IP>"
    export DECODE1_IP="<SECOND_DECODE_NODE_IP>"
    export LOCAL_IP="$DECODE1_IP"
    export DP_SIZE=8
    export TP_SIZE=4
    export PP_SIZE=1
    export LOCAL_DP_SIZE=4
    export DP_RANK_START=4
    export DP_ADDRESS="$DECODE0_IP"
    export DP_RPC_PORT=31422
    export SERVICE_PORT=31260

    python launch_online_dp.py \
        --dp-size "$DP_SIZE" --tp-size "$TP_SIZE" --pp-size "$PP_SIZE" \
        --dp-size-local "$LOCAL_DP_SIZE" --dp-rank-start "$DP_RANK_START" \
        --dp-address "$DP_ADDRESS" --dp-rpc-port "$DP_RPC_PORT" \
        --vllm-start-port "$SERVICE_PORT"
    ```

    Set `LOCAL_IP` and `NIC_NAME` in the Decode template for each node separately. Both Decode nodes use `DECODE0_IP` as the DP coordinator address. Each node exposes API ports `31260` through `31263`; wait for all eight DP ranks to print `Application startup complete`.

    To set up request forwarding, run the following script on a node that can reach every Prefill and Decode API port. You can get the proxy program in the repository's examples: [load_balance_proxy_server_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py). For A3 1P1D, the proxy forwards requests to one Prefill node and eight DP ranks across three nodes.

    ```bash
    export PREFILL_IP="<PREFILL_NODE_IP>"
    export DECODE0_IP="<FIRST_DECODE_NODE_IP>"
    export DECODE1_IP="<SECOND_DECODE_NODE_IP>"

    unset http_proxy https_proxy ftp_proxy all_proxy HTTP_PROXY HTTPS_PROXY FTP_PROXY ALL_PROXY

    python load_balance_proxy_server_example.py \
    --port 8009 \
    --host $PREFILL_IP \
    --prefiller-hosts \
        $PREFILL_IP \
    --prefiller-ports \
        31050 \
    --decoder-hosts \
        $DECODE0_IP $DECODE0_IP $DECODE0_IP $DECODE0_IP \
        $DECODE1_IP $DECODE1_IP $DECODE1_IP $DECODE1_IP \
    --decoder-ports \
        31260 31261 31262 31263 31260 31261 31262 31263
    ```

    The service is then accessible at `<proxy_ip>:8009`. For PD disaggregation, use this proxy endpoint in Section 7.

    To use MindIE PyMotor instead of the example proxy, follow [Deploy vLLM-Ascend with MindIE-Motor](../../user_guide/deployment_guide/using_mindie_motor.md) and register the same one Prefill and eight Decode endpoints. Send benchmark requests to the Motor inference endpoint. Changing the request router does not require changing the P/D parallel sizes or scheduler limits above.

=== "950DT products"

    Prefill-Decode disaggregation can be deployed on two 950DT nodes (96GB × 8, 1P1D) for `MiniMax-M3-MXFP8` with `MiniMax-M3-EAGLE3-GQA`, and scales to three nodes while remaining 1P1D (one Prefill service group and one Decode service group; see below). On SuperPod systems (UB fabric), export `ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'` as in the templates below; mounting `/etc/hixlep/` applies to RoCE-attached 950DT instead. The examples explicitly enable Model Runner V2.

    Set `PREFILL_IP` and `DECODE_IP` to the Prefill and Decode communication IPs. For the three-node variant, also set `DECODE0_IP` and `DECODE1_IP`. Set `LOCAL_IP` and `NIC_NAME` separately in each node's template.

    1. Prefill node

    ```bash
    unset ftp_proxy https_proxy http_proxy all_proxy
    unset FTP_PROXY HTTPS_PROXY HTTP_PROXY ALL_PROXY

    # Replace IP, NIC and weight-path placeholders for this node.
    export LOCAL_IP="${LOCAL_IP:-<CURRENT_NODE_IP>}"
    export NIC_NAME="${NIC_NAME:-<CURRENT_NODE_NIC>}"
    export MODEL_PATH="${MODEL_PATH:-<MINIMAX_M3_MXFP8_PATH>}"
    export EAGLE3_WEIGHT_PATH="${EAGLE3_WEIGHT_PATH:-<MINIMAX_M3_EAGLE3_GQA_PATH>}"

    # Role defaults; the launcher overrides these for each engine.
    export DP_SIZE=${DP_SIZE:-2}
    export TP_SIZE=${TP_SIZE:-4}
    export PP_SIZE=${PP_SIZE:-1}
    export DP_RANK=${DP_RANK:-0}
    export DP_ADDRESS="${DP_ADDRESS:-$LOCAL_IP}"
    export DP_RPC_PORT=${DP_RPC_PORT:-6884}
    export SERVICE_PORT=${SERVICE_PORT:-31050}
    export VISIBLE_DEVICES="${VISIBLE_DEVICES:-0,1,2,3}"

    # SuperPod (UB fabric): Mooncake auto-negotiates local endpoints. On RoCE-attached
    # 950DT products, mount /etc/hixlep and use ascend_local_comm_res_path instead.
    export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'

    export VLLM_USE_V2_MODEL_RUNNER=1
    export HCCL_BUFFSIZE=256
    export HCCL_IF_IP=$LOCAL_IP
    export HCCL_OP_EXPANSION_MODE="AIV"
    export HCCL_SOCKET_IFNAME=$NIC_NAME
    export ASCEND_RT_VISIBLE_DEVICES=$VISIBLE_DEVICES
    export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH
    export GLOO_SOCKET_IFNAME=$NIC_NAME
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    vllm serve "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port "$SERVICE_PORT" \
        --data-parallel-size "$DP_SIZE" \
        --data-parallel-rank "$DP_RANK" \
        --data-parallel-address "$DP_ADDRESS" \
        --data-parallel-rpc-port "$DP_RPC_PORT" \
        --tensor-parallel-size "$TP_SIZE" \
        --pipeline-parallel-size "$PP_SIZE" \
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
        --speculative-config '{"model":"'"${EAGLE3_WEIGHT_PATH}"'", "method":"eagle3", "num_speculative_tokens":3, "kv_cache_dtype":"bfloat16"}' \
        --enforce-eager \
        --enable-chunked-prefill \
        --enable-prefix-caching \
        --additional-config '{"enable_cpu_binding":true,"ascend_compilation_config":{"fuse_qknorm_rope":true,"fuse_norm_quant":true},"multistream_overlap_shared_expert":true,"enable_shared_expert_dp":true,"enable_flashcomm1":true}' \
        --kv-transfer-config \
        '{
            "kv_connector":"MooncakeConnectorV1",
            "kv_role":"kv_producer",
            "kv_port":"30000",
            "engine_id":"0",
            "kv_connector_extra_config":{
                "use_ascend_direct":true,
                "prefill":{"dp_size":2,"tp_size":4,"pp_size":1},
                "decode":{"dp_size":2,"tp_size":4,"pp_size":1}
            }
        }'
    ```

    2. Decode node

    ```bash
    unset ftp_proxy https_proxy http_proxy all_proxy
    unset FTP_PROXY HTTPS_PROXY HTTP_PROXY ALL_PROXY

    # Replace IP, NIC and weight-path placeholders for this node.
    export LOCAL_IP="${LOCAL_IP:-<CURRENT_NODE_IP>}"
    export NIC_NAME="${NIC_NAME:-<CURRENT_NODE_NIC>}"
    export MODEL_PATH="${MODEL_PATH:-<MINIMAX_M3_MXFP8_PATH>}"
    export EAGLE3_WEIGHT_PATH="${EAGLE3_WEIGHT_PATH:-<MINIMAX_M3_EAGLE3_GQA_PATH>}"

    # Role defaults; the launcher overrides these for each engine.
    export DP_SIZE=${DP_SIZE:-2}
    export TP_SIZE=${TP_SIZE:-4}
    export PP_SIZE=${PP_SIZE:-1}
    export DP_RANK=${DP_RANK:-0}
    export DP_ADDRESS="${DP_ADDRESS:-<DECODE_COORDINATOR_IP>}"
    export DP_RPC_PORT=${DP_RPC_PORT:-5964}
    export SERVICE_PORT=${SERVICE_PORT:-31060}
    export VISIBLE_DEVICES="${VISIBLE_DEVICES:-0,1,2,3}"

    # SuperPod (UB fabric): Mooncake auto-negotiates local endpoints. On RoCE-attached
    # 950DT products, mount /etc/hixlep and use ascend_local_comm_res_path instead.
    export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'

    export VLLM_USE_V2_MODEL_RUNNER=1
    export HCCL_BUFFSIZE=2048
    export HCCL_IF_IP=$LOCAL_IP
    export HCCL_OP_EXPANSION_MODE="AIV"
    export HCCL_SOCKET_IFNAME=$NIC_NAME
    export ASCEND_RT_VISIBLE_DEVICES=$VISIBLE_DEVICES
    export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH
    export GLOO_SOCKET_IFNAME=$NIC_NAME
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    vllm serve "$MODEL_PATH" \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
        --host 0.0.0.0 \
        --port "$SERVICE_PORT" \
        --data-parallel-size "$DP_SIZE" \
        --data-parallel-rank "$DP_RANK" \
        --data-parallel-address "$DP_ADDRESS" \
        --data-parallel-rpc-port "$DP_RPC_PORT" \
        --tensor-parallel-size "$TP_SIZE" \
        --pipeline-parallel-size "$PP_SIZE" \
        --enable-expert-parallel \
        --seed 1024 \
        --served-model-name minimax-m3 \
        --reasoning-parser minimax_m3 \
        --safetensors-load-strategy prefetch \
        --distributed-executor-backend mp \
        --max-model-len 133000 \
        --max-num-batched-tokens 32768 \
        --trust-remote-code \
        --max-num-seqs 256 \
        --gpu-memory-utilization 0.92 \
        --dtype bfloat16 \
        --quantization mxfp8 \
        --kv-cache-dtype fp8 \
        --speculative-config '{"model":"'"${EAGLE3_WEIGHT_PATH}"'", "method":"eagle3", "num_speculative_tokens":3, "kv_cache_dtype":"bfloat16"}' \
        --additional-config '{"enable_cpu_binding":true,"ascend_compilation_config":{"fuse_norm_quant":false},"multistream_overlap_shared_expert":true,"enable_shared_expert_dp":true,"enable_flashcomm1":true}' \
        --kv-transfer-config \
        '{
            "kv_connector":"MooncakeConnectorV1",
            "kv_role":"kv_consumer",
            "kv_port":"26900",
            "engine_id":"1",
            "kv_connector_extra_config":{
                "use_ascend_direct":true,
                "prefill":{"dp_size":2,"tp_size":4,"pp_size":1},
                "decode":{"dp_size":2,"tp_size":4,"pp_size":1}
            }
        }'
    ```

    Once the preparation is done, start the server with the following command on each node:

    1. Prefill node

    ```bash
    export PREFILL_IP="<PREFILL_NODE_IP>"
    export LOCAL_IP="$PREFILL_IP"
    export DP_SIZE=2
    export TP_SIZE=4
    export PP_SIZE=1
    export LOCAL_DP_SIZE=2
    export DP_RANK_START=0
    export DP_ADDRESS="$PREFILL_IP"
    export DP_RPC_PORT=6884
    export SERVICE_PORT=31050

    python launch_online_dp.py \
        --dp-size "$DP_SIZE" --tp-size "$TP_SIZE" --pp-size "$PP_SIZE" \
        --dp-size-local "$LOCAL_DP_SIZE" --dp-rank-start "$DP_RANK_START" \
        --dp-address "$DP_ADDRESS" --dp-rpc-port "$DP_RPC_PORT" \
        --vllm-start-port "$SERVICE_PORT"
    ```

    This starts two Prefill API servers on ports `31050` and `31051`. Wait until both ranks print `Application startup complete`.

    2. Decode node

    ```bash
    export DECODE_IP="<DECODE_NODE_IP>"
    export LOCAL_IP="$DECODE_IP"
    export DP_SIZE=2
    export TP_SIZE=4
    export PP_SIZE=1
    export LOCAL_DP_SIZE=2
    export DP_RANK_START=0
    export DP_ADDRESS="$DECODE_IP"
    export DP_RPC_PORT=5964
    export SERVICE_PORT=31060

    python launch_online_dp.py \
        --dp-size "$DP_SIZE" --tp-size "$TP_SIZE" --pp-size "$PP_SIZE" \
        --dp-size-local "$LOCAL_DP_SIZE" --dp-rank-start "$DP_RANK_START" \
        --dp-address "$DP_ADDRESS" --dp-rpc-port "$DP_RPC_PORT" \
        --vllm-start-port "$SERVICE_PORT"
    ```

    This starts two Decode API servers on ports `31060` and `31061`.

    To set up request forwarding, run the following script on a node that can reach every Prefill and Decode API port. You can get the proxy program in the repository's examples: [load_balance_proxy_server_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py). For 950DT products 1P1D, the proxy forwards requests to 2 Prefill ranks and 2 DP ranks.

    ```bash
    export PREFILL_IP="<PREFILL_NODE_IP>"
    export DECODE_IP="<DECODE_NODE_IP>"

    unset ftp_proxy
    unset https_proxy
    unset http_proxy

    python load_balance_proxy_server_example.py \
    --port 8009 \
    --host $PREFILL_IP \
    --prefiller-hosts \
        $PREFILL_IP $PREFILL_IP \
    --prefiller-ports \
        31050 31051 \
    --decoder-hosts \
        $DECODE_IP $DECODE_IP \
    --decoder-ports \
        31060 31061
    ```

    The service is then accessible at `<proxy_ip>:8009`. For PD disaggregation, use this proxy endpoint in Section 7.

    **Scaling to three nodes (still 1P1D)**

    The deployment above scales to three nodes: one Prefill node (DP2 × TP4) plus two Decode nodes forming one global DP4 group (two local DP ranks per Decode node, TP4 each; 24 NPUs in total). Only one template change is required: in **both** the Prefill and Decode `run_dp_template.sh`, update the Mooncake extra config to describe the new Decode group — `"decode":{"dp_size":4,"tp_size":4,"pp_size":1}` (Prefill stays `dp_size: 2`). Also set `LOCAL_IP` in each node's template to that node's compute-network IP.

    Start the Prefill node exactly as in the two-node example. Then start the first Decode node (`DECODE0_IP`, global DP ranks 0–1), which also acts as the DP group master:

    ```bash
    export DECODE0_IP="<FIRST_DECODE_NODE_IP>"
    export LOCAL_IP="$DECODE0_IP"
    export DP_SIZE=4
    export TP_SIZE=4
    export PP_SIZE=1
    export LOCAL_DP_SIZE=2
    export DP_RANK_START=0
    export DP_ADDRESS="$DECODE0_IP"
    export DP_RPC_PORT=5964
    export SERVICE_PORT=31060

    python launch_online_dp.py \
        --dp-size "$DP_SIZE" --tp-size "$TP_SIZE" --pp-size "$PP_SIZE" \
        --dp-size-local "$LOCAL_DP_SIZE" --dp-rank-start "$DP_RANK_START" \
        --dp-address "$DP_ADDRESS" --dp-rpc-port "$DP_RPC_PORT" \
        --vllm-start-port "$SERVICE_PORT"
    ```

    Wait for rank 0 to print `Started DP Coordinator process`. From another shell, start the second Decode node (`DECODE1_IP`, global DP ranks 2–3), which joins the same DP group:

    ```bash
    export DECODE0_IP="<FIRST_DECODE_NODE_IP>"
    export DECODE1_IP="<SECOND_DECODE_NODE_IP>"
    export LOCAL_IP="$DECODE1_IP"
    export DP_SIZE=4
    export TP_SIZE=4
    export PP_SIZE=1
    export LOCAL_DP_SIZE=2
    export DP_RANK_START=2
    export DP_ADDRESS="$DECODE0_IP"
    export DP_RPC_PORT=5964
    export SERVICE_PORT=31062

    python launch_online_dp.py \
        --dp-size "$DP_SIZE" --tp-size "$TP_SIZE" --pp-size "$PP_SIZE" \
        --dp-size-local "$LOCAL_DP_SIZE" --dp-rank-start "$DP_RANK_START" \
        --dp-address "$DP_ADDRESS" --dp-rpc-port "$DP_RPC_PORT" \
        --vllm-start-port "$SERVICE_PORT"
    ```

    The first Decode node serves API ports `31060`/`31061` and the second `31062`/`31063`. Wait until all four DP ranks print `Application startup complete`. The template base `kv_port` (`26900`) and `engine_id` (`"1"`) can remain the same on both Decode nodes. Keep the global DP ranks distinct (0–1 and 2–3); Mooncake derives handshake ports using the global DP-rank offset. Start the proxy only after all six engines are up (the example proxy waits only a few seconds per backend), registering two Prefill ranks and four DP ranks:

    ```bash
    export PREFILL_IP="<PREFILL_NODE_IP>"
    export DECODE0_IP="<FIRST_DECODE_NODE_IP>"
    export DECODE1_IP="<SECOND_DECODE_NODE_IP>"

    python load_balance_proxy_server_example.py \
    --port 8009 \
    --host $PREFILL_IP \
    --prefiller-hosts \
        $PREFILL_IP $PREFILL_IP \
    --prefiller-ports \
        31050 31051 \
    --decoder-hosts \
        $DECODE0_IP $DECODE0_IP $DECODE1_IP $DECODE1_IP \
    --decoder-ports \
        31060 31061 31062 31063
    ```

    In a contributor-verified text-only run using this three-node layout with MXFP8, EAGLE3, fp8 KV cache and the optional `--language-model-only` flag, all 232 requests succeeded at 128k input / 1k output and concurrency 58. Average TPOT was 27.3 ms and aggregate output throughput was about 1840 tokens/s. In that setup, enabling Decode FlashComm1 increased output throughput from 1807.9 to 1840.7 tokens/s (about 1.8%). These measurements describe that workload and configuration; the launch templates leave `--language-model-only` optional.

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

- `--language-model-only` (optional): For text-only tests, add this flag as needed based on the available KV cache capacity to avoid reserving memory for multimodal processing. This flag can be used together with FLASHCOMM1 for text-only serving.
- `--pipeline-parallel-size` (A3 Prefill: `4`): Splits the 60 MiniMax-M3 layers across four pipeline stages. A3 sets `VLLM_PP_LAYER_PARTITION=16,15,15,14` and also writes `pp_layer_partition` into the Mooncake extra config. With `TP=4`, use `DP=1` to fit all four stages on one 16-NPU node. The 950DT products launch uses `--pp-size 1` on both Prefill and Decode (no pipeline parallel), so no layer partition is needed.
- `--max-num-batched-tokens` (A3 Prefill: `8192`): For better prefill performance, use `8192` when the input length is at least 8192 tokens; for shorter inputs, set it to the input token length. This recommendation also applies to A3 Prefill with KV Cache Pool in Section 5.3.2.
- `--enforce-eager`: Prefill nodes retain eager execution without graph capture.
- `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`: Decode nodes and single-node mixed-serving examples on A3 and 950DT use full graph capture for decode-only batches.
- `--speculative-config '{"method":"eagle3", ...}'`: Enables the `MiniMax-M3-EAGLE3-GQA` draft model. Do not replace this with GLM MTP options.

**Decode node-specific configurations:**

- `--language-model-only` (optional): For text-only tests, add this flag as needed based on the available KV cache capacity to avoid reserving memory for multimodal processing.
- `"max_cudagraph_capture_size"` (optional, omitted by default): Limits the maximum decode batch size covered by ACL graph capture and the graph memory reserved for it; the program default is `512`. With EAGLE3 speculative decoding, each request is expanded to `1 + num_speculative_tokens` tokens in one decode step (`4` tokens when `num_speculative_tokens=3`). Since DP distributes requests per rank, the per-rank batch size matters: for example, with `DP2` and `--max-num-seqs 256`, each DP rank handles 128 requests, producing `4 × 128 = 512` tokens per step — exactly at the default limit. If concurrency rises to 257, one DP rank handles 129 requests, giving `4 × 129 = 516 > 512`; batches above 512 skip graph capture and fall back to eager execution, lowering decode throughput. In that case set `"max_cudagraph_capture_size":1024` in `--compilation-config` (e.g., `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":1024}'`). Because a larger capture size reserves additional NPU memory, the configurations in this tutorial keep the default; add this option only when your target concurrency requires it.
- `--max-num-seqs 256`: Decode concurrency used by the verified 950DT products 1P1D launch. A3 uses `128`.

**Mooncake KV transfer configuration (`--kv-transfer-config`):**

- `"kv_connector": "MooncakeConnectorV1"`: Uses Mooncake as the KV cache transfer connector between prefill and decode nodes.
- `"kv_role": "kv_producer"` / `"kv_consumer"`: `kv_producer` on prefill nodes, `kv_consumer` on decode nodes.
- `"kv_port"`: Port for Mooncake KV transfer. Use different ports for prefill and decode. The verified values are A3 `36000`/`36100` and 950DT products `30000`/`26900`.
- `"use_ascend_direct": true`: Enables Ascend direct transfer for KV cache.
- `"ascend_local_comm_res_path": "/etc/hixlep"` (950DT products only): Used for RoCE-attached 950DT with `/etc/hixlep/` mounted. On SuperPod UB fabric, omit this option and export `ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'` instead, as in the Section 5.3.1 templates. The Section 5.3.2 950DT KV-pool templates retain the static-path setup; adapt the Mooncake connector inside the `MultiConnector` the same way when using UB fabric.
- `"prefill"` / `"decode"` sections: `dp_size`, `tp_size`, and `pp_size` must match the actual global layout on every node. A3 uses `prefill: dp1 tp4 pp4` on one node and `decode: dp8 tp4 pp1` across two nodes (four local DP ranks per Decode node). 950DT products uses `prefill: dp2 tp4 pp1` and `decode: dp2 tp4 pp1` for two nodes, or `decode: dp4 tp4 pp1` for three nodes.

**Request forwarding (proxy):**

- Wait until Prefill node and DP ranks prints `Application startup complete` before starting the proxy.
- The proxy maps every prefill engine endpoint and every decode engine endpoint to a single entry point on port `8009`.
- If requests reach the proxy but no output is returned, check that the proxy host list includes every healthy Prefill node and DP ranks, and that all nodes still have free NPU memory after the previous run.

Please refer to [envs.py](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/envs.py) for further explanation and restrictions of the environment variables above.

#### 5.3.2 Prefill-Decode Disaggregation with KV Cache Pool

This section builds on [Section 5.3.1](#531-prefill-decode-disaggregation). The A3 example uses the same three-node 1P1D topology, `launch_online_dp.py`, proxy mapping, and MiniMax `vllm serve` settings as Section 5.3.1 (one P plus two D nodes). The 950DT pool example retains its two-node 1P1D topology. Prefill and Decode switch to `MultiConnector` so live P→D KV transfer and Mooncake KV Cache Pool work together:

- `MooncakeConnectorV1` transfers KV from Prefill to Decode in real time (same role as Section 5.3.1).
- `AscendStoreConnector` stores KV in the Mooncake KV Cache Pool so later Prefills with the same prefix can hit the pool instead of recomputing.
- By default the Prefill node performs pool lookup, load, and write.

The examples below enable three-tier KV storage: the active NPU KV cache, the Mooncake memory pool, and SSD offload.

For backend selection, `mooncake.json`, Mooncake Master, eviction, and tenant options, refer to the [KV Cache Pool Deployment Guide](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md). For the hardware/communication environment variables required by pooling, refer to [Environment Variables Description](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md). For MiniMax-specific issues, refer to [Chapter 10 FAQ](#10-faq).

**Compared with normal PD in Section 5.3.1, pay attention to these pooling-only requirements:**

| Item | Normal PD (5.3.1) | Pooled PD (this section) |
| ---- | --------------- | ------------------------ |
| Connector | Single `MooncakeConnectorV1` | `MultiConnector` wrapping `MooncakeConnectorV1` + `AscendStoreConnector` |
| Mooncake Master | Not required | Start `mooncake_master --enable_offload=true` before Decode / Prefill |
| `mooncake.json` | Not required | Required on every rank; A3 P and both D nodes use `15GB`; 950DT P and D use `64GB`; enable SSD offload on all nodes |
| `engine_id` / `lookup_rpc_port` | Fixed example IDs are OK | **Must be unique per DP rank** (`37000/37100 + DP_RANK`) to avoid port / engine collisions |
| Extra env | Section 5.3.1 `HCCL_*` only | Keep 5.3.1 env, then add pool fabric/UB exports from [KV Cache Pool Deployment Guide](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md) |
| Container mounts | 950DT needs `/etc/hixlep/` | Also mount `/etc/hccn.conf` and the local SSD offload directory; keep `/etc/hixlep/` on 950DT |
| Startup order | Decode → Prefill → Proxy | **Mooncake Master → Decode → Prefill → Proxy** |
| Verification | P→D KV transfer only | Also check Prefill pool lookup/get/put hits after a repeated-prefix warmup |

##### 5.3.2.1 Prerequisites

Mount the host HCCN config into every container that participates in pooling:

```bash
-v /etc/hccn.conf:/etc/hccn.conf:ro
```

On 950DT products, also keep the `/etc/hixlep/` mount from Section 5.3.1 for Ascend direct KV transfer.

Use a local SSD directory on each Prefill and Decode host. The examples use `/nvme/mooncake_offload`; replace it with a directory on the actual SSD mount. Create it before starting the container:

```bash
mkdir -p /nvme/mooncake_offload
```

Add the corresponding bind mount to every model container:

```bash
-v /nvme/mooncake_offload:/nvme/mooncake_offload
```

Set `ssd_offload_path` to the absolute directory visible inside the container and make it writable by the vLLM process. The same path refers to separate local SSD storage on each host. The backend creates per-rank subdirectories beneath it. The installed Mooncake package must support the `enable_ssd_offload` and `ssd_offload_path` arguments in `MooncakeDistributedStore.setup()`; use the dependencies in the [KV Cache Pool Deployment Guide](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md).

Place `mooncake.json` next to `run_dp_template.sh` on every node. Use the same reachable Mooncake Master address and `tenant_id` across P and D. `global_segment_size` is allocated per worker rank, not per node; a non-zero value must be aligned to `1GB`. If `MOONCAKE_GLOBAL_SEGMENT_SIZE` is already exported, update it to the intended size or unset it before startup because it overrides the JSON value.

A3 `mooncake.json` on P, first D and second D (replace `<MOONCAKE_MASTER_IP>` with the actual Master IP):

```json
{
  "metadata_server": "P2PHANDSHAKE",
  "protocol": "ascend",
  "device_name": "",
  "master_server_address": "<MOONCAKE_MASTER_IP>:50088",
  "global_segment_size": "15GB",
  "preferred_segment": false,
  "prefer_alloc_in_same_node": true,
  "enable_ssd_offload": true,
  "ssd_offload_path": "/nvme/mooncake_offload",
  "tenant_id": "default"
}
```

The A3 example uses `15GB` per worker rank on P and both D nodes. Adjust this value to each machine's available allocatable memory and fragmentation after reserving memory for vLLM and other processes. Allow additional space for the SSD I/O buffer and allocation headroom, and account for all contributing worker ranks on the node.

950DT Prefill `mooncake.json` (replace `<MOONCAKE_MASTER_IP>`):

```json
{
  "metadata_server": "P2PHANDSHAKE",
  "protocol": "ascend",
  "device_name": "",
  "master_server_address": "<MOONCAKE_MASTER_IP>:50088",
  "global_segment_size": "64GB",
  "preferred_segment": false,
  "prefer_alloc_in_same_node": true,
  "enable_ssd_offload": true,
  "ssd_offload_path": "/nvme/mooncake_offload",
  "tenant_id": "default"
}
```

950DT Decode `mooncake.json` (same Master and `tenant_id`; both P and D contribute memory and enable SSD offload):

```json
{
  "metadata_server": "P2PHANDSHAKE",
  "protocol": "ascend",
  "device_name": "",
  "master_server_address": "<MOONCAKE_MASTER_IP>:50088",
  "global_segment_size": "64GB",
  "preferred_segment": true,
  "prefer_alloc_in_same_node": true,
  "enable_ssd_offload": true,
  "ssd_offload_path": "/nvme/mooncake_offload",
  "tenant_id": "default"
}
```

The 950DT examples use `64GB` per worker rank on both P and D. Adjust the size to the actual available memory and fragmentation on each machine rather than treating `64GB` as a fixed requirement. Budget for all worker ranks on the node, SSD I/O buffers, and other allocations; reduce the segment size if the machine cannot register the requested memory.

##### 5.3.2.2 Environment Variables

Start from the Section 5.3.1 environment block (`HCCL_IF_IP`, `HCCL`/`GLOO` socket IFNAME, `HCCL_BUFFSIZE`, `HCCL_OP_EXPANSION_MODE`, `PYTORCH_NPU_ALLOC_CONF`, `ASCEND_RT_VISIBLE_DEVICES`, and A3 Prefill `VLLM_PP_LAYER_PARTITION="16,15,15,14"`). Then add the pooling-required variables below. Choose the hardware/communication exports from [Environment Variables Description](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md) according to your machine and link type.

Common pooling exports on Prefill node and DP ranks:

```bash
# Optional: only if Mooncake is installed in a non-standard path (same as Section 5.3.1).
if [ -n "${MOONCAKE_LIB_DIRS:-}" ]; then
    export LD_LIBRARY_PATH="${MOONCAKE_LIB_DIRS}:${LD_LIBRARY_PATH:-}"
fi
export MOONCAKE_CONFIG_PATH=./mooncake.json
export PYTHONHASHSEED=0
# SSD I/O buffer per worker: 1GB, aligned for A3 fabric memory.
export MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=1073741824
```

The SSD I/O buffer is allocated separately from `global_segment_size`. The `1GB` value above is an aligned starting point; size it for the KV read/write batches and include it in the per-rank memory budget. Set per-rank SSD capacity limits according to the actual disk capacity and number of contributing workers, as described in [Mooncake SSD Offload Parameter Tuning](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md#mooncake-ssd-offload-parameter-tuning).

| Hardware | Dependencies | Required pooling exports | Notes |
| -------- | ------------- | ------------------------ | ----- |
| 800I/T A3 (HCCS, recommended) | HDK >= 26.0, or HDK >= 25.5 with mooncake >= v0.3.11; CANN >= 9.0.0 | `export ASCEND_ENABLE_USE_FABRIC_MEM=1` | Keep the Section 5.3.1 `HCCL_IF_IP` / socket IFNAME exports. |
| 800I/T A3 (RoCE) or 800I/T A2 | A2: HDK >= 25.5 recommended | `export HCCL_INTRA_ROCE_ENABLE=1`, plus `HCCL_IF_IP` / socket IFNAME, and `nr_hugepages=200000` | Use the RoCE path from the KV Cache Pool guide. |
| 950PR/DT (Device UB) | HDK >= 25.6 with mooncake >= v0.3.11; CANN >= 9.1.0 | `export ASCEND_LOCAL_COMM_RES_PATH=/etc/hixlep/` and `export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'`; `unset ASCEND_GLOBAL_RESOURCE_CONFIG` | Mount `/etc/hixlep/`. For UBOE instead, use `ASCEND_GLOBAL_RESOURCE_CONFIG` as described in the KV Cache Pool guide. |

##### 5.3.2.3 Prefill / Decode Scripts

Reuse Section 5.3.1 `launch_online_dp.py`. Replace each role's `run_dp_template.sh` with the pooled version below. Keep the Section 5.3.1 `vllm serve` flag style; only `--kv-transfer-config` switches to `MultiConnector`. Expand `engine_id` and `lookup_rpc_port` from `DP_RANK` — do not hardcode the same values across ranks.

=== "A3 series (W8A8)"

    Prefill-Decode disaggregation with KV Cache Pool on 3 Atlas 800I A3 (64GB × 16) for `MiniMax-M3-w8a8` (W8A8) with `MiniMax-M3-EAGLE3-GQA`.

    One Prefill node uses `DP1 × TP4 × PP4`; two Decode nodes form a global `DP8 × TP4 × PP1` group, with four local ranks per node. Copy the same Decode template to both D nodes and use the node-specific launch blocks below. Service settings match Section 5.3.1; `MultiConnector` adds the KV pool. The `--language-model-only` option remains in the existing optional guidance rather than the launch commands.

    1. Prefill node

    ```bash
    unset http_proxy https_proxy ftp_proxy

    # Replace IP, NIC and weight-path placeholders for this node.
    export LOCAL_IP="${LOCAL_IP:-<CURRENT_NODE_IP>}"
    export NIC_NAME="${NIC_NAME:-<CURRENT_NODE_NIC>}"
    export MODEL_PATH="${MODEL_PATH:-<MINIMAX_M3_W8A8_PATH>}"
    export EAGLE3_WEIGHT_PATH="${EAGLE3_WEIGHT_PATH:-<MINIMAX_M3_EAGLE3_GQA_PATH>}"

    # Role defaults; the launcher overrides these for each engine.
    export DP_SIZE=${DP_SIZE:-1}
    export TP_SIZE=${TP_SIZE:-4}
    export PP_SIZE=${PP_SIZE:-4}
    export DP_RANK=${DP_RANK:-0}
    export DP_ADDRESS="${DP_ADDRESS:-$LOCAL_IP}"
    export DP_RPC_PORT=${DP_RPC_PORT:-6884}
    export SERVICE_PORT=${SERVICE_PORT:-31050}
    export VISIBLE_DEVICES="${VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"

    # Pooling extras: all three A3 nodes use the 15GB mooncake.json in Section 5.3.2.1.
    export VLLM_PP_LAYER_PARTITION="16,15,15,14"
    export VLLM_SERVER_DEV_MODE=1
    export VLLM_USE_V2_MODEL_RUNNER=1
    export HCCL_BUFFSIZE=1024
    export HCCL_IF_IP=$LOCAL_IP
    export HCCL_OP_EXPANSION_MODE="AIV"
    export HCCL_SOCKET_IFNAME=$NIC_NAME
    export ASCEND_ENABLE_USE_FABRIC_MEM=1
    export ASCEND_RT_VISIBLE_DEVICES=$VISIBLE_DEVICES
    # If Mooncake is installed in a non-standard path, set this before startup.
    if [ -n "${MOONCAKE_LIB_DIRS:-}" ]; then
        export LD_LIBRARY_PATH="${MOONCAKE_LIB_DIRS}:${LD_LIBRARY_PATH:-}"
    fi
    export GLOO_SOCKET_IFNAME=$NIC_NAME
    export MOONCAKE_CONFIG_PATH=./mooncake.json
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export PYTHONHASHSEED=0
    export MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=1073741824

    # For better prefill performance, keep max_num_batched_tokens at 8192 for input
    # lengths >= 8192 tokens; for shorter inputs, set it to the input token length.
    vllm serve "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port "$SERVICE_PORT" \
        --data-parallel-size "$DP_SIZE" \
        --data-parallel-rank "$DP_RANK" \
        --data-parallel-address "$DP_ADDRESS" \
        --data-parallel-rpc-port "$DP_RPC_PORT" \
        --tensor-parallel-size "$TP_SIZE" \
        --pipeline-parallel-size "$PP_SIZE" \
        --enforce-eager \
        --distributed-executor-backend mp \
        --served-model-name minimax-m3 \
        --enable-expert-parallel \
        --seed 1024 \
        --max-model-len 133000 \
        --max-num-seqs 32 \
        --max-num-batched-tokens 8192 \
        --enable-chunked-prefill \
        --enable-prefix-caching \
        --trust-remote-code \
        --quantization ascend \
        --gpu-memory-utilization 0.92 \
        --limit-mm-per-prompt '{"image":1,"video":0}' \
        --reasoning-parser minimax_m3 \
        --additional-config '{"enable_cpu_binding":true,"ascend_compilation_config":{"fuse_norm_quant":false},"multistream_overlap_shared_expert":true,"weight_nz_mode":2,"enable_shared_expert_dp":true,"enable_flashcomm1":true}' \
        --speculative-config '{"model":"'"${EAGLE3_WEIGHT_PATH}"'", "method":"eagle3", "num_speculative_tokens":3}' \
        --kv-transfer-config \
        '{
            "kv_connector":"MultiConnector",
            "kv_role":"kv_producer",
            "engine_id":"minimax-m3-prefill-dp'"$DP_RANK"'",
            "kv_connector_extra_config":{
                "connectors":[
                    {
                        "kv_connector":"MooncakeConnectorV1",
                        "kv_buffer_device":"npu",
                        "kv_role":"kv_producer",
                        "kv_port":"36000",
                        "kv_connector_extra_config":{
                            "use_ascend_direct":true,
                            "prefill":{"dp_size":1,"tp_size":4,"pp_size":4,"pp_layer_partition":"16,15,15,14"},
                            "decode":{"dp_size":8,"tp_size":4,"pp_size":1}
                        }
                    },
                    {
                        "kv_connector":"AscendStoreConnector",
                        "kv_role":"kv_producer",
                        "kv_connector_extra_config":{
                            "backend":"mooncake",
                            "lookup_rpc_port":'$((37000 + $DP_RANK))'
                        }
                    }
                ]
            }
        }'
    ```

    2. Decode node

    ```bash
    unset http_proxy https_proxy ftp_proxy all_proxy HTTP_PROXY HTTPS_PROXY FTP_PROXY ALL_PROXY

    # Replace IP, NIC and weight-path placeholders for this node.
    export LOCAL_IP="${LOCAL_IP:-<CURRENT_NODE_IP>}"
    export NIC_NAME="${NIC_NAME:-<CURRENT_NODE_NIC>}"
    export MODEL_PATH="${MODEL_PATH:-<MINIMAX_M3_W8A8_PATH>}"
    export EAGLE3_WEIGHT_PATH="${EAGLE3_WEIGHT_PATH:-<MINIMAX_M3_EAGLE3_GQA_PATH>}"

    # Role defaults; the launcher overrides these for each engine.
    export DP_SIZE=${DP_SIZE:-8}
    export TP_SIZE=${TP_SIZE:-4}
    export PP_SIZE=${PP_SIZE:-1}
    export DP_RANK=${DP_RANK:-0}
    export DP_ADDRESS="${DP_ADDRESS:-<DECODE_COORDINATOR_IP>}"
    export DP_RPC_PORT=${DP_RPC_PORT:-31422}
    export SERVICE_PORT=${SERVICE_PORT:-31260}
    export VISIBLE_DEVICES="${VISIBLE_DEVICES:-0,1,2,3}"

    # Pooling extras: all three A3 nodes use the 15GB mooncake.json in Section 5.3.2.1.
    export VLLM_SERVER_DEV_MODE=1
    export VLLM_USE_V2_MODEL_RUNNER=1
    export HCCL_BUFFSIZE=2048
    export HCCL_IF_IP=$LOCAL_IP
    export HCCL_OP_EXPANSION_MODE="AIV"
    export HCCL_SOCKET_IFNAME=$NIC_NAME
    export ASCEND_ENABLE_USE_FABRIC_MEM=1
    export ASCEND_RT_VISIBLE_DEVICES=$VISIBLE_DEVICES
    # If Mooncake is installed in a non-standard path, set this before startup.
    if [ -n "${MOONCAKE_LIB_DIRS:-}" ]; then
        export LD_LIBRARY_PATH="${MOONCAKE_LIB_DIRS}:${LD_LIBRARY_PATH:-}"
    fi
    export GLOO_SOCKET_IFNAME=$NIC_NAME
    export MOONCAKE_CONFIG_PATH=./mooncake.json
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export PYTHONHASHSEED=0
    export MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=1073741824

    vllm serve "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port "$SERVICE_PORT" \
        --data-parallel-size "$DP_SIZE" \
        --data-parallel-rank "$DP_RANK" \
        --data-parallel-address "$DP_ADDRESS" \
        --data-parallel-rpc-port "$DP_RPC_PORT" \
        --tensor-parallel-size "$TP_SIZE" \
        --pipeline-parallel-size "$PP_SIZE" \
        --enable-expert-parallel \
        --enable-chunked-prefill \
        --enable-prefix-caching \
        --seed 1024 \
        --served-model-name minimax-m3 \
        --reasoning-parser minimax_m3 \
        --distributed-executor-backend mp \
        --max-model-len 133000 \
        --max-num-batched-tokens 32768 \
        --trust-remote-code \
        --quantization ascend \
        --max-num-seqs 128 \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
        --gpu-memory-utilization 0.92 \
        --limit-mm-per-prompt '{"image":1,"video":0}' \
        --additional-config '{"enable_cpu_binding":true,"ascend_compilation_config":{"fuse_norm_quant":false},"multistream_overlap_shared_expert":true,"weight_nz_mode":2,"enable_shared_expert_dp":true,"enable_fused_mc2":0,"enable_flashcomm1":true}' \
        --speculative-config '{"model":"'"${EAGLE3_WEIGHT_PATH}"'", "method":"eagle3", "num_speculative_tokens":3}' \
        --kv-transfer-config \
        '{
            "kv_connector":"MultiConnector",
            "kv_role":"kv_consumer",
            "engine_id":"minimax-m3-decode-dp'"$DP_RANK"'",
            "kv_connector_extra_config":{
                "connectors":[
                    {
                        "kv_connector":"MooncakeConnectorV1",
                        "kv_buffer_device":"npu",
                        "kv_role":"kv_consumer",
                        "kv_port":"36100",
                        "kv_connector_extra_config":{
                            "use_ascend_direct":true,
                            "prefill":{"dp_size":1,"tp_size":4,"pp_size":4,"pp_layer_partition":"16,15,15,14"},
                            "decode":{"dp_size":8,"tp_size":4,"pp_size":1}
                        }
                    },
                    {
                        "kv_connector":"AscendStoreConnector",
                        "kv_role":"kv_consumer",
                        "kv_connector_extra_config":{
                            "backend":"mooncake",
                            "lookup_rpc_port":'$((37100 + $DP_RANK))'
                        }
                    }
                ]
            }
        }'
    ```

    Start order on A3: Mooncake Master → both Decode nodes → Prefill → Proxy. Use the three-node commands in Section 5.3.2.4, with the same DP/TP/PP sizes and API ports as Section 5.3.1 A3. The single P API and all eight D APIs must be healthy before starting the proxy.

=== "950DT products"

    Prefill-Decode disaggregation with KV Cache Pool on 2 950DT products (96GB × 8) for `MiniMax-M3-MXFP8` with `MiniMax-M3-EAGLE3-GQA`. Mount `/etc/hixlep/`, `/etc/hccn.conf`, and the local SSD directory from Section 5.3.2.1. Place the Prefill / Decode `mooncake.json` from Section 5.3.2.1 next to `run_dp_template.sh` (`global_segment_size` is `64GB` on both roles, with SSD offload enabled). Set `master_server_address` to the same reachable Mooncake Master on both nodes.

    1. Prefill node

    ```bash
    unset ftp_proxy https_proxy http_proxy all_proxy
    unset FTP_PROXY HTTPS_PROXY HTTP_PROXY ALL_PROXY

    # Replace IP, NIC and weight-path placeholders for this node.
    export LOCAL_IP="${LOCAL_IP:-<CURRENT_NODE_IP>}"
    export NIC_NAME="${NIC_NAME:-<CURRENT_NODE_NIC>}"
    export MODEL_PATH="${MODEL_PATH:-<MINIMAX_M3_MXFP8_PATH>}"
    export EAGLE3_WEIGHT_PATH="${EAGLE3_WEIGHT_PATH:-<MINIMAX_M3_EAGLE3_GQA_PATH>}"

    # Role defaults; the launcher overrides these for each engine.
    export DP_SIZE=${DP_SIZE:-2}
    export TP_SIZE=${TP_SIZE:-4}
    export PP_SIZE=${PP_SIZE:-1}
    export DP_RANK=${DP_RANK:-0}
    export DP_ADDRESS="${DP_ADDRESS:-$LOCAL_IP}"
    export DP_RPC_PORT=${DP_RPC_PORT:-6884}
    export SERVICE_PORT=${SERVICE_PORT:-31050}
    export VISIBLE_DEVICES="${VISIBLE_DEVICES:-0,1,2,3}"

    # Runtime environment and pooling extras (Device UB).
    export VLLM_USE_V2_MODEL_RUNNER=1
    export HCCL_BUFFSIZE=256
    export HCCL_IF_IP=$LOCAL_IP
    export HCCL_OP_EXPANSION_MODE="AIV"
    export HCCL_SOCKET_IFNAME=$NIC_NAME
    export ASCEND_RT_VISIBLE_DEVICES=$VISIBLE_DEVICES
    export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH
    export GLOO_SOCKET_IFNAME=$NIC_NAME
    export MOONCAKE_CONFIG_PATH=./mooncake.json
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export PYTHONHASHSEED=0
    export MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=1073741824
    export ASCEND_LOCAL_COMM_RES_PATH=/etc/hixlep/
    export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
    unset ASCEND_GLOBAL_RESOURCE_CONFIG

    vllm serve "$MODEL_PATH" \
      --host 0.0.0.0 \
      --port "$SERVICE_PORT" \
      --data-parallel-size "$DP_SIZE" \
      --data-parallel-rank "$DP_RANK" \
      --data-parallel-address "$DP_ADDRESS" \
      --data-parallel-rpc-port "$DP_RPC_PORT" \
      --tensor-parallel-size "$TP_SIZE" \
      --pipeline-parallel-size "$PP_SIZE" \
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
      --speculative-config '{"model":"'"${EAGLE3_WEIGHT_PATH}"'", "method":"eagle3", "num_speculative_tokens":3, "kv_cache_dtype":"bfloat16"}' \
      --enforce-eager \
      --enable-chunked-prefill \
      --enable-prefix-caching \
      --additional-config '{"enable_cpu_binding":true,"ascend_compilation_config":{"fuse_qknorm_rope":true,"fuse_norm_quant":true},"multistream_overlap_shared_expert":true,"enable_shared_expert_dp":true,"enable_flashcomm1":true}' \
      --kv-transfer-config \
      '{
          "kv_connector":"MultiConnector",
          "kv_role":"kv_producer",
          "engine_id":"minimax-m3-prefill-dp'"$DP_RANK"'",
          "kv_connector_extra_config":{
              "connectors":[
                  {
                      "kv_connector":"MooncakeConnectorV1",
                      "kv_buffer_device":"npu",
                      "kv_role":"kv_producer",
                      "kv_port":"30000",
                      "kv_connector_extra_config":{
                          "use_ascend_direct":true,
                          "ascend_local_comm_res_path":"/etc/hixlep",
                          "prefill":{"dp_size":2,"tp_size":4,"pp_size":1},
                          "decode":{"dp_size":2,"tp_size":4,"pp_size":1}
                      }
                  },
                  {
                      "kv_connector":"AscendStoreConnector",
                      "kv_role":"kv_producer",
                      "kv_connector_extra_config":{
                          "backend":"mooncake",
                          "lookup_rpc_port":'$((37000 + $DP_RANK))'
                      }
                  }
              ]
          }
      }'
    ```

    2. Decode node

    ```bash
    unset ftp_proxy https_proxy http_proxy all_proxy
    unset FTP_PROXY HTTPS_PROXY HTTP_PROXY ALL_PROXY

    # Replace IP, NIC and weight-path placeholders for this node.
    export LOCAL_IP="${LOCAL_IP:-<CURRENT_NODE_IP>}"
    export NIC_NAME="${NIC_NAME:-<CURRENT_NODE_NIC>}"
    export MODEL_PATH="${MODEL_PATH:-<MINIMAX_M3_MXFP8_PATH>}"
    export EAGLE3_WEIGHT_PATH="${EAGLE3_WEIGHT_PATH:-<MINIMAX_M3_EAGLE3_GQA_PATH>}"

    # Role defaults; the launcher overrides these for each engine.
    export DP_SIZE=${DP_SIZE:-2}
    export TP_SIZE=${TP_SIZE:-4}
    export PP_SIZE=${PP_SIZE:-1}
    export DP_RANK=${DP_RANK:-0}
    export DP_ADDRESS="${DP_ADDRESS:-<DECODE_COORDINATOR_IP>}"
    export DP_RPC_PORT=${DP_RPC_PORT:-5964}
    export SERVICE_PORT=${SERVICE_PORT:-31060}
    export VISIBLE_DEVICES="${VISIBLE_DEVICES:-0,1,2,3}"

    # Runtime environment and pooling extras (Device UB).
    export VLLM_USE_V2_MODEL_RUNNER=1
    export HCCL_BUFFSIZE=2048
    export HCCL_IF_IP=$LOCAL_IP
    export HCCL_OP_EXPANSION_MODE="AIV"
    export HCCL_SOCKET_IFNAME=$NIC_NAME
    export ASCEND_RT_VISIBLE_DEVICES=$VISIBLE_DEVICES
    export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH
    export GLOO_SOCKET_IFNAME=$NIC_NAME
    export MOONCAKE_CONFIG_PATH=./mooncake.json
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export PYTHONHASHSEED=0
    export MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=1073741824
    export ASCEND_LOCAL_COMM_RES_PATH=/etc/hixlep/
    export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
    unset ASCEND_GLOBAL_RESOURCE_CONFIG

    vllm serve "$MODEL_PATH" \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
        --host 0.0.0.0 \
        --port "$SERVICE_PORT" \
        --data-parallel-size "$DP_SIZE" \
        --data-parallel-rank "$DP_RANK" \
        --data-parallel-address "$DP_ADDRESS" \
        --data-parallel-rpc-port "$DP_RPC_PORT" \
        --tensor-parallel-size "$TP_SIZE" \
        --pipeline-parallel-size "$PP_SIZE" \
        --enable-expert-parallel \
        --seed 1024 \
        --served-model-name minimax-m3 \
        --reasoning-parser minimax_m3 \
        --safetensors-load-strategy prefetch \
        --distributed-executor-backend mp \
        --max-model-len 133000 \
        --max-num-batched-tokens 32768 \
        --trust-remote-code \
        --max-num-seqs 256 \
        --gpu-memory-utilization 0.92 \
        --dtype bfloat16 \
        --quantization mxfp8 \
        --kv-cache-dtype fp8 \
        --speculative-config '{"model":"'"${EAGLE3_WEIGHT_PATH}"'", "method":"eagle3", "num_speculative_tokens":3, "kv_cache_dtype":"bfloat16"}' \
        --additional-config '{"enable_cpu_binding":true,"ascend_compilation_config":{"fuse_norm_quant":false},"multistream_overlap_shared_expert":true,"enable_shared_expert_dp":true}' \
        --kv-transfer-config \
        '{
            "kv_connector":"MultiConnector",
            "kv_role":"kv_consumer",
            "engine_id":"minimax-m3-decode-dp'"$DP_RANK"'",
            "kv_connector_extra_config":{
                "connectors":[
                    {
                        "kv_connector":"MooncakeConnectorV1",
                        "kv_buffer_device":"npu",
                        "kv_role":"kv_consumer",
                        "kv_port":"26900",
                        "kv_connector_extra_config":{
                            "use_ascend_direct":true,
                            "ascend_local_comm_res_path":"/etc/hixlep",
                            "prefill":{"dp_size":2,"tp_size":4,"pp_size":1},
                            "decode":{"dp_size":2,"tp_size":4,"pp_size":1}
                        }
                    },
                    {
                        "kv_connector":"AscendStoreConnector",
                        "kv_role":"kv_consumer",
                        "kv_connector_extra_config":{
                            "backend":"mooncake",
                            "lookup_rpc_port":'$((37100 + $DP_RANK))'
                        }
                    }
                ]
            }
        }'
    ```

    Start order on 950DT: Mooncake Master → Decode → Prefill → Proxy. Use the Section 5.3.1 950DT two-node commands (`DP2` on both roles, proxy on `8009`), with the pooled templates above; do not use the three-node scaling commands for this two-node example.

##### 5.3.2.4 Start the Services

Start in this order:

```text
Mooncake Master
    ↓
First Decode node → Second Decode node (A3 only)
    ↓
Prefill
    ↓
Proxy (:8009)
```

1. Start `mooncake_master` on the Prefill node or another reachable host, and use its actual IP in every `mooncake.json`. Confirm port `50088` is reachable from all model nodes:

    ```bash
    mooncake_master \
    --port 50088 \
    --eviction_high_watermark_ratio 0.9 \
    --eviction_ratio 0.1 \
    --default_kv_lease_ttl 11000 \
    --enable_offload=true \
    --client_ttl=120
    ```

2. Start the first A3 Decode node (global DP ranks 0–3). Use the pooled Decode template, not the normal-PD template:

    ```bash
    export DECODE0_IP="<FIRST_DECODE_NODE_IP>"
    export LOCAL_IP="$DECODE0_IP"
    export DP_SIZE=8
    export TP_SIZE=4
    export PP_SIZE=1
    export LOCAL_DP_SIZE=4
    export DP_RANK_START=0
    export DP_ADDRESS="$DECODE0_IP"
    export DP_RPC_PORT=31422
    export SERVICE_PORT=31260

    python launch_online_dp.py \
        --dp-size "$DP_SIZE" --tp-size "$TP_SIZE" --pp-size "$PP_SIZE" \
        --dp-size-local "$LOCAL_DP_SIZE" --dp-rank-start "$DP_RANK_START" \
        --dp-address "$DP_ADDRESS" --dp-rpc-port "$DP_RPC_PORT" \
        --vllm-start-port "$SERVICE_PORT"
    ```

    Wait for rank 0 to print `Started DP Coordinator process`. Keep this shell running while starting the second node from another shell; initialization needs both D nodes. On 950DT, use the Section 5.3.1 two-node Decode launch command with the 950DT pooled template, then skip step 3.

3. Start the second A3 Decode node (global DP ranks 4–7). `LOCAL_IP` is the second node's IP, but `DP_ADDRESS` remains the first Decode node's IP:

    ```bash
    export DECODE0_IP="<FIRST_DECODE_NODE_IP>"
    export DECODE1_IP="<SECOND_DECODE_NODE_IP>"
    export LOCAL_IP="$DECODE1_IP"
    export DP_SIZE=8
    export TP_SIZE=4
    export PP_SIZE=1
    export LOCAL_DP_SIZE=4
    export DP_RANK_START=4
    export DP_ADDRESS="$DECODE0_IP"
    export DP_RPC_PORT=31422
    export SERVICE_PORT=31260

    python launch_online_dp.py \
        --dp-size "$DP_SIZE" --tp-size "$TP_SIZE" --pp-size "$PP_SIZE" \
        --dp-size-local "$LOCAL_DP_SIZE" --dp-rank-start "$DP_RANK_START" \
        --dp-address "$DP_ADDRESS" --dp-rpc-port "$DP_RPC_PORT" \
        --vllm-start-port "$SERVICE_PORT"
    ```

    Both Decode nodes expose API ports `31260`–`31263`. Wait for all eight DP ranks to print `Application startup complete`.

4. Start Prefill. On A3, use the pooled P template:

    ```bash
    export PREFILL_IP="<PREFILL_NODE_IP>"
    export LOCAL_IP="$PREFILL_IP"
    export DP_SIZE=1
    export TP_SIZE=4
    export PP_SIZE=4
    export LOCAL_DP_SIZE=1
    export DP_RANK_START=0
    export DP_ADDRESS="$PREFILL_IP"
    export DP_RPC_PORT=6884
    export SERVICE_PORT=31050

    python launch_online_dp.py \
        --dp-size "$DP_SIZE" --tp-size "$TP_SIZE" --pp-size "$PP_SIZE" \
        --dp-size-local "$LOCAL_DP_SIZE" --dp-rank-start "$DP_RANK_START" \
        --dp-address "$DP_ADDRESS" --dp-rpc-port "$DP_RPC_PORT" \
        --vllm-start-port "$SERVICE_PORT"
    ```

    On 950DT, use the Section 5.3.1 two-node Prefill command with its pooled template. Wait for every Prefill rank to print `Application startup complete`.

5. Start the proxy. The A3 proxy registers one Prefill endpoint and eight Decode endpoints:

    ```bash
    export PREFILL_IP="<PREFILL_NODE_IP>"
    export DECODE0_IP="<FIRST_DECODE_NODE_IP>"
    export DECODE1_IP="<SECOND_DECODE_NODE_IP>"

    unset http_proxy https_proxy ftp_proxy all_proxy HTTP_PROXY HTTPS_PROXY FTP_PROXY ALL_PROXY

    python load_balance_proxy_server_example.py \
    --port 8009 \
    --host $PREFILL_IP \
    --prefiller-hosts \
        $PREFILL_IP \
    --prefiller-ports \
        31050 \
    --decoder-hosts \
        $DECODE0_IP $DECODE0_IP $DECODE0_IP $DECODE0_IP \
        $DECODE1_IP $DECODE1_IP $DECODE1_IP $DECODE1_IP \
    --decoder-ports \
        31260 31261 31262 31263 31260 31261 31262 31263
    ```

    On 950DT, use the Section 5.3.1 two-node proxy command (two P and two D endpoints). The service is accessible at `<proxy_ip>:8009`; use this endpoint in Section 7.

##### 5.3.2.5 Verification

1. Confirm Mooncake Master port `50088` is reachable, Master runs with `--enable_offload=true`, and every node uses the same Master address and `tenant_id`. For A3, confirm P and both D nodes load `global_segment_size="15GB"`; for 950DT, confirm P and D both load `"64GB"`. Check worker logs for successful segment registration.
2. Confirm every Decode engine port is ready, then every Prefill engine port. A3 has nine APIs (one P plus eight D); 950DT has four (two P plus two D).
3. Send requests only to the proxy on port `8009`.
4. Warm up with a repeated-prefix request, then send again and check Prefill logs for KV Pool lookup/get/put and hit information.
5. Confirm Decode logs still show a normal Mooncake P→D KV transfer.
6. Confirm SSD storage initializes under `ssd_offload_path` for each contributing worker. Use a working set larger than the memory pool to verify SSD offload and subsequent reads; a memory-pool hit alone does not exercise the SSD tier.

### 5.4 1M-Token Context Deployment

MiniMax-M3 supports a 1M-token context on a single Atlas 800I A3 node with W8A8 weights or a single 950DT node with MXFP8 weights. The following examples use the previously verified long-context deployment layouts with `FULL_DECODE_ONLY` as the graph mode. Here, 1M means 1,048,576 tokens. `--max-model-len` limits the total input and generated tokens, so reserve output space when preparing a request.

| Platform / weights | NPUs per node | TP / PP / DP | `max-model-len` | `max-num-batched-tokens` | KV cache dtype |
| --- | --- | --- | --- | --- | --- |
| Atlas 800I A3 / W8A8 | 16 × 64GB | 4 / 4 / 1 | 1049600 | 8192 | `auto` (default) |
| 950DT / MXFP8 | 8 × 96GB | 8 / 1 / 1 | 1048576 | 2048 | `fp8`, excluding layers 0–2 |
| 950DT / MXFP8 | 8 × 96GB | 4 / 2 / 1 | 1048576 | 8192 | `fp8`, excluding layers 0–2 |

All three configurations use Model Runner V2, `max-num-seqs=1`, `FULL_DECODE_ONLY` graph mode, and chunked prefill. Prefix caching, asynchronous scheduling, and static kernels are disabled; no speculative model is loaded. The smaller batched-token budget controls the size of each prefill chunk rather than the total context length.

Prepare the container and weights as described in Sections 3 and 4. Replace `<MODEL_PATH>` (A3) or `<WEIGHT_PATH>` (950DT) with the local weight directory. Replace `<NIC_NAME>` in both `HCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME` with the same local communication interface. The commands use the CANN environment-script paths from the deployment configurations; adjust these paths to the installed CANN directory. The 950DT examples also require the matching `experimental_950_transformer` vendor package before sourcing its environment script.

=== "Atlas 800I A3 (W8A8)"

    Use one 16-NPU node with TP4/PP4/DP1. This long-context configuration splits the 60 transformer layers evenly as `15,15,15,15`. The `1049600` context limit leaves room for 1,048,576 input tokens and up to 1024 generated tokens. `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` permits the configured limit when it exceeds the checkpoint's declared maximum.

    ```bash
    MODEL_PATH="<MODEL_PATH>"  # MiniMax-M3 W8A8 weights

    source /usr/local/Ascend/cann-9.1.0/set_env.sh

    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200
    export VLLM_PP_LAYER_PARTITION="15,15,15,15"
    export VLLM_USE_V2_MODEL_RUNNER=1
    export HCCL_BUFFSIZE=256
    export HCCL_OP_EXPANSION_MODE="AIV"
    export HCCL_SOCKET_IFNAME="<NIC_NAME>"
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
    export GLOO_SOCKET_IFNAME="<NIC_NAME>"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    vllm serve "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port 8000 \
        --served-model-name minimax-m3 \
        --trust-remote-code \
        --language-model-only \
        --reasoning-parser minimax_m3 \
        --distributed-executor-backend mp \
        --tensor-parallel-size 4 \
        --pipeline-parallel-size 4 \
        --data-parallel-size 1 \
        --enable-expert-parallel \
        --quantization ascend \
        --block-size 128 \
        --max-model-len 1049600 \
        --max-num-batched-tokens 8192 \
        --long-prefill-token-threshold 8192 \
        --max-num-seqs 1 \
        --enable-chunked-prefill \
        --no-enable-prefix-caching \
        --no-async-scheduling \
        --gpu-memory-utilization 0.95 \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
        --additional-config '{
            "enable_cpu_binding": true,
            "ascend_compilation_config": {
                "enable_static_kernel": false,
                "fuse_norm_quant": false
            },
            "multistream_overlap_shared_expert": true,
            "enable_shared_expert_dp": false,
            "weight_nz_mode": 2,
            "enable_flashcomm1": true,
            "enable_reduce_sample": false
        }' > ./minimax-m3-1m-a3.log 2>&1 &
    ```

    `--long-prefill-token-threshold 8192` sets the long-prefill threshold used with chunked prefill. The scheduler uses this threshold to limit the prefill tokens scheduled for a long request in one iteration, while `--max-num-batched-tokens 8192` bounds the total token budget for the iteration. In this single-request example both values are 8192, so the 1M prompt is processed in chunks within that budget. This parameter does not reduce the total context limit configured by `--max-model-len`. See the [vLLM scheduler arguments](https://docs.vllm.ai/en/latest/configuration/engine_args/#--long-prefill-token-threshold) for details.

=== "950DT Products (MXFP8)"

    Choose either of the following single-node layouts on eight NPUs. Both use FP8 KV cache while keeping layers 0, 1, and 2 in the default KV dtype. The total input and generated tokens must fit within 1,048,576 tokens; for example, reserve 1024 output tokens by limiting the input to at most 1,047,552 tokens.

    === "TP8 / PP1 / DP1"

        ```bash
        export WEIGHT_PATH="<WEIGHT_PATH>"  # MiniMax-M3 MXFP8 weights

        export VLLM_USE_V2_MODEL_RUNNER=1
        export HCCL_BUFFSIZE=256
        export HCCL_OP_EXPANSION_MODE="AIV"
        export HCCL_SOCKET_IFNAME="<NIC_NAME>"
        export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        export GLOO_SOCKET_IFNAME="<NIC_NAME>"
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

        source /usr/local/Ascend/cann-9.1.0/opp/vendors/experimental_950_transformer/bin/set_env.bash

        vllm serve "$WEIGHT_PATH" \
            --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
            --host 0.0.0.0 \
            --port 31050 \
            --served-model-name minimax-m3 \
            --trust-remote-code \
            --distributed-executor-backend mp \
            --tensor-parallel-size 8 \
            --pipeline-parallel-size 1 \
            --data-parallel-size 1 \
            --dtype bfloat16 \
            --quantization mxfp8 \
            --max-model-len 1048576 \
            --max-num-seqs 1 \
            --max-num-batched-tokens 2048 \
            --gpu-memory-utilization 0.92 \
            --kv-cache-dtype fp8 \
            --kv-cache-dtype-skip-layers 0 1 2 \
            --limit-mm-per-prompt '{"image":0,"video":0}' \
            --enable-chunked-prefill \
            --no-enable-prefix-caching \
            --no-async-scheduling \
            --reasoning-parser minimax_m3 \
            --safetensors-load-strategy prefetch \
            --additional-config '{
                "enable_cpu_binding": true,
                "ascend_compilation_config": {
                    "fuse_qknorm_rope": false,
                    "fuse_norm_quant": false,
                    "enable_static_kernel": false
                },
                "multistream_overlap_shared_expert": false,
                "enable_shared_expert_dp": false,
                "enable_flashcomm1": false
            }' > ./minimax-m3-1m-950dt-tp8.log 2>&1 &
        ```

    === "TP4 / PP2 / DP1"

        ```bash
        export WEIGHT_PATH="<WEIGHT_PATH>"  # MiniMax-M3 MXFP8 weights

        export VLLM_USE_V2_MODEL_RUNNER=1
        export HCCL_BUFFSIZE=256
        export HCCL_OP_EXPANSION_MODE="AIV"
        export HCCL_SOCKET_IFNAME="<NIC_NAME>"
        export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        export GLOO_SOCKET_IFNAME="<NIC_NAME>"
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

        source /usr/local/Ascend/cann-9.1.0/opp/vendors/experimental_950_transformer/bin/set_env.bash

        vllm serve "$WEIGHT_PATH" \
            --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
            --host 0.0.0.0 \
            --port 31050 \
            --served-model-name minimax-m3 \
            --trust-remote-code \
            --distributed-executor-backend mp \
            --tensor-parallel-size 4 \
            --pipeline-parallel-size 2 \
            --data-parallel-size 1 \
            --dtype bfloat16 \
            --quantization mxfp8 \
            --max-model-len 1048576 \
            --max-num-seqs 1 \
            --max-num-batched-tokens 8192 \
            --gpu-memory-utilization 0.92 \
            --kv-cache-dtype fp8 \
            --kv-cache-dtype-skip-layers 0 1 2 \
            --enable-chunked-prefill \
            --no-enable-prefix-caching \
            --no-async-scheduling \
            --reasoning-parser minimax_m3 \
            --safetensors-load-strategy prefetch \
            --additional-config '{
                "enable_cpu_binding": true,
                "ascend_compilation_config": {
                    "fuse_qknorm_rope": false,
                    "fuse_norm_quant": false,
                    "enable_static_kernel": false
                },
                "multistream_overlap_shared_expert": false,
                "enable_shared_expert_dp": false,
                "enable_flashcomm1": false
            }' > ./minimax-m3-1m-950dt-tp4pp2.log 2>&1 &
        ```

Wait for `Application startup complete` in the log file at the end of the selected command before sending requests. Check the reported KV cache capacity against the configured context length. These examples demonstrate the verified long-context deployment shapes; tune concurrency and performance separately for the target workload.

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

FLASHCOMM1 and `--language-model-only` can be enabled together for text-only serving, including the A3 1M-context example in [Section 5.4](#54-1m-token-context-deployment). Enable FLASHCOMM1 through `additional_config.enable_flashcomm1`. The `--language-model-only` flag is optional and must be omitted for multimodal inputs.

```bash
# Enable FLASHCOMM1.
--additional-config '{"enable_flashcomm1": true}'

# Enable language-model-only mode.
--language-model-only
```

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
export VLLM_USE_V2_MODEL_RUNNER=1

vllm serve ${WEIGHT_PATH} \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
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
export VLLM_USE_V2_MODEL_RUNNER=1

vllm serve ${WEIGHT_PATH} \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
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

The context-limit entries below use the unified serving setting `max-model-len=133000`. Accuracy scores are retained from the existing evaluations and have not been remeasured with this documentation update.

### 8.1 Using AISBench

For detailed instructions, refer to [Using AISBench for accuracy evaluation](../../developer_guide/evaluation/using_ais_bench.md).

### 8.2 Text Evaluation

| Dataset | Hardware | Score | max-model-len | max-num-seqs | max_out_len | batch_size | generation_kwargs |
|---------|----------|-------|---------------|--------------|-------------|------------|-------------------|
| GSM8K   | 8 H20 (96G × 8)     | 96.72 | 133000 | 16           | 49152       | 16         | temperature=1.0, top_p=0.95 |
| GSM8K   | 8 Atlas 800I A3 (64GB × 16)      | 96.36 | 133000 | 16           | 9500        | 20         | temperature=1.0, top_p=0.95 |
| AIME2025 | 8 H20 (96G × 8)     | 95@repeat4 | -        | -            | -           | -          | -                 |
| AIME2025 | 8 Atlas 800I A3 (64GB × 16)      | 93.3@repeat2    | 133000 | 32         | 65536           | 8         | temperature=1.0, top_p=0.95 |
| GPQA-Diamond | 8 H20 (96G × 8)     | 92.42    | 133000 | 64        | 75776       | 8       | temperature=0.6, top_p=0.95 |
| GPQA-Diamond | 8 Atlas 800I A3 (64GB × 16)      | 92.42    | 133000 | 32        | 65536       | 8       | temperature=0.6, top_p=0.95 |
| GPQA-Diamond | 8 950DT products (96GB × 8)      | 92.9    | 133000 | 128       | 131072       | 128       | temperature=0.6, top_p=0.95 |
| MMMU-pro | 8 950DT products (96GB × 8)      | 78.9    | 133000 | 128       | 131072       | 50       | temperature=0.6, top_p=0.95 |

### 8.3 Multimodal Evaluation

MiniMax-M3 multimodal accuracy is evaluated with AISBench. The ViT DP path is optional and can be enabled by adding `--mm-encoder-tp-mode data` to the serving command, but it is not required for all multimodal accuracy runs. For video evaluation, if no frame count is specified in the request or evaluation config, vLLM samples 32 frames by default.

The Video-MME results below are measured on chunk1 and chunk2, not the full dataset.

For Video-MME evaluation, run the vLLM OpenAI-compatible service with video input enabled and use AISBench to send the Video-MME requests. The official AISBench guide may not list Video-MME as a built-in example, so the recommended MiniMax-M3 settings are:

- serve with `--limit-mm-per-prompt '{"image":0,"video":1}'`;
- do not set `media_io_kwargs.video.num_frames`, so vLLM uses the default 32 sampled frames;
- use `max-model-len=133000` and `max_out_len=8192`;
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
| TextVQA | Image | AISBench | GPU | disabled | 133000 | 512 | `--limit-mm-per-prompt '{"image":1,"video":0}'` | temperature=1.0, top_p=0.95 | 70.82 |
| TextVQA | Image | AISBench | NPU | disabled | 133000 | 512 | `--limit-mm-per-prompt '{"image":1,"video":0}'` | temperature=1.0, top_p=0.95 | 72.75 |
| Video-MME chunk1+chunk2 | Video | AISBench | GPU | - | 133000 | 8192 | `--limit-mm-per-prompt '{"image":0,"video":1}'`, default 32 frames | temperature=1.0, top_p=0.95 | 73.41 |
| Video-MME chunk1+chunk2 | Video | AISBench | NPU | - | 133000 | 8192 | `--limit-mm-per-prompt '{"image":0,"video":1}'`, default 32 frames | temperature=1.0, top_p=0.95 | 74.21 |

## 9 Performance Tuning

> **Note**: The following configurations are validated in specific test environments and are for reference only. The optimal configuration depends on factors such as maximum input/output length, prefix cache hit rate, precision requirements, and deployment machine ratios. It is recommended to refer to Section 9.2 for tuning based on actual conditions.

### 9.1 Recommended Configurations

The recommended configurations are the same as those specified in Chapter 5, "Online Service Deployment."

### 9.2 Tuning Guidelines

#### 9.2.1 General Tuning Reference

Please refer to the [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md) for general tuning methods.

Please refer to the [Feature Matrix](../../user_guide/support_matrix/feature_matrix.md) for detailed feature descriptions.

#### 9.2.2 Workload-Specific Tuning

Use the following adjustments as starting points for the target workload. Recheck memory usage, latency, and throughput after each change.

| Scenario | Parameter | Example Adjustment | Tuning Goal |
| --- | --- | --- | --- |
| Device-memory-bound workloads where KV cache capacity limits concurrency | `HCCL_BUFFSIZE` | Reduce from `2048` to `1024` (MB). | Reduce HCCL communication-buffer memory usage to make more device memory available for KV cache and allow higher concurrency. |
| Short-input workloads, such as 4K input tokens | `--max-num-batched-tokens` | Reduce from `32768` to `8192`. | Use a smaller per-iteration token budget to reduce memory pressure and tune short-sequence performance. |

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
