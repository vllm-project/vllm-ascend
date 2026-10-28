# Qwen3.5-397B-A17B

## 1 Introduction

!!! note

    The "PD Separation Deployment" mentioned in this document is provided only as an example/reference design to demonstrate the basic process of deploying Prefill and Decode in separated mode. It is not recommended for use in the production environment.

    **For the production environment on the customer side, [MindIE Motor](https://www.hiascend.com/document/detail/en/mindie/latest/quickstart/docs/en/user_guide/quick_start.md) + vLLM-Ascend is recommended.**

Qwen3.5-397B-A17B is a large-scale Qwen3.5 MoE model that combines multimodal capability, long-context inference, MTP speculative decoding, and W8A8 quantized deployment for production serving on Ascend hardware.

This document describes the main validation steps for the model, including supported features, prerequisites, installation, single-node online deployment, multi-node deployment, Prefill-Decode (PD) disaggregation, functional verification, accuracy and performance evaluation, performance tuning, and FAQs.

`Qwen3.5-397B-A17B` is supported starting from **vLLM-Ascend v0.17.0rc1**. For 950DT Products, model support starts from **vLLM-Ascend v0.23.0rc1**. This document is written based on **vLLM-Ascend v0.30.0**. To use the latest features, it is recommended to use the latest release candidate or official version.

## 2 Supported Features

Refer to the [model support matrix](../../user_guide/support_matrix/supported_models.md) for the model's supported features, including BF16, W8A8 quantization, chunked prefill, automatic prefix caching, speculative decoding, asynchronous scheduling, tensor parallelism, expert parallelism, data parallelism, PD disaggregation, and ACLGraph support.

Refer to the [Feature Guide](../../user_guide/feature_guide/index.md) for feature configuration details.

!!! note

    The startup examples use configuration values for specific workloads. Adjust `--max-model-len`, `--max-num-seqs`, and `--max-num-batched-tokens` based on your service workload and available KV cache.

## 3 Prerequisites

### 3.1 Model Weight

| Weight Version                                       | Hardware Requirements | KV Cache Selection | Download Links |
|------------------------------------------------------|-----------------------|--------------------|----------------|
| `Qwen3.5-397B-A17B` (BF16 version)                   | 2 950DT Products (96 GB x 8) nodes or 2 Atlas 800I A3 (64 GB x 16) nodes or 4 Atlas 800I A2 (64 GB x 8) nodes | `auto` for the baseline | [ModelScope](https://www.modelscope.cn/models/Qwen/Qwen3.5-397B-A17B) |
| `Qwen3.5-397B-A17B-w8a8` (quantized version)         | 1 Atlas 800I A3 (64 GB x 16) node or 2 Atlas 800I A2 (64 GB x 8) nodes | Follow the checkpoint KV quantization recipe | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/Qwen3.5-397B-A17B-w8a8-mtp) |
| `Qwen3.5-397B-A17B-w4a8` (quantized version)         | 1 Atlas 800I A3 (64 GB x 16) node or 2 Atlas 800I A2 (64 GB x 8) nodes | Follow the checkpoint KV quantization recipe | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/Qwen3.5-397B-A17B-w4a8-mtp) |
| `Qwen3.5-397B-A17B-w8a8-mxfp8` (quantized version)   | 1 950DT Products (96 GB x 8) node | Check KV metadata; the model name alone is insufficient | [ModelScope](https://modelscope.cn/models/Eco-Tech/Qwen3.5-397B-A17B-w8a8-mxfp8) |
| `Qwen3.5-397B-A17B-w4a4-mxfp4` (quantized version)   | 1 950DT Products (96 GB x 8) node | Check KV metadata; the model name alone is insufficient | [ModelScope](https://modelscope.cn/models/Eco-Tech/Qwen3.5-397B-A17B-w4a4-mxfp4) |
| `Qwen3.5-397B-A17B-w8a8c8-mxfp8` (quantized version) | 1 950DT Products (96 GB x 8) node | `mxfp8` for the supported C8 recipe | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/Qwen3.5-397B-A17B-w8a8c8-mxfp8) |
| `Qwen3.5-397B-A17B-w4a8c8-mxfp` (quantized version)  | 1 950DT Products (96 GB x 8) node | `mxfp8` for the C8 recipe in [Section 5.5](#55-prefill-decode-disaggregation-950dt-products) | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/Qwen3.5-397B-A17B-w4a8c8-mxfp) |

!!! note

    `--quantization ascend` selects the Ascend model quantization implementation; `--kv-cache-dtype mxfp8` selects MXFP8 KV cache storage. They configure different parts of inference. Enable `mxfp8` only with a supported checkpoint and its required KV scale metadata. A W4A8/W8A8 name, or an MXFP weight format, does not by itself establish C8 KV cache compatibility. Do not copy a W4A8C8 or W8A8C8 KV setting to a different checkpoint without checking its quantization description and serving logs. Keep the weight revision, tokenizer, quantization metadata, and KV cache settings consistent on P and D.

It is recommended to download the model weights to a shared directory across multiple nodes, such as `/root/.cache/`, so that all serving nodes can load the same path.

> **Path description**: Download the model weights to a directory of your choice and record it. Ensure the model path in the subsequent deployment command matches this directory.

### 3.2 Verify Multi-node Communication

For multi-node deployment, verify inter-node connectivity by following [Verify Multi-node Communication](../../getting_started/installation.md#installation-multi-node-interconnect).

Use mutually reachable service IPs on all nodes. In the scripts below, `local_ip` must belong to `nic_name`.

For 950DT PD deployment, prepare the [Mooncake communication environment](../features/pd_disaggregation_mooncake_multi_node.md) and mount the required host-specific communication resources into each serving container. When `ascend_local_comm_res_path=/etc/hixlep` is set, ensure that `/etc/hixlep` contains the configuration required for that host.

## 4 Installation

### 4.1 Docker Image Installation

Select an image for your hardware and start a container on each node. Refer to [Using Docker](../../getting_started/installation.md#installation-prebuilt-image) for details.

For PD deployment, use the same vLLM-Ascend release image and dependency versions on P and D.

=== "950DT Products"

    Start a Docker container on each node.

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a5
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
      --device /dev/davinci_manager \
      --device /dev/hisi_hdc \
      --device /dev/ummu \
      --device /dev/uburma \
      -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
      -v /etc/ascend_install.info:/etc/ascend_install.info \
      -v /etc/hccl_rootinfo.json:/etc/hccl_rootinfo.json \
      -v /etc/hixlep/:/etc/hixlep/ \
      -v /root/.cache:/root/.cache \
      -v /usr/local/sbin:/usr/local/sbin \
      -v /usr/local/dcmi:/usr/local/dcmi \
      -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
      -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
      -itd $IMAGE bash
    ```

    Keep the container's `/usr/lib64` from the selected image. Mounting the entire host directory over it can hide image-provided system libraries and break the matched CANN/runtime dependencies; making that whole-directory mount read-only does not prevent library shadowing. If the selected URMA/RoCE transport requires additional host libraries, mount only the specific compatible files or vendor directory required by its installation guide, read-only, and verify the library loader paths inside the container.

=== "A3 series"

    Start a Docker container on each node.

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a3
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
      -v /root/.cache:/root/.cache \
      -it $IMAGE bash
    ```

=== "A2 series"

    Start a Docker container on each node.

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
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

After entering the actual serving container on each node, check the executables, versions, and imported paths:

```shell
command -v python3 vllm
python3 - <<'PY'
import importlib.metadata as metadata
import sys

import torch_npu
import vllm
import vllm_ascend

print("Python:", sys.executable)
for module in (vllm, vllm_ascend, torch_npu):
    print(module.__name__, module.__file__)
for package in (
    "vllm", "vllm-ascend", "torch", "torch-npu", "triton-ascend",
    "mooncake-transfer-engine-npu", "mooncake-transfer-engine",
):
    try:
        print(package, metadata.version(package))
    except metadata.PackageNotFoundError:
        print(package, "not installed under this distribution name")
PY
```

If you want to deploy a multi-node service, set up the same environment on each node.

### 4.2 Source Code Installation

You can also build and install `vllm-ascend` from source. Refer to [set up using python](../../getting_started/installation.md#installation-existing-cann-install).

If you want to deploy a multi-node service, install the same version of vLLM and vLLM-Ascend on each node.

## 5 Online Service Deployment {: #5-online-service-deployment }

### 5.1 Single-Node Online Deployment

Single-node deployment runs both Prefill and Decode on the same node. It is suitable for functional validation and long-context serving.

=== "950DT Products"

    Run the following script for online inference on one 950DT Products node (96 GB x 8). The quantized versions (`Qwen3.5-397B-A17B-w8a8-mxfp8` and `Qwen3.5-397B-A17B-w4a4-mxfp4`) can be deployed on a single node and require `--quantization ascend`.

    ```shell
    #!/bin/sh

    # Load model from ModelScope to speed up download.
    export VLLM_USE_MODELSCOPE=True
    export HCCL_BUFFSIZE=400
    export HCCL_INTRA_ROCE_ENABLE=0
    export HCCL_OP_EXPANSION_MODE="AIV"
    export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1
    export VLLM_USE_V2_MODEL_RUNNER=0

    # Ensure the model path matches the directory recorded during download
    MODEL_PATH="Eco-Tech/Qwen3.5-397B-A17B-w4a4-mxfp4"
    vllm serve "${MODEL_PATH}" \
      --host 0.0.0.0 \
      --port 8000 \
      --distributed-executor-backend mp \
      --data-parallel-size 1 \
      --tensor-parallel-size 8 \
      --enable-expert-parallel \
      --seed 1024 \
      --quantization ascend \
      --served-model-name qwen3.5 \
      --max-num-seqs 128 \
      --max-model-len 133000 \
      --max-num-batched-tokens 8192 \
      --trust-remote-code \
      --enable-prefix-caching \
      --gpu-memory-utilization 0.90 \
      --async-scheduling \
      --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3}' \
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
      --additional-config '{"enable_cpu_binding": true, "enable_fused_mc2": 1, "enable_flashcomm1": true}'
    ```

=== "A3 series"

    Run the following script for online inference with a 133k context on one Atlas 800I A3 node (64 GB x 16). W8A8 deployment requires `--quantization ascend`.

    ```shell
    #!/bin/sh

    # Load model from ModelScope to speed up download.
    export VLLM_USE_MODELSCOPE=True
    export HCCL_BUFFSIZE=1024
    export HCCL_OP_EXPANSION_MODE="AIV"
    export VLLM_USE_V2_MODEL_RUNNER=0

    echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
    sysctl -w vm.swappiness=0
    sysctl -w kernel.numa_balancing=0
    sysctl kernel.sched_migration_cost_ns=50000

    # Ensure the model path matches the directory recorded during download
    MODEL_PATH="Eco-Tech/Qwen3.5-397B-A17B-w8a8-mtp"
    vllm serve "${MODEL_PATH}" \
      --host 0.0.0.0 \
      --port 8000 \
      --data-parallel-size 1 \
      --tensor-parallel-size 16 \
      --enable-expert-parallel \
      --seed 1024 \
      --quantization ascend \
      --served-model-name qwen3.5 \
      --max-num-seqs 128 \
      --max-model-len 133000 \
      --max-num-batched-tokens 16384 \
      --trust-remote-code \
      --gpu-memory-utilization 0.90 \
      --enable-prefix-caching \
      --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}' \
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
      --additional-config '{"enable_cpu_binding": true, "enable_fused_mc2": 1, "enable_flashcomm1": true}'
    ```

=== "A2 series"

    For W8A8 deployment, 2 Atlas 800I A2 (64 GB x 8) nodes are required. Refer to [Section 5.2](#52-multi-node-deployment-with-mp-recommended) for multi-node MP deployment.

Common Issues Tip: If the service fails to start, HBM is insufficient, or requests are not scheduled as expected, refer to [Public FAQs](../../faqs.md) first, and then check the model-specific FAQ in [Section 10](#10-faq).

**Key parameters:**

- `--data-parallel-size 1` uses one DP group. `--tensor-parallel-size` is 16 for the A3 example and 8 for the 950DT example.
- `--enable-expert-parallel` enables expert parallelism for MoE layers. Do not mix MoE tensor parallelism and expert parallelism in the same MoE layer.
- `--max-model-len` is the maximum input plus output length for a single request. Increase it only when enough KV cache is available.
- `--max-num-seqs` is the maximum number of active requests scheduled by each DP group. For performance tests, set `--max-num-seqs * --data-parallel-size` greater than or equal to the test concurrency.
- `--max-num-batched-tokens` is the maximum number of tokens processed in one scheduler step. A larger value can improve prefill efficiency but consumes more activation memory.
- `--gpu-memory-utilization` controls how much HBM vLLM can use to calculate KV cache capacity. A higher value increases KV cache size but can trigger OOM if runtime memory is higher than the profile run.
- `--enable-prefix-caching` enables prefix caching. For Qwen3.5, short prefixes may not be cached when the hybrid KV cache manager adjusts the block size to a large value.
- `--quantization ascend` enables Ascend quantization for the quantized weights used above. Remove this option when deploying the BF16 model.
- `--speculative-config` enables Qwen3.5 MTP speculative decoding. Reduce `num_speculative_tokens` or remove this option if the workload is sensitive to first-token latency or if MTP is unstable in your environment.
- `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` enables full decode ACLGraph replay to reduce dispatch overhead.
- `VLLM_USE_V2_MODEL_RUNNER=0` selects Model Runner V1 for the single-node examples above.
- `--additional-config` configures Ascend-specific optimizations: `enable_cpu_binding=true` enables Ascend-native CPU binding; `enable_fused_mc2=1` selects dispatch-FFN-combine and is used by the single-node 950DT and A3 examples for compatibility with the documented MTP and batching configurations; `2` selects CANN MegaMoe and should only be enabled with a matching CANN/operator stack and supported token-shape limits; `0` disables the fused path. `enable_flashcomm1=true` enables MoE sequence parallelism across TP ranks and requires `--enable-expert-parallel` with a TP size greater than 1.

### 5.2 Multi-Node Deployment with MP (Recommended)

Multi-node MP deployment uses vLLM data parallelism across nodes and tensor parallelism within each node. It is recommended for the W8A8 model on 2 Atlas 800I A2 (64 GB x 8) nodes.

Assume you have 2 Atlas 800I A2 nodes and want to deploy `Qwen3.5-397B-A17B-w8a8-mtp` across them. Replace `nic_name`, `local_ip`, and `node0_ip` with the actual network interface and IP addresses in your environment.

Run the following script on node 0.

```shell
#!/bin/sh

# Get these values through ifconfig.
# nic_name is the network interface name corresponding to local_ip.
nic_name="xxxx"
local_ip="xxxx"
export IP_ADDRESS="${local_ip}"
export NETWORK_CARD_NAME="${nic_name}"

export VLLM_USE_MODELSCOPE=True
export HCCL_BUFFSIZE=1024
export HCCL_IF_IP=$local_ip
export HCCL_SOCKET_IFNAME=$nic_name
export GLOO_SOCKET_IFNAME=$nic_name

# Ensure the model path matches the directory recorded during download
vllm serve Eco-Tech/Qwen3.5-397B-A17B-w8a8-mtp \
  --host 0.0.0.0 \
  --port 8000 \
  --data-parallel-size 2 \
  --api-server-count 2 \
  --data-parallel-size-local 1 \
  --data-parallel-address $local_ip \
  --data-parallel-rpc-port 13389 \
  --seed 1024 \
  --served-model-name qwen3.5 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --max-num-seqs 16 \
  --max-model-len 32768 \
  --max-num-batched-tokens 4096 \
  --trust-remote-code \
  --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching \
  --quantization ascend \
  --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --additional-config '{"enable_cpu_binding":true, "multistream_overlap_shared_expert": true}'
```

Common Issues Tip: If node 1 cannot join the service or HCCL initialization times out, refer to [verify multi-node communication environment](../../getting_started/installation.md#installation-multi-node-interconnect) and [FAQs](../../faqs.md). Make sure the network interface names, IP addresses, and RPC ports are consistent across nodes.

Run the following script on node 1.

```shell
#!/bin/sh

# Get these values through ifconfig.
# nic_name is the network interface name corresponding to local_ip.
nic_name="xxxx"
local_ip="xxxx"
export IP_ADDRESS="${local_ip}"
export NETWORK_CARD_NAME="${nic_name}"

export VLLM_USE_MODELSCOPE=True
export HCCL_BUFFSIZE=1024
export HCCL_IF_IP=$local_ip
export HCCL_SOCKET_IFNAME=$nic_name
export GLOO_SOCKET_IFNAME=$nic_name

# The value of node0_ip must be consistent with local_ip on node 0.
node0_ip="xxxx"

# Ensure the model path matches the directory recorded during download
vllm serve Eco-Tech/Qwen3.5-397B-A17B-w8a8-mtp \
  --host 0.0.0.0 \
  --port 8000 \
  --headless \
  --data-parallel-size 2 \
  --data-parallel-size-local 1 \
  --data-parallel-start-rank 1 \
  --data-parallel-address $node0_ip \
  --data-parallel-rpc-port 13389 \
  --seed 1024 \
  --tensor-parallel-size 8 \
  --served-model-name qwen3.5 \
  --max-num-seqs 16 \
  --max-model-len 32768 \
  --max-num-batched-tokens 4096 \
  --enable-expert-parallel \
  --trust-remote-code \
  --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching \
  --quantization ascend \
  --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --additional-config '{"enable_cpu_binding":true, "multistream_overlap_shared_expert": true}'
```

Common Issues Tip: If the headless node exits immediately, check whether node 0 is already running, whether `--data-parallel-address` points to node 0, and whether `--data-parallel-start-rank` is unique for each node.

If the service starts successfully, the following information is displayed on node 0:

```output
INFO:     Started server process [44610]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Started server process [44611]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

**Key parameters for MP deployment:**

- `--data-parallel-size` is the global DP size across all nodes. In the example, 2 DP ranks are used.
- `--data-parallel-size-local` is the number of DP ranks on the current node. In the example, each A2 node has 1 local DP rank.
- `--data-parallel-start-rank` is the first DP rank on the current node. Node 0 starts from 0 by default, and node 1 starts from 1.
- `--data-parallel-address` must point to the master DP node. Use node 0 `local_ip` on node 0 and `node0_ip` on other nodes.
- `--data-parallel-rpc-port` is the DP RPC port. Use the same value on all nodes and ensure the port is available.
- `--api-server-count` controls how many API server processes are started on the master node.
- `--headless` starts a worker node without exposing an API server. Use it on non-master nodes.
- `--tensor-parallel-size 8` maps one TP group to the 8 NPUs on each A2 node.
- `HCCL_IF_IP`, `GLOO_SOCKET_IFNAME`, and `HCCL_SOCKET_IFNAME` bind HCCL and Gloo communication to the selected network.
- `multistream_overlap_shared_expert` overlaps shared expert computation for better throughput on MoE workloads.

### 5.3 Multi-Node Deployment with Ray

For Ray-based distributed deployment, refer to [Ray Distributed (Qwen/Qwen3-235B-A22B)](../features/ray.md). The same model weight, communication verification, and parameter tuning principles apply to Qwen3.5-397B-A17B.

Common Issues Tip: If Ray workers cannot discover each other, check Ray cluster status first, then verify the same HCCL and network interface settings used in MP deployment.

### 5.4 Prefill-Decode Disaggregation (A3)

PD disaggregation separates Prefill and Decode into different service groups. This performance-oriented example deploys 1P1D on one 16-NPU Atlas 800I A3 host with the `Qwen3.5-397B-A17B-w4a8` checkpoint. In this section, 1P1D describes two service groups; it does not mean that P and D must occupy separate hosts.

The 16 NPUs are divided into two non-overlapping device sets:

| Service group | Physical devices | TP | DP | EP world size |
|---------------|------------------|----|----|---------------|
| Prefill | 0-7 | 8 | 1 | 8 |
| Decode | 8-15 | 4 | 2 | 8 |

Prefill and Decode use the same service NIC and IP because they run on the same host, but they use different HTTP and Mooncake ports. The Decode launcher creates two local DP engines, each with TP4. Do not start a second copy of the Decode command.

We recommend using Mooncake for deployment. Refer to [Mooncake](../features/pd_disaggregation_mooncake_multi_node.md) for the general PD disaggregation workflow. This example uses `MooncakeConnectorV1` with the standard Pull proxy in [Section 5.7](#57-request-forwarding).

#### 5.4.1 Prefill Service Group

Create `run_p.sh` on the A3 host. Set `nic_name`, `local_ip`, and `MODEL_PATH` for the local environment.

```shell
#!/usr/bin/env bash
set -euo pipefail

unset ftp_proxy
unset https_proxy
unset http_proxy

nic_name="<SERVICE_NETWORK_INTERFACE>"
local_ip="<LOCAL_SERVICE_IP>"
MODEL_PATH="<LOCAL_W4A8_MODEL_PATH>"

export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["hccs:device"]}'
export ASCEND_LOCAL_COMM_RES='{"version":"1.2"}'

export IP_ADDRESS="${local_ip}"
export NETWORK_CARD_NAME="${nic_name}"
export HCCL_IF_IP="${local_ip}"
export GLOO_SOCKET_IFNAME="${nic_name}"
export HCCL_SOCKET_IFNAME="${nic_name}"

export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE=AIV
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_USE_FASTOKENS=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LD_LIBRARY_PATH="/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:${LD_LIBRARY_PATH:-}"

if [[ -f /usr/lib/aarch64-linux-gnu/libjemalloc.so.2 ]]; then
    export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:${LD_PRELOAD:-}"
fi

vllm serve "${MODEL_PATH}" \
  --host "${local_ip}" \
  --port 30060 \
  --data-parallel-size 1 \
  --data-parallel-size-local 1 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --seed 1024 \
  --served-model-name qwen3.5 \
  --max-model-len 133120 \
  --max-num-batched-tokens 20480 \
  --max-num-seqs 64 \
  --enforce-eager \
  --trust-remote-code \
  --allowed-local-media-path / \
  --gpu-memory-utilization 0.90 \
  --quantization ascend \
  --enable-prefix-caching \
  --async-scheduling \
  --mm-processor-cache-gb 0 \
  --mm-encoder-tp-mode data \
  --distributed-executor-backend mp \
  --no-disable-hybrid-kv-cache-manager \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}' \
  --additional-config '{
    "enable_flashcomm1": true,
    "ascend_compilation_config": {
      "enable_npugraph_ex": false,
      "enable_static_kernel": false
    },
    "weight_nz_mode": 1,
    "enable_cpu_binding": true,
    "enable_fused_mc2": 2,
    "enable_shared_expert_dp": true,
    "multistream_overlap_shared_expert": false,
    "scheduler_config": {
      "recompute_scheduler_enable": false
    }
  }' \
  --kv-transfer-config '{
    "kv_connector": "MooncakeConnectorV1",
    "kv_buffer_device": "npu",
    "kv_role": "kv_producer",
    "kv_port": "55220",
    "engine_id": "0",
    "kv_connector_extra_config": {
      "prefill": {
        "dp_size": 1,
        "tp_size": 8
      },
      "decode": {
        "dp_size": 2,
        "tp_size": 4
      }
    }
  }'
```

Run the Prefill script:

```shell
bash run_p.sh
```

If the Prefill service starts successfully, information similar to the following example is displayed:

```text
INFO:     Started server process [44610]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

Common Issues Tip: If the Prefill service does not become ready, confirm that devices 0-7 are available, port 30060 and Mooncake port 55220 are unused, and `local_ip` belongs to `nic_name`.

#### 5.4.2 Decode Service Group

Create `run_d.sh` on the same host. Use the same `nic_name`, `local_ip`, and checkpoint revision as Prefill.

```shell
#!/usr/bin/env bash
set -euo pipefail

unset ftp_proxy
unset https_proxy
unset http_proxy

nic_name="<SERVICE_NETWORK_INTERFACE>"
local_ip="<LOCAL_SERVICE_IP>"
MODEL_PATH="<LOCAL_W4A8_MODEL_PATH>"

export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["hccs:device"]}'
export ASCEND_LOCAL_COMM_RES='{"version":"1.2"}'

export IP_ADDRESS="${local_ip}"
export NETWORK_CARD_NAME="${nic_name}"
export HCCL_IF_IP="${local_ip}"
export GLOO_SOCKET_IFNAME="${nic_name}"
export HCCL_SOCKET_IFNAME="${nic_name}"

export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE=AIV
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_USE_FASTOKENS=1
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export LD_LIBRARY_PATH="/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:${LD_LIBRARY_PATH:-}"

if [[ -f /usr/lib/aarch64-linux-gnu/libjemalloc.so.2 ]]; then
    export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:${LD_PRELOAD:-}"
fi

vllm serve "${MODEL_PATH}" \
  --host "${local_ip}" \
  --port 30050 \
  --data-parallel-size 2 \
  --data-parallel-size-local 2 \
  --data-parallel-address "${local_ip}" \
  --data-parallel-rpc-port 6884 \
  --api-server-count 1 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --seed 1024 \
  --served-model-name qwen3.5 \
  --max-model-len 133120 \
  --max-num-batched-tokens 300 \
  --max-num-seqs 100 \
  --trust-remote-code \
  --allowed-local-media-path / \
  --gpu-memory-utilization 0.90 \
  --quantization ascend \
  --enable-prefix-caching \
  --async-scheduling \
  --mm-processor-cache-gb 0 \
  --mm-encoder-tp-mode data \
  --distributed-executor-backend mp \
  --no-disable-hybrid-kv-cache-manager \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --additional-config '{
    "enable_flashcomm1": true,
    "ascend_compilation_config": {
      "enable_npugraph_ex": true,
      "enable_static_kernel": false
    },
    "weight_nz_mode": 1,
    "enable_cpu_binding": true,
    "enable_fused_mc2": 2,
    "enable_shared_expert_dp": true,
    "multistream_overlap_shared_expert": false,
    "scheduler_config": {
      "recompute_scheduler_enable": true
    }
  }' \
  --kv-transfer-config '{
    "kv_connector": "MooncakeConnectorV1",
    "kv_buffer_device": "npu",
    "kv_role": "kv_consumer",
    "kv_port": "56010",
    "engine_id": "2",
    "kv_connector_extra_config": {
      "prefill": {
        "dp_size": 1,
        "tp_size": 8
      },
      "decode": {
        "dp_size": 2,
        "tp_size": 4
      }
    }
  }'
```

Run the Decode script in a separate terminal:

```shell
bash run_d.sh
```

If the Decode service starts successfully, information similar to the following example is displayed:

```text
INFO:     Started server process [44611]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

Common Issues Tip: If Decode fails to initialize, confirm that devices 8-15 are available and that `--data-parallel-size 2`, `--data-parallel-size-local 2`, `--tensor-parallel-size 4`, and the Mooncake topology describe the same TP4/DP2 layout. Ports 30050, 6884, and 56010 must be available.

#### 5.4.3 Proxy

After both services report ready, start the standard Pull proxy on the same A3 host. Because this is a single-host 1P1D topology, Prefill, Decode, and the proxy use the same `local_ip` and are distinguished by their ports.

Run the command from the directory containing `load_balance_proxy_server_example.py`:

```shell
unset ftp_proxy
unset https_proxy
unset http_proxy

LOCAL_SERVICE_IP="<LOCAL_SERVICE_IP>"

python3 load_balance_proxy_server_example.py \
  --prefiller-hosts "${LOCAL_SERVICE_IP}" \
  --prefiller-ports 30060 \
  --decoder-hosts "${LOCAL_SERVICE_IP}" \
  --decoder-ports 30050 \
  --host "${LOCAL_SERVICE_IP}" \
  --port 8000
```

Send client requests to `http://<LOCAL_SERVICE_IP>:8000`.

**Key parameters for PD disaggregation:**

- Prefill uses TP8/DP1 on devices 0-7. Decode uses TP4/DP2 on devices 8-15. With expert parallelism enabled, both service groups have an EP world size of 8.
- The P and D device lists must not overlap. P and D can share the same service IP on a single host, but their HTTP and Mooncake ports must be different.
- The Decode command starts both local DP ranks. Do not launch one Decode command per DP rank.
- `VLLM_USE_V2_MODEL_RUNNER=0` selects Model Runner V1 on both service groups.
- Use the same MTP method and `num_speculative_tokens` on P and D. To disable MTP for a comparison, remove `--speculative-config` from both commands.
- `kv_connector_extra_config.prefill` must remain DP1/TP8, and `decode` must remain DP2/TP4 for this topology.
- `enable_flashcomm1=true`, `enable_fused_mc2=2`, and `enable_shared_expert_dp=true` reproduce the performance-case MoE configuration. They require the matching CANN operators and software stack.
- Prefill runs eagerly. Decode uses `FULL_DECODE_ONLY`, enables NPU graph extensions, and enables the recompute scheduler.
- `--max-num-batched-tokens 300` and `--max-num-seqs 100` are Decode scheduler limits applied to each DP engine by vLLM.
- Prefix caching and asynchronous scheduling are enabled on both service groups in this validated case. Revalidate output correctness if either setting or the connector version changes.

### 5.5 Prefill-Decode Disaggregation (950DT Products)

This performance-oriented example deploys 1P1D on two 8-NPU 950DT hosts with the [Qwen3.5-397B-A17B-w4a8c8-mxfp](https://www.modelscope.cn/models/Eco-Tech/Qwen3.5-397B-A17B-w4a8c8-mxfp) checkpoint. Set `MODEL_PATH` on both nodes to the local directory containing this checkpoint:

| Node role | NPUs | TP | DP | EP world size |
|-----------|------|----|----|---------------|
| Prefill | 8 | 8 | 1 | 8 |
| Decode | 8 | 4 | 2 | 8 |

Both nodes must use the same checkpoint revision, tokenizer, quantization metadata, and `--kv-cache-dtype mxfp8`. This example uses `MooncakeConnectorV2`: the proxy calls Prefill first, and Decode pulls the KV cache. The connector topology must remain DP1/TP8 for Prefill and DP2/TP4 for Decode.

Use a high-speed service network that is reachable in both directions. On each host, `local_ip` must be the IP assigned to `nic_name`. Prepare and mount that host's own HIXL communication resources at `/etc/hixlep` as described in [Section 3.2](#32-verify-multi-node-communication). Do not reuse another host's endpoint resources.

#### 5.5.1 Prefill Node

Create `run_p.sh` on the Prefill host. Replace the NIC, IP, and model-path placeholders. The HTTP port is 30060 and the Mooncake port is 55220.

```shell
#!/usr/bin/env bash
set -euo pipefail

unset ftp_proxy
unset https_proxy
unset http_proxy

nic_name="<P_HIGH_SPEED_SERVICE_NIC>"
local_ip="<P_HIGH_SPEED_SERVICE_IP>"
MODEL_PATH="<LOCAL_W4A8C8_MODEL_PATH>"

export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000

export IP_ADDRESS="${local_ip}"
export NETWORK_CARD_NAME="${nic_name}"
export HCCL_IF_IP="${local_ip}"
export GLOO_SOCKET_IFNAME="${nic_name}"
export HCCL_SOCKET_IFNAME="${nic_name}"

export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE=AIV
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_USE_FASTOKENS=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LD_LIBRARY_PATH="/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:${LD_LIBRARY_PATH:-}"

vllm serve "${MODEL_PATH}" \
  --host "${local_ip}" \
  --port 30060 \
  --data-parallel-size 1 \
  --data-parallel-size-local 1 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --seed 1024 \
  --served-model-name qwen3.5 \
  --max-model-len 133120 \
  --max-num-batched-tokens 20480 \
  --max-num-seqs 64 \
  --enforce-eager \
  --trust-remote-code \
  --allowed-local-media-path / \
  --gpu-memory-utilization 0.90 \
  --quantization ascend \
  --kv-cache-dtype mxfp8 \
  --enable-prefix-caching \
  --async-scheduling \
  --mm-processor-cache-gb 0 \
  --mm-encoder-tp-mode data \
  --distributed-executor-backend mp \
  --no-disable-hybrid-kv-cache-manager \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":true}' \
  --additional-config '{
    "enable_flashcomm1": true,
    "ascend_compilation_config": {
      "enable_npugraph_ex": false,
      "enable_static_kernel": false
    },
    "weight_nz_mode": 1,
    "enable_cpu_binding": true,
    "enable_fused_mc2": 0,
    "enable_prefill_mc2": false,
    "enable_shared_expert_dp": true,
    "multistream_overlap_shared_expert": false,
    "scheduler_config": {
      "recompute_scheduler_enable": false
    }
  }' \
  --kv-transfer-config '{
    "kv_connector": "MooncakeConnectorV2",
    "kv_buffer_device": "npu",
    "kv_role": "kv_producer",
    "kv_port": "55220",
    "engine_id": "0",
    "kv_connector_extra_config": {
      "ascend_local_comm_res_path": "/etc/hixlep",
      "prefill": {
        "dp_size": 1,
        "tp_size": 8
      },
      "decode": {
        "dp_size": 2,
        "tp_size": 4
      }
    }
  }'
```

Run the Prefill script on the Prefill node:

```shell
bash run_p.sh
```

If the Prefill service starts successfully, information similar to the following example is displayed:

```text
INFO:     Started server process [44610]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

Common Issues Tip: If Prefill does not become ready, verify the checkpoint metadata, all eight visible NPUs, the selected high-speed IP/NIC pair, port 30060, Mooncake port 55220, and the host-specific `/etc/hixlep` resources.

#### 5.5.2 Decode Node

Create `run_d.sh` on the Decode host. A single launcher starts two local DP engines, each using TP4. Do not start two copies of this command. Replace the NIC, IP, and model-path placeholders. The HTTP port is 30050, the DP RPC port is 6884, and the Mooncake port is 56010.

```shell
#!/usr/bin/env bash
set -euo pipefail

unset ftp_proxy
unset https_proxy
unset http_proxy

nic_name="<D_HIGH_SPEED_SERVICE_NIC>"
local_ip="<D_HIGH_SPEED_SERVICE_IP>"
MODEL_PATH="<LOCAL_W4A8C8_MODEL_PATH>"

export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000

export IP_ADDRESS="${local_ip}"
export NETWORK_CARD_NAME="${nic_name}"
export HCCL_IF_IP="${local_ip}"
export GLOO_SOCKET_IFNAME="${nic_name}"
export HCCL_SOCKET_IFNAME="${nic_name}"

export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE=AIV
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_USE_FASTOKENS=1
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LD_LIBRARY_PATH="/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:${LD_LIBRARY_PATH:-}"

vllm serve "${MODEL_PATH}" \
  --host "${local_ip}" \
  --port 30050 \
  --data-parallel-size 2 \
  --data-parallel-size-local 2 \
  --data-parallel-address "${local_ip}" \
  --data-parallel-rpc-port 6884 \
  --api-server-count 1 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --seed 1024 \
  --served-model-name qwen3.5 \
  --max-model-len 133120 \
  --max-num-batched-tokens 300 \
  --max-num-seqs 100 \
  --trust-remote-code \
  --allowed-local-media-path / \
  --gpu-memory-utilization 0.90 \
  --quantization ascend \
  --kv-cache-dtype mxfp8 \
  --enable-prefix-caching \
  --async-scheduling \
  --mm-processor-cache-gb 0 \
  --mm-encoder-tp-mode data \
  --distributed-executor-backend mp \
  --no-disable-hybrid-kv-cache-manager \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":true}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --additional-config '{
    "enable_flashcomm1": true,
    "ascend_compilation_config": {
      "enable_npugraph_ex": true,
      "enable_static_kernel": false
    },
    "weight_nz_mode": 1,
    "enable_cpu_binding": true,
    "enable_fused_mc2": 0,
    "enable_prefill_mc2": false,
    "enable_shared_expert_dp": true,
    "multistream_overlap_shared_expert": false,
    "scheduler_config": {
      "recompute_scheduler_enable": true
    }
  }' \
  --kv-transfer-config '{
    "kv_connector": "MooncakeConnectorV2",
    "kv_buffer_device": "npu",
    "kv_role": "kv_consumer",
    "kv_port": "56010",
    "engine_id": "2",
    "kv_connector_extra_config": {
      "ascend_local_comm_res_path": "/etc/hixlep",
      "prefill": {
        "dp_size": 1,
        "tp_size": 8
      },
      "decode": {
        "dp_size": 2,
        "tp_size": 4
      }
    }
  }'
```

Run the Decode script on the Decode node:

```shell
bash run_d.sh
```

If the Decode service starts successfully, information similar to the following example is displayed:

```text
INFO:     Started server process [44611]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

Common Issues Tip: If Decode fails to initialize, confirm that `--data-parallel-size 2`, `--data-parallel-size-local 2`, `--tensor-parallel-size 4`, and the Mooncake topology describe the same DP2/TP4 layout. Verify ports 30050, 6884, and 56010, and confirm that both local DP engines start.

#### 5.5.3 Proxy

After both services report ready, start the standard Pull proxy on the Prefill host. Use the same reachable service IPs assigned to `local_ip` in the P and D commands. The Decode backend is the HTTP port `30050`, not the DP RPC port `6884`.

Run the command from the directory containing `load_balance_proxy_server_example.py` (for example, `examples/disaggregated_prefill_v1/` in the matching vLLM-Ascend revision):

```shell
unset ftp_proxy
unset https_proxy
unset http_proxy

PREFILL_HOST="<P_HIGH_SPEED_SERVICE_IP>"
DECODE_HOST="<D_HIGH_SPEED_SERVICE_IP>"

python3 load_balance_proxy_server_example.py \
  --prefiller-hosts "${PREFILL_HOST}" \
  --prefiller-ports 30060 \
  --decoder-hosts "${DECODE_HOST}" \
  --decoder-ports 30050 \
  --host "${PREFILL_HOST}" \
  --port 8000
```

Send client requests to `http://<P_HIGH_SPEED_SERVICE_IP>:8000`. Before testing generation, check the P endpoint, D endpoint, and proxy health as described in [Section 6](#6-functional-verification).

#### 5.5.4 C8 Accuracy and Compatibility Requirements

C8 KV cache is part of this performance recipe, not just a model-name convention. Keep the following conditions together:

- Use the same compatible W4A8C8 checkpoint revision, tokenizer, chat template, quantization metadata, and `--kv-cache-dtype mxfp8` on P and D.
- Use `MooncakeConnectorV2` and the standard Pull proxy in [Section 5.7](#57-request-forwarding).
- Use a v0.30.0 MRv1 build containing [the MXFP8 KV-scale fix #19017](https://github.com/vllm-project/vllm-ascend/pull/19017), merged into `releases/v0.30.0` as `14c6c3f43`, or a build with an equivalent verified fix.
- Keep `VLLM_USE_V2_MODEL_RUNNER=0`, the P DP1/TP8 and D DP2/TP4 topology, and the connector topology consistent.
- Keep MTP settings identical on P and D. To run an MTP-off comparison, remove `--speculative-config` from both commands.
- Prefix caching and asynchronous scheduling are enabled on both nodes for this performance case.
- Prefill uses eager execution. Decode uses `FULL_DECODE_ONLY` and enables NPU graph extensions.
- `enable_fused_mc2=0` and `enable_prefill_mc2=false` reproduce this performance case. Do not infer the active setting from a profiling directory name.
- `enable_shared_expert_dp=true` and `enable_flashcomm1=true` require expert parallelism and the matching CANN/software stack.
- The D HTTP endpoint fronts both local DP engines. Do not use the DP RPC port as a Decode endpoint.

Use the standard Pull proxy with Prefill port 30060 and Decode port 30050. Start with short non-streaming requests, then verify request churn and the full accuracy baseline before performance testing. HTTP success or Mooncake transfer completion alone does not prove that C8 scale contents are correct. Check predictions for empty output, repeated punctuation, incoherent text, and abnormal repetition; evaluate MTP acceptance together with actual output correctness.

### 5.6 Prefill-Decode Disaggregation (A2)

PD disaggregation separates Prefill and Decode into different service groups. Prefill nodes process large prompt chunks, Decode nodes serve token generation, and a proxy forwards requests between them. These examples demonstrate how to tune prefill and decode resource ratios.

We recommend using Mooncake for deployment. Refer to [Mooncake](../features/pd_disaggregation_mooncake_multi_node.md) for the general PD disaggregation workflow.

For Atlas 800I A2 (64 GB x 8), we recommend deploying 1P1D with 2 nodes for `Qwen3.5-397B-A17B-w4a8-mtp`:

- 1 Prefill node: 1 Atlas 800I A2 (64 GB x 8).
- 1 Decode node: 1 Atlas 800I A2 (64 GB x 8).

Deploy `run_p.sh` and `run_d.sh` on the corresponding nodes, and deploy a proxy script on the prefill master node to forward requests. For details about how to deploy the agent script, see [Section 5.7](#57-request-forwarding).

#### 5.6.1 Prefill Node

Create `run_p.sh` on the prefill node.

```shell
#!/usr/bin/env bash
set -euo pipefail

unset ftp_proxy
unset https_proxy
unset http_proxy

export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000

# Get these values through ifconfig.
# nic_name is the network interface name corresponding to local_ip.
nic_name="<NETWORK_INTERFACE>"
local_ip="<PREFILL_NODE_IP>"
export IP_ADDRESS="${local_ip}"
export NETWORK_CARD_NAME="${nic_name}"

export HCCL_IF_IP="${local_ip}"
export GLOO_SOCKET_IFNAME="${nic_name}"
export HCCL_SOCKET_IFNAME="${nic_name}"

export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE=AIV
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_USE_FASTOKENS=1
export ASCEND_RT_VISIBLE_DEVICES="$1"
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2

vllm serve Eco-Tech/Qwen3.5-397B-A17B-w4a8-mtp \
  --host "${local_ip}" \
  --port "$2" \
  --data-parallel-size "$3" \
  --data-parallel-size-local 1 \
  --tensor-parallel-size "$7" \
  --enable-expert-parallel \
  --seed 1024 \
  --served-model-name qwen3.5 \
  --max-model-len 133120 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64 \
  --enforce-eager \
  --trust-remote-code \
  --allowed-local-media-path / \
  --gpu-memory-utilization 0.90 \
  --quantization ascend \
  --enable-prefix-caching \
  --async-scheduling \
  --mm-processor-cache-gb 0 \
  --mm-encoder-tp-mode data \
  --distributed-executor-backend mp \
  --no-disable-hybrid-kv-cache-manager \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}' \
  --additional-config '{"enable_flashcomm1":true,"ascend_compilation_config":{"enable_npugraph_ex":false,"enable_static_kernel":false}, "weight_nz_mode":1, "enable_shared_expert_dp":true,"scheduler_config":{"recompute_scheduler_enable":false}}' \
  --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_buffer_device":"npu","kv_role":"kv_producer","kv_port":"55220","engine_id":"0","kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":8},"decode":{"dp_size":1,"tp_size":8}}}'
```

#### 5.6.2 Decode Node

Create `run_d.sh` on the decode node.

```shell
#!/usr/bin/env bash
set -euo pipefail
unset ftp_proxy
unset https_proxy
unset http_proxy

export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000

# Get these values through ifconfig.
# nic_name is the network interface name corresponding to local_ip.
nic_name="<NETWORK_INTERFACE>"
local_ip="<DECODE_NODE_IP>"
export IP_ADDRESS="${local_ip}"
export NETWORK_CARD_NAME="${nic_name}"

export HCCL_IF_IP="${local_ip}"
export GLOO_SOCKET_IFNAME="${nic_name}"
export HCCL_SOCKET_IFNAME="${nic_name}"

export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE=AIV

export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_USE_FASTOKENS=1
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
export ASCEND_RT_VISIBLE_DEVICES="$1"
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2

vllm serve Eco-Tech/Qwen3.5-397B-A17B-w4a8-mtp \
  --host "${local_ip}" \
  --port "$2" \
  --data-parallel-size "$3" \
  --data-parallel-size-local 1 \
  --data-parallel-rank "$4" \
  --data-parallel-address "$5" \
  --data-parallel-rpc-port "$6" \
  --tensor-parallel-size "$7" \
  --enable-expert-parallel \
  --seed 1024 \
  --served-model-name qwen3.5 \
  --max-model-len 133120 \
  --max-num-batched-tokens 300 \
  --max-num-seqs 200 \
  --trust-remote-code \
  --allowed-local-media-path / \
  --gpu-memory-utilization 0.90 \
  --quantization ascend \
  --enable-prefix-caching \
  --async-scheduling \
  --mm-processor-cache-gb 0 \
  --mm-encoder-tp-mode data \
  --distributed-executor-backend mp \
  --no-disable-hybrid-kv-cache-manager \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false}, "weight_nz_mode":1, "enable_shared_expert_dp":false, "scheduler_config":{"recompute_scheduler_enable":true}}' \
  --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_buffer_device":"npu","kv_role":"kv_consumer","kv_port":"56010","engine_id":"2","kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":8},"decode":{"dp_size":1,"tp_size":8}}}'
```

Common Issues Tip: If the decode node fails to initialize, check that `--tensor-parallel-size` is 8, `--data-parallel-size-local` matches the global decode DP size (1), and `kv_connector_extra_config.decode.dp_size` matches the global decode DP size.

**Key parameters for PD disaggregation:**

- `VLLM_USE_V2_MODEL_RUNNER=0` selects Model Runner V1 on both nodes.
- `--distributed-executor-backend mp` uses multiprocessing on each node for the local workers.
- Prefill uses `--data-parallel-size 1` and `--tensor-parallel-size 8`. This creates 1 DP group with TP8.
- Decode uses `--data-parallel-size 1` and `--tensor-parallel-size 8`. This creates 1 DP group with TP8.
- `--data-parallel-address` and `--data-parallel-rpc-port` define the DP control plane.
- `--max-num-batched-tokens` is larger on the prefill node and smaller on the decode node because prefill is prompt-token intensive while decode is latency sensitive.
- `scheduler_config.recompute_scheduler_enable` sends requests back to the prefill side to recompute KV cache when decode KV cache is insufficient. Enable it only on the decode node in PD mode.
- `--kv-transfer-config` sets the Mooncake connector. `kv_role` is `kv_producer` on prefill and `kv_consumer` on decode.
- `kv_connector_extra_config.prefill.dp_size/tp_size` and `decode.dp_size/tp_size` must match the actual global DP and TP layout.
- `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'` is recommended on the decode node to reduce decode dispatch overhead.

Run the two scripts on their respective hosts after filling in each host's NIC and IP. The examples use seven argument positions: visible devices, HTTP port, global DP size, DP rank, DP master address, DP RPC port, and TP size. P reads only positions 1–3 and 7; positions 4–6 are unused and retained for invocation compatibility. D uses all seven positions; its argument 5 is the D host's service IP. Use DP1/TP8 to match the fixed KV topology; changing topology also requires updating the connector JSON and device allocation. Set the A2 proxy's `--prefiller-ports` and `--decoder-ports` in [Section 5.6.3](#563-proxy) to the HTTP ports passed as argument 2 to the respective scripts.

```shell
# On P:
bash run_p.sh 0,1,2,3,4,5,6,7 30060 1 0 "<P_SERVICE_IP>" 6884 8
# On D:
bash run_d.sh 0,1,2,3,4,5,6,7 30050 1 0 "<D_SERVICE_IP>" 6884 8
```

#### 5.6.3 Proxy

After both services report ready, start the standard Pull proxy on the Prefill host. Use the same service IPs configured as `local_ip` in the two scripts and the HTTP ports passed as argument 2.

Run the command from the directory containing `load_balance_proxy_server_example.py`:

```shell
unset ftp_proxy
unset https_proxy
unset http_proxy

PREFILL_HOST="<PREFILL_NODE_IP>"
DECODE_HOST="<DECODE_NODE_IP>"

python3 load_balance_proxy_server_example.py \
  --prefiller-hosts "${PREFILL_HOST}" \
  --prefiller-ports 30060 \
  --decoder-hosts "${DECODE_HOST}" \
  --decoder-ports 30050 \
  --host "${PREFILL_HOST}" \
  --port 8000
```

Send client requests to `http://<PREFILL_NODE_IP>:8000`. Use the Decode HTTP port `30050`, not the DP RPC port `6884`.

### 5.7 Request Forwarding

Use the proxy paired with the actual connector, rather than choosing it by hardware name. The standard proxy script is in `examples/disaggregated_prefill_v1/`; run the command from that directory. Replace the example hosts with reachable service IPs and use each service's HTTP port. P, D, and the client must use the same `--served-model-name`.

| Connector | Request Order | Proxy | Scope |
|-----------|---------------|-------|-------|
| `MooncakeConnectorV1` | Proxy calls P, then D; D pulls KV | `load_balance_proxy_server_example.py` | A3 and A2 PD recipes in [Section 5.4](#54-prefill-decode-disaggregation-a3) and [Section 5.6](#56-prefill-decode-disaggregation-a2) |
| `MooncakeConnectorV2` | Proxy calls P, then D; D pulls KV | `load_balance_proxy_server_example.py` | 950DT W4A8C8 unequal-TP recipe in [Section 5.5](#55-prefill-decode-disaggregation-950dt-products) |

Equal TP sizes do not make these connectors interchangeable. For unequal TP, verify the direction, divisibility, replicated KV heads, hybrid state layout, and quantization support of the selected implementation. The TP8-to-TP4 example does not imply arbitrary TP ratios or reverse-direction support. Use scripts from the same vLLM-Ascend revision as the serving stack.

Use the complete proxy command for the selected deployment:

- [A3 single-host 1P1D](#543-proxy): P and D use the same service IP and different HTTP ports.
- [950DT two-host 1P1D](#553-proxy): P and D use their respective high-speed service IPs; the D backend is HTTP port `30050`.
- [A2 two-host 1P1D](#563-proxy): P and D use their respective service IPs and the HTTP ports passed to the launcher scripts.

Do not use a DP RPC port or Mooncake KV-transfer port as a proxy backend. If requests reach the proxy but return no output, verify the P and D `/health` endpoints from the proxy host, confirm that the configured backend addresses exactly match each service's `--host` and HTTP port, and then run the end-to-end checks in [Section 6](#6-functional-verification).

## 6 Functional Verification

Wait for both P and D to report ready, verify each HTTP service, then start the matching proxy. For single-node and MP deployment, use the API endpoint on node 0. For PD disaggregation, send generation requests through the proxy so that the test covers KV transfer.

```shell
curl -f "http://<prefill_ip>:<prefill_http_port>/health"
curl -f "http://<decode_ip>:<decode_http_port>/health"
curl -f "http://<proxy_ip>:<proxy_port>/healthcheck"
curl -f "http://<prefill_ip>:<prefill_http_port>/v1/models"
```

Check that the advertised model name matches the request. A health check only establishes HTTP readiness; it does not verify the PD data path. Send several short requests, followed by concurrent requests that complete and reuse cache slots, and inspect P, D, and proxy logs.

```shell
curl "http://<server_ip>:<port>/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5",
    "prompt": "The future of AI is",
    "max_tokens": 50,
    "temperature": 0
  }'
```

Expected result: the HTTP status is 200 and the JSON response contains nonempty, coherent generated text in `choices`. Also verify a chat request with the checkpoint's chat template; a short completion request alone is not an accuracy test. Check for transfer errors, empty results, repeated punctuation, and abnormal repetition before running a full benchmark. Start with non-streaming requests, then repeat with streaming to test the proxy's chunked-response path.

### 6.1 Function Calling

To enable tool calling, append the following options to the `vllm serve` command. For PD deployment, add them to both the Prefill and Decode commands and restart both services.

```text
--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3
```

`--enable-auto-tool-choice` allows the model to select a tool, `--tool-call-parser qwen3_coder` parses tool calls, and `--reasoning-parser qwen3` separates reasoning from the final answer.

Send a chat request with a tool definition to the service endpoint. For PD deployment, use the proxy endpoint.

```shell
curl "http://<server_ip>:<port>/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5",
    "messages": [
      {"role": "user", "content": "Use get_weather to check the current weather in Beijing."}
    ],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"]
        }
      }
    }],
    "tool_choice": "auto",
    "max_tokens": 2048,
    "temperature": 0
  }'
```

When the model selects the tool, check that `choices[0].message.tool_calls` contains the function name `get_weather` and a JSON argument string with `city` set to `Beijing`. The client must execute the function, then send the assistant tool-call message and a `role: "tool"` result message with the matching `tool_call_id` in the next request to obtain the final answer. The serving process parses tool calls; it does not execute the function.

## 7 Accuracy Evaluation

Here are two accuracy evaluation methods.

### 7.1 Using AISBench

Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for installation and dataset preparation.

Configure `vllm_api_general_chat` with the service IP, HTTP port, and served model name (`qwen3.5`). For PD deployment, use the proxy endpoint. The following configuration is an example for GPQA Diamond:

```python
batch_size = 16
max_out_len = 65536
generation_kwargs = dict(
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    min_p=0.0,
    presence_penalty=0.0,
    repetition_penalty=1.0,
)
```

Set `max_out_len` so that the input length plus output budget does not exceed the service's `--max-model-len`. The A3 PD example in [Section 5.4](#54-prefill-decode-disaggregation-a3) is configured with `--max-model-len 133120`; keep the prompt length and output budget within that limit.

Run the evaluation:

```shell
ais_bench --models vllm_api_general_chat \
  --datasets gpqa_gen_0_shot_cot_chat_prompt --debug
```

The following result for `Qwen3.5-397B-A17B-w8a8` on `vLLM-Ascend v0.30.0 release` is provided for reference only.

| dataset | version | metric   | mode | vllm-api-general-chat |
| ------- | ------- | -------- | ---- | --------------------- |
| gpqa    | -       | accuracy | gen  | 88                    |

Record software and checkpoint versions, and keep the dataset, prompt, chat template, sampling parameters, output budget, and concurrency consistent when comparing results. For empty responses, truncated outputs, or unexpectedly low scores, see [FAQ Q7](#q7-why-is-the-benchmark-score-low-despite-apparently-normal-model-answers).

### 7.2 Using Language Model Evaluation Harness

Refer to [Using lm_eval](../../developer_guide/evaluation/using_lm_eval.md) for installation and usage details. For single-node or MP deployment, set `base_url` to the serving API endpoint; for PD deployment, use the proxy endpoint. Replace `127.0.0.1:8000` below with the IP and HTTP port reachable from the evaluation client, and ensure that `model` matches `--served-model-name`.

```shell
lm_eval \
  --model local-completions \
  --model_args model=qwen3.5,base_url=http://127.0.0.1:8000/v1/completions,tokenized_requests=False,trust_remote_code=True \
  --tasks gsm8k \
  --output_path ./
```

## 8 Performance Evaluation

### 8.1 Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### 8.2 Using vLLM Benchmark

Run performance evaluation of `Qwen3.5-397B-A17B-w8a8` as an example. Refer to [vLLM benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

There are three `vllm bench` subcommands:

- `latency`: benchmark the latency of a single batch of requests.
- `serve`: benchmark online serving throughput.
- `throughput`: benchmark offline inference throughput.

Take `serve` as an example:

```shell
export VLLM_USE_MODELSCOPE=True

vllm bench serve \
  --model Eco-Tech/Qwen3.5-397B-A17B-w8a8 \
  --served-model-name qwen3.5 \
  --dataset-name random \
  --random-input-len 200 \
  --num-prompts 200 \
  --request-rate 1 \
  --save-result \
  --result-dir ./
```

After several minutes, you can get the performance evaluation result.

## 9 Performance Tuning

### 9.1 Recommended Configurations

The table summarizes the deployment examples in [Section 5](#5-online-service-deployment). Follow the linked scripts for the complete configuration, and tune the values for your workload. NPUs are allocated to the listed role on each host; the A3 PD roles share one host with 8 NPUs each. DP is shown as global/local. `Max Num Seqs` and `Max Num Batched Tokens` apply to each DP group.

| Deployment Example | Node Role | NPUs | TP | DP (global/local) | Max Model Len | Max Num Seqs | Max Num Batched Tokens | MTP Tokens | Prefix Cache |
| ------------------ | --------- | ---- | -- | ----------------- | ------------- | ------------ | ---------------------- | ---------- | ------------ |
| [5.1 A3 single node](#51-single-node-online-deployment) | Mixed | 16 | 16 | 1/1 | 133000 | 128 | 16384 | 3 | On |
| [5.1 950DT single node](#51-single-node-online-deployment) | Mixed | 8 | 8 | 1/1 | 133000 | 128 | 8192 | 3 | On |
| [5.2 A2 MP](#52-multi-node-deployment-with-mp-recommended) | Mixed, 2 nodes | 8 | 8 | 2/1 | 32768 | 16 | 4096 | 3 | Off |
| [5.4 A3 PD](#54-prefill-decode-disaggregation-a3) | Prefill | 8 | 8 | 1/1 | 133120 | 64 | 20480 | 3 | On |
| [5.4 A3 PD](#54-prefill-decode-disaggregation-a3) | Decode | 8 | 4 | 2/2 | 133120 | 100 per DP | 300 per DP | 3 | On |
| [5.5 950DT W4A8C8 PD](#55-prefill-decode-disaggregation-950dt-products) | Prefill | 8 | 8 | 1/1 | 133120 | 64 | 20480 | 3 | On |
| [5.5 950DT W4A8C8 PD](#55-prefill-decode-disaggregation-950dt-products) | Decode | 8 | 4 | 2/2 | 133120 | 100 per DP | 300 per DP | 3 | On |
| [5.6 A2 PD](#56-prefill-decode-disaggregation-a2) | Prefill | 8 | 8 | 1/1 | 133120 | 64 | 16384 | 3 | On |
| [5.6 A2 PD](#56-prefill-decode-disaggregation-a2) | Decode | 8 | 8 | 1/1 | 133120 | 200 | 300 | 3 | On |

### 9.2 Tuning Guidelines

Refer to [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md) for general methods and the [Feature Matrix](../../user_guide/support_matrix/feature_matrix.md) for feature support.

1. Choose a [Section 5](#5-online-service-deployment) deployment example and fix the input/output lengths and test concurrency before comparing performance.
2. Set `--max-model-len` for the workload, then balance `--max-num-seqs` and `--gpu-memory-utilization` against KV cache capacity and runtime memory usage.
3. Tune `--max-num-batched-tokens` separately for Prefill and Decode. Compare throughput, TTFT, and TPOT using the methods in [Section 8](#8-performance-evaluation).
4. Compare MTP, ACLGraph, and the optimizations below one at a time, checking output correctness as well as performance.

### 9.3 Model-Specific Optimizations

Use the settings in the selected deployment script as a starting point. Merge the Ascend options below into its existing `--additional-config` JSON object.

| Optimization | Configuration | Usage |
| ------------ | ------------- | ----- |
| MTP | `--speculative-config` with `method=qwen3_5_mtp` | Use the token count in [Section 9.1](#91-recommended-configurations), tuning Prefill and Decode together for PD deployment. |
| Full decode ACLGraph | `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` | Used for single-node, MP, and PD Decode examples; PD Prefill uses eager mode. |
| Fused MC2 | `enable_fused_mc2` | `1`: dispatch-FFN-combine; `2`: CANN MegaMoe with matching operators. `0` disables fusion. The A3 PD performance case uses `2`; the 950DT W4A8C8 and A2 PD examples use `0` or do not enable it. |
| MoE sequence parallelism | `enable_flashcomm1=true` | Requires `--enable-expert-parallel` and TP greater than 1. |
| Shared expert data parallelism | `enable_shared_expert_dp=true` | Enabled on both P and D in [Section 5.4](#54-prefill-decode-disaggregation-a3) and [Section 5.5](#55-prefill-decode-disaggregation-950dt-products), and on Prefill in [Section 5.6](#56-prefill-decode-disaggregation-a2). Requires EP and TP greater than 1; increases weight memory usage. |
| Shared expert overlap | `multistream_overlap_shared_expert=true` | Used by the [MP example](#52-multi-node-deployment-with-mp-recommended). |
| Recompute scheduler | `scheduler_config.recompute_scheduler_enable=true` | Enabled on PD Decode, where `kv_role=kv_consumer`; keep it nested under `scheduler_config`. |
| CPU binding | `enable_cpu_binding=true` | Explicitly enabled in the single-node, MP, A3 PD, and 950DT W4A8C8 PD scripts. |

## 10 FAQ

For general installation and environment issues, refer to [Public FAQs](../../faqs.md).

### Q1: Why does the service report OOM during startup or soon after accepting requests?

Check that the weight precision and device allocation match [Section 3.1](#31-model-weight), and use `--quantization ascend` for the quantized examples. Reduce context length, concurrency, or batched tokens as needed, and leave runtime memory headroom when setting `--gpu-memory-utilization`. Follow the tuning order in [Section 9.2](#92-tuning-guidelines).

### Q2: Why does multi-node MP deployment hang during initialization?

Verify connectivity using [Section 3.2](#32-verify-multi-node-communication). Set `HCCL_IF_IP` to the selected NIC's IP and `HCCL_SOCKET_IFNAME`/`GLOO_SOCKET_IFNAME` to its interface name. Check the DP master address, matching RPC ports, non-overlapping rank ranges, and the worker node's `--headless` option against [Section 5.2](#52-multi-node-deployment-with-mp-recommended).

### Q3: Can I enable prefix caching in PD disaggregation?

The [A3 PD](#54-prefill-decode-disaggregation-a3), [950DT W4A8C8 PD](#55-prefill-decode-disaggregation-950dt-products), and [A2 PD](#56-prefill-decode-disaggregation-a2) examples enable prefix caching. When changing this setting, compare cached and uncached outputs, request completion, and cache reuse with the same connector and software version.

### Q4: Why does performance regress after enabling sequence parallelism or Fused MC2?

Use the hardware-specific settings in [Section 9.3](#93-model-specific-optimizations). Under the same workload, compare FlashComm and MC2 separately against their disabled settings (`enable_flashcomm1=false` and `enable_fused_mc2=0`). Preserve other required `--additional-config` fields and check output correctness before comparing performance. The A2 examples do not enable MC2.

### Q5: How should I tune MTP speculative decoding for this model?

Start with `num_speculative_tokens=3`, then compare TTFT, TPOT, and throughput when adjusting it. For PD deployment, keep `method` and `num_speculative_tokens` identical on Prefill and Decode; for an MTP-off comparison, remove `--speculative-config` from both sides. Change one setting at a time and check generated answers alongside acceptance statistics; acceptance rate alone does not establish accuracy.

### Q6: Why do C8 PD outputs contain repeated exclamation marks or lose accuracy?

Check that both nodes use the same C8 checkpoint revision and `--kv-cache-dtype mxfp8`, with the topology and connector settings in [Section 5.5](#55-prefill-decode-disaggregation-950dt-products) and the requirements in [Section 5.5.4](#554-c8-accuracy-and-compatibility-requirements). Inspect checkpoint-loading and KV-transfer logs. After correcting the environment or configuration, restart both services and verify short requests before running the full accuracy evaluation. Earlier prerelease K/V-scale issues were fixed by [#19017](https://github.com/vllm-project/vllm-ascend/pull/19017), included in v0.30.0.

### Q7: Why is the benchmark score low despite apparently normal model answers?

Inspect predictions for empty responses, truncated outputs, and answer-extraction failures. Correlate failed requests with P/D and proxy logs, including transfer timeouts and streaming errors. Report the full-dataset score and failure counts, and verify the dataset, chat template, sampling parameters, and output budget against [Section 7.1](#71-using-aisbench).

### Q8: Why do HIXL/Mooncake connections time out after the HTTP service starts?

HTTP readiness does not verify KV transfer. Check P/D service IPs, NICs, routes, physical NPU IDs, communication resource mounts, and KV ports using [Section 3.2](#32-verify-multi-node-communication) and the [Mooncake deployment guide](../features/pd_disaggregation_mooncake_multi_node.md). Ensure both nodes use matching runtime environments. After correcting the configuration, restart both services and send a request through the proxy as described in [Section 6](#6-functional-verification).
