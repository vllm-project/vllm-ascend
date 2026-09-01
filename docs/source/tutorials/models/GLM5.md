# GLM-5 & GLM-5.1

## 1 Introduction

This document applies to both `GLM-5` and `GLM-5.1`. Unless otherwise specified, all descriptions, configurations, and deployment procedures for `GLM-5` in this document also apply to `GLM-5.1`. For brevity, `GLM-5` is used hereafter as a unified reference to both `GLM-5` and `GLM-5.1`.

[GLM-5](https://huggingface.co/zai-org/GLM-5) uses a Mixture-of-Experts (MoE) architecture and targets complex systems engineering and long-horizon agentic tasks.

The `GLM-5` model is first supported in `vllm-ascend:v0.17.0rc1`(for Ascend950DT, the model is supported from `vllm-ascend:v0.23.0rc1`), and all **v0.17.0rc1 and later versions** can run stably. To use the latest features (e.g., PD separation, MTP), it is recommended to use the latest release candidate or official version. The version of transformers need to be upgraded to 5.2.0 or later versions.

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, single-node and multi-node deployment, accuracy and performance evaluation.

## 2 Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## 3 Prerequisites

### 3.1 Model Weight

- `GLM-5-w4a8`(Quantized version): [Download model weight](https://www.modelscope.cn/models/Eco-Tech/GLM-5-w4a8).
- `GLM-5-w8a8`(Quantized version): [Download model weight](https://www.modelscope.cn/models/Eco-Tech/GLM-5-w8a8).
- `GLM-5.1-w4a8`(Quantized version): [Download model weight](https://modelers.cn/models/Eco-Tech/GLM-5.1-w4a8).
- `GLM-5.1-w8a8`(Quantized version): [Download model weight](https://modelers.cn/models/Eco-Tech/GLM-5.1-w8a8).
- `GLM-5.1-w8a8c8`(Quantized version for Atlas 800 A3): [Download model weight](https://modelers.cn/models/Eco-Tech/GLM-5.1-w8a8c8-MTP).
- `GLM-5.1-w4a4`(Ascend950DT mxfp4 Quantized): [Download model weight](https://www.modelscope.cn/models/Eco-Tech/GLM-5.1-w4a4c8-mxfp4).

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`

### 3.2 Verify Multi-node Communication (Optional)

If multi-node deployment is required, please follow the [Verify Multi-node Communication Environment](../../installation.md#verify-multi-node-communication) guide for communication verification.

## 4 Installation

### 4.1 Docker Image Installation

You can use our official docker image to run GLM-5/5.1 directly.

:::::{tab-set}
:sync-group: install

::::{tab-item} Ascend950DT series
:sync: Ascend950DT

Start the docker image on each node.

```{code-block} bash
   :substitutions:

export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|-a5
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
-v /usr/bin/urma_admin:/usr/bin/urma_admin \
-v /lib/route.conf:/lib/route.conf \
-v /usr/lib64:/usr/lib64 \
-itd $IMAGE bash
```

::::

::::{tab-item} A3 series
:sync: A3

Start the docker image on each node.

```{code-block} bash
   :substitutions:

export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|-a3
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

::::
::::{tab-item} A2 series
:sync: A2

Start the docker image on each node.

```{code-block} bash
   :substitutions:

export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
docker run --rm \
    --name vllm-ascend \
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
    -v /root/.cache:/root/.cache \
    -it $IMAGE bash
```

If you want to deploy multi-node environment, you need to set up environment on each node.

To verify the successful installation of the environment, please refer to [installation](../../installation.md).

::::
:::::

### 4.2 Source Code Installation

In addition, if you don't want to use the docker image as above, you can also build all from source:

- Install `vllm-ascend` from source, refer to [installation](../../installation.md).

If you want to deploy multi-node environment, you need to set up environment on each node.

## 5 Online Service Deployment

### 5.1 Single-Node Online Deployment

:::::{tab-set}
:sync-group: install

::::{tab-item} Ascend950DT series
:sync: Ascend950DT

- Quantized model `glm-5-w4a4` can be deployed on 1 Ascend950DT (96GB × 8) .

Run the following script to execute online inference.

Common Issues Tip: If you encounter issues, Refer to [FAQs](../../faqs.md).

```shell
#!/usr/bin/env bash
source /root/.bashrc
export PROMETHEUS_MULTIPROC_DIR=/dev/shm/vllm_metrics && mkdir -p $PROMETHEUS_MULTIPROC_DIR
export HCCL_DFS_CONFIG="task_exception:off,inconsistent_check:off"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip of the current node
nic_name="xxx"
local_ip="xxx"

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100
export HCCL_BUFFSIZE=400
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1
export HCCL_INTRA_PCIE_ENABLE=1
export HCCL_INTRA_ROCE_ENABLE=0
export DYNAMIC_EPLB="true"

vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM5-w4a4 \
--host 0.0.0.0 \
--port 8077 \
--data-parallel-size 1 \
--tensor-parallel-size 8 \
--seed 1024 \
--served-model-name glm-5 \
--enable-expert-parallel \
--max-num-seqs 128 \
--max-model-len 202752 \
--max-num-batched-tokens 8192 \
--trust-remote-code \
--enable-prefix-caching \
--gpu-memory-utilization 0.95 \
--quantization ascend \
--enable-auto-tool-choice \
--tool-call-parser glm47 \
--reasoning-parser glm45 \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
--hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' \
--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}' \
--additional-config '{"enable_cpu_binding": "True", "multistream_overlap_shared_expert": "True", "enable_sparse_c8": "True", "enable_dsa_cp": true, "eplb_config": {"dynamic_eplb": true, "expert_heat_collection_interval": 50, "algorithm_execution_interval": 5, "eplb_policy_type": 2, "num_redundant_experts": 0 }}'
```

::::

::::{tab-item} A3 series
:sync: A3

- Quantized model `glm-5-w4a8` and `glm-5.1-w4a8` can be deployed on 1 Atlas 800 A3 (128GB × 8) .

Run the following script to execute online inference.

Common Issues Tip: If you encounter issues, Refer to [FAQs](../../faqs.md).

```{code-block} bash
   :substitutions:
# The version of transformers needs to be upgraded to 5.2.0.
# pip install transformers==5.2.0 --upgrade

export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_BALANCE_SCHEDULING=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM5-w4a8 \
--host 0.0.0.0 \
--port 8077 \
--data-parallel-size 1 \
--tensor-parallel-size 16 \
--enable-expert-parallel \
--seed 1024 \
--served-model-name glm-5 \
--max-num-seqs 16 \
--max-model-len 200000 \
--max-num-batched-tokens 4096 \
--trust-remote-code \
--gpu-memory-utilization 0.95 \
--quantization ascend \
--enable-chunked-prefill \
--enable-prefix-caching \
--additional-config '{"multistream_overlap_shared_expert": true}' \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
--hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' \
--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}'
```

::::
::::{tab-item} A2 series
:sync: A2

- Quantized model `glm-5-w4a8` can be deployed on 1 Atlas 800 A2 (64GB × 8) .

Run the following script to execute online inference.

Common Issues Tip: If you encounter issues, Refer to [FAQs](../../faqs.md).

```{code-block} bash
   :substitutions:
export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_BALANCE_SCHEDULING=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM5-w4a8 \
--host 0.0.0.0 \
--port 8077 \
--data-parallel-size 1 \
--tensor-parallel-size 8 \
--enable-expert-parallel \
--seed 1024 \
--served-model-name glm-5 \
--max-num-seqs 8 \
--max-model-len 32768 \
--max-num-batched-tokens 4096 \
--trust-remote-code \
--gpu-memory-utilization 0.95 \
--quantization ascend \
--enable-chunked-prefill \
--enable-prefix-caching \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
--additional-config '{"multistream_overlap_shared_expert": true}' \
--hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' \
--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}'
```

::::
:::::

Key Parameter Descriptions:

Only the key parameters specific to this model/scenario are described below. `max-model-len` and `max-num-seqs` need to be set according to the actual usage scenario.

**Model-specific parameters:**

- `--enable-expert-parallel`: Must be enabled for the MoE architecture of GLM-5.
- `--tensor-parallel-size 16` / `--tensor-parallel-size 8`: Tensor parallelism within each DP rank. For A3 (16 NPUs), use `tp16`; for A2 (8 NPUs), use `tp8`.
- `--quantization ascend`: Enables Ascend quantization for w4a8/w8a8 quantized weights.
- `--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}'`: Enables Multi-Token Prediction (MTP) speculative decoding with GLM-5's DeepSeek-style MTP draft model. `num_speculative_tokens` (3-5) controls how many tokens are speculated per step; `enforce_eager: true` is required because GLM-5 does not support graph-mode speculative decoding.
- `--enable-chunked-prefill` / `--enable-prefix-caching`: Recommended for long-context and multi-user scenarios — chunked prefill splits long prompts to improve TTFT, prefix caching reuses KV cache for shared prefixes (e.g., system prompts).
- `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'`: Enables graph capture for the decode phase only, improving decode performance by reducing kernel launch overhead.
- `--additional-config '{"multistream_overlap_shared_expert": true}'`: Overlaps shared-expert computation on an additional stream. Note: automatically disabled when `VLLM_ASCEND_ENABLE_FUSED_MC2=1`, as the two optimizations conflict.

**Key environment variables:**

- `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`: Enables FlashComm optimization to reduce communication overhead (mainly benefits the prefill path). With FlashComm enabled, `layer_sharding` cannot include `o_proj`.
- `VLLM_ASCEND_ENABLE_MLAPO=1`: Enables the MLA preprocess fusion operator (MlaPreprocessOperation). Enabled by default for w8a8 models — significantly improves Decode performance but consumes more NPU memory; set `VLLM_ASCEND_ENABLE_MLAPO=0` if memory is a priority. Recommended for w8a8; w4a8 may not benefit.
- `VLLM_ASCEND_BALANCE_SCHEDULING=1`: Enables balance scheduling to improve output throughput and reduce TPOT in the v1 scheduler.

**Performance tuning notes for single-node:**

- For low-latency scenarios, use `dp1tp16` (data-parallel-size 1, tensor-parallel-size 16) and consider reducing `--max-num-seqs` and `--max-num-batched-tokens`.
- For high-throughput scenarios, increase `--max-num-seqs` and enable `--enable-prefix-caching`.
- For long-context scenarios (e.g., 200K), use w4a8 weight (more memory for KV cache) and set `--max-model-len` to the desired context length. Consider enabling `--enable-chunked-prefill`.
- If you encounter OOM, reduce `--gpu-memory-utilization`, `--max-num-seqs`, or `--max-model-len`. Disabling `VLLM_ASCEND_ENABLE_MLAPO` can also reduce memory usage (at the cost of performance).

### 5.2 Multi-node Deployment

If you want to deploy multi-node environment, you need to verify multi-node communication according to [verify multi-node communication environment](../../installation.md#verify-multi-node-communication).

Common Issues Tip: If you encounter issues, Refer to [FAQs](../../faqs.md).

:::::{tab-set}
:sync-group: install

::::{tab-item} A3 series
:sync: A3

**High-Throughput Scenario (DP8 TP4)**

- `glm-5.1-w8a8c8`: can be deployed on 2 Atlas 800 A3 (128GB × 8) for high-throughput scenarios.

Run the following scripts on two nodes respectively.

**node 0**

```{code-block} bash
   :substitutions:
# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip of the current node
nic_name="xxx"
local_ip="xxx"
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_TRANSFER_TIMEOUT=600
export HCCL_EXEC_TIMEOUT=3600
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=400
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_MLAPO=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.1-W8A8C8-MTP \
--host 0.0.0.0 \
--port 8077 \
--data-parallel-size 8 \
--data-parallel-size-local 4 \
--data-parallel-address $local_ip \
--enable-expert-parallel \
--data-parallel-rpc-port 12980 \
--tensor-parallel-size 4 \
--hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' \
--seed 1024 \
--served-model-name glm-5 \
--tool-call-parser glm47 \
--reasoning-parser glm45 \
--enable-auto-tool-choice \
--trust-remote-code \
--gpu-memory-utilization 0.92 \
--quantization ascend \
--enable-chunked-prefill \
--enable-prefix-caching \
--async-scheduling \
--additional-config '{"enable_dsa_cp": true, "enable_sparse_sfa_c8": true, "enable_sparse_li_c8": true, "enable_balance_scheduling": true, "fuse_muls_add": true}' \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp","enforce_eager":true}'
```

**node 1**

```{code-block} bash
   :substitutions:
# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip of the current node
nic_name="xxx"
local_ip="xxx"
# IP of node 0 (the data parallel master node), must be consistent with the local_ip of node 0
node0_ip="xxxx"

export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_TRANSFER_TIMEOUT=600
export HCCL_EXEC_TIMEOUT=3600
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=400
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_MLAPO=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.1-W8A8C8-MTP \
--host 0.0.0.0 \
--port 8077 \
--headless \
--data-parallel-size 8 \
--data-parallel-size-local 4 \
--data-parallel-start-rank 4 \
--data-parallel-address $node0_ip \
--enable-expert-parallel \
--data-parallel-rpc-port 12980 \
--tensor-parallel-size 4 \
--hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' \
--seed 1024 \
--served-model-name glm-5 \
--tool-call-parser glm47 \
--reasoning-parser glm45 \
--enable-auto-tool-choice \
--trust-remote-code \
--gpu-memory-utilization 0.92 \
--quantization ascend \
--enable-chunked-prefill \
--enable-prefix-caching \
--async-scheduling \
--additional-config '{"enable_dsa_cp": true, "enable_sparse_sfa_c8": true, "enable_sparse_li_c8": true, "enable_balance_scheduling": true, "fuse_muls_add": true}' \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp","enforce_eager":true}'
```

**Notice:**

- When testing with a prefix cache hit rate > 0, keep `--enable-prefix-caching` (as in the scripts above); when the hit rate is 0, replace it with `--no-enable-prefix-caching`.
- `VLLM_ASCEND_ENABLE_FUSED_MC2=1` conflicts with `"multistream_overlap_shared_expert": true` — the runtime automatically disables `multistream_overlap_shared_expert` when fused MC2 is enabled.

::::
::::{tab-item} A2 series
:sync: A2

Run the following scripts on two nodes respectively.

**node 0**

```{code-block} bash
   :substitutions:
# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip of the current node
nic_name="xxx"
local_ip="xxx"

# The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
node0_ip="xxx"

export HCCL_OP_EXPANSION_MODE="AIV"

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=200
export VLLM_ASCEND_BALANCE_SCHEDULING=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM5-w4a8 \
--host 0.0.0.0 \
--port 8077 \
--data-parallel-size 2 \
--data-parallel-size-local 1 \
--data-parallel-address $node0_ip \
--data-parallel-rpc-port 13389 \
--tensor-parallel-size 8 \
--quantization ascend \
--seed 1024 \
--served-model-name glm-5 \
--enable-expert-parallel \
--max-num-seqs 2 \
--max-model-len 131072 \
--max-num-batched-tokens 4096 \
--trust-remote-code \
--gpu-memory-utilization 0.95 \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
--additional-config '{"multistream_overlap_shared_expert": true}' \
--hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' \
--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}'
```

**node 1**

```{code-block} bash
   :substitutions:
# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip of the current node
nic_name="xxx"
local_ip="xxx"

# The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
node0_ip="xxx"

export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=200
export VLLM_ASCEND_BALANCE_SCHEDULING=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM5-w4a8 \
--host 0.0.0.0 \
--port 8077 \
--headless \
--data-parallel-size 2 \
--data-parallel-size-local 1 \
--data-parallel-start-rank 1 \
--data-parallel-address $node0_ip \
--data-parallel-rpc-port 13389 \
--tensor-parallel-size 8 \
--quantization ascend \
--seed 1024 \
--served-model-name glm-5 \
--enable-expert-parallel \
--max-num-seqs 2 \
--max-model-len 131072 \
--max-num-batched-tokens 4096 \
--trust-remote-code \
--gpu-memory-utilization 0.95 \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
--additional-config '{"multistream_overlap_shared_expert": true}' \
--hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' \
--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}'
```

::::
:::::

Key Parameter Descriptions for multi-node deployment:

In addition to all single-node parameters described in [Single-Node Online Deployment](#51-single-node-online-deployment), the following parameters are specific to multi-node deployment:

**Network and data parallel configuration:**

- `HCCL_IF_IP`, `GLOO_SOCKET_IFNAME`, `TP_SOCKET_IFNAME`, `HCCL_SOCKET_IFNAME`: Network interface configuration for multi-node communication. Set `nic_name` to the network interface name (obtained via `ifconfig`) and `local_ip` to the current node's IP address. These must be correctly configured on each node for successful multi-node communication.
- `--data-parallel-size`: Total number of data parallel ranks across all nodes. For 2-node deployment, typically set to `2`.
- `--data-parallel-size-local`: Number of data parallel ranks on the current node. Usually set to `1` (one DP rank per node).
- `--data-parallel-address`: IP address of the data parallel master node (node 0). Must match the `local_ip` of the master node.
- `--data-parallel-rpc-port`: RPC port for data parallel master communication. Must be the same across all nodes.
- `--headless`: Indicates this is a non-master node. Do not use on node 0.
- `--data-parallel-start-rank`: Starting rank offset for data parallel ranks on this node. Node 0 uses `0`; node 1 uses the number of DP ranks on node 0 (e.g., `1` for DP2, `4` for DP8).

**Multi-node performance tuning:**

- For low-latency multi-node scenarios, keep `--data-parallel-size-local 1` to minimize cross-node communication.
- `--max-num-seqs` should be tuned based on available KV cache memory after model loading. For the w8a8c8 198K high-throughput scenario on A3, `6` is recommended; for the 198K low-latency scenario, `16` is recommended. For w4a8 on A2 multi-node with long context, start with `2` and increase if memory permits.
- All nodes in a multi-node deployment must use identical `--tensor-parallel-size`, `--enable-expert-parallel`, and model weight path configurations.

**w8a8c8-specific `--additional-config` fields:**

- `"enable_dsa_cp": true`: Enables DSA context parallelism to accelerate long-context prefill.
- `"enable_sparse_sfa_c8": true` / `"enable_sparse_li_c8": true`: Sparse attention optimizations of the C8 quantized model.
- `"enable_balance_scheduling": true`: Improves output throughput and reduces TPOT in the v1 scheduler. Not recommended when Prefill-Decode is separated.
- `"fuse_muls_add": true`: Fuses multiply-add operations.
- `"multistream_overlap_shared_expert": true`: Overlaps shared-expert computation on an additional stream. Automatically disabled when `VLLM_ASCEND_ENABLE_FUSED_MC2=1`.

(glm5-prefill-decode-disaggregation)=

### 5.3 Prefill-Decode Disaggregation

We'd like to show the deployment guide of `GLM-5` on multi-node environment with Prefill-Decode (PD) disaggregation for better performance. *Prefill-Decode Disaggregation* refers to the separation of the prefill stage and the decode stage across different nodes to improve throughput and latency.

In the PD disaggregation scenario, Mooncake is used as the KV cache transfer connector between the prefill and decode nodes. Please refer to [KV Cache Pool (Ascend Store) Deployment Guide](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/user_guide/feature_guide/kv_pool.md) for the Mooncake configuration.

#### 5.3.1 Prefill-Decode Disaggregation (Ascend950DT series)

Before you start, please

prepare the script `launch_online_dp.py` on each node:

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
        visible_devices = ",".join(str(x) for x in range(i * tp_size, (i + 1) * tp_size))
        process = multiprocessing.Process(target=run_command,
                                        args=(visible_devices, dp_rank,
                                                vllm_engine_port))
        processes.append(process)
        process.start()
    for process in processes:
        process.join()
```

1. prepare the script `run_dp_template.sh` on each node.

    1. Prefill node 0

        ```shell
        #!/usr/bin/env bash
        source /root/.bashrc
        export PROMETHEUS_MULTIPROC_DIR=/dev/shm/vllm_metrics && mkdir -p $PROMETHEUS_MULTIPROC_DIR
        export HCCL_DFS_CONFIG="task_exception:off,inconsistent_check:off"

        # this obtained through ifconfig
        # nic_name is the network interface name corresponding to local_ip of the current node
        nic_name="xxx"
        local_ip="xxx"

        export HCCL_IF_IP=$local_ip
        export GLOO_SOCKET_IFNAME=$nic_name
        export TP_SOCKET_IFNAME=$nic_name
        export HCCL_SOCKET_IFNAME=$nic_name
        export HCCL_ALGO=level0:fullmesh
        export VLLM_RPC_TIMEOUT=3600000
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_EXEC_TIMEOUT=204
        export HCCL_CONNECT_TIMEOUT=180
        export HCCL_BUFFSIZE=300
        export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=10
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
        export ASCEND_RT_VISIBLE_DEVICES=$1

        vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM5-w4a4 \
            --host 0.0.0.0 \
            --port $2 \
            --data-parallel-size $3 \
            --tensor-parallel-size $7 \
            --max-model-len 135000 \
            --max-num-batched-tokens 8192 \
            --served-model-name glm-5 \
            --gpu-memory-utilization 0.95 \
            --enable-expert-parallel \
            --max-num-seqs 8 \
            --enable-prefix-caching \
            --trust-remote-code \
            --enforce-eager \
            --quantization ascend \
            --enable-auto-tool-choice \
            --tool-call-parser glm47 \
            --reasoning-parser glm45 \
            --speculative-config '{"num_speculative_tokens": 1, "method": "deepseek_mtp", "enforce_eager": true}' \
            --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' \
            --kv-transfer-config \
            '{"kv_connector": "MooncakeConnectorV1",
            "kv_role": "kv_producer",
            "kv_port": "30100",
            "engine_id": "1",
            "kv_connector_extra_config": {
                        "prefill": {
                                "dp_size": 1,
                                "tp_size": 8
                        },
                        "decode": {
                                "dp_size": 16,
                                "tp_size": 1
                        },
                        "ascend_local_comm_res_path": "/etc/hixlep"
                }
            }' \
            --additional-config '{"enable_cpu_binding": "True", "multistream_overlap_shared_expert": "True", "recompute_scheduler_enable": "True", "enable_sparse_c8": "True", "enable_dsa_cp": true }'
        ```

    2. Prefill node 1

        ```shell
        #!/usr/bin/env bash
        source /root/.bashrc
        export PROMETHEUS_MULTIPROC_DIR=/dev/shm/vllm_metrics && mkdir -p $PROMETHEUS_MULTIPROC_DIR
        export HCCL_DFS_CONFIG="task_exception:off,inconsistent_check:off"

        # this obtained through ifconfig
        # nic_name is the network interface name corresponding to local_ip of the current node
        nic_name="xxx"
        local_ip="xxx"

        export HCCL_IF_IP=$local_ip
        export GLOO_SOCKET_IFNAME=$nic_name
        export TP_SOCKET_IFNAME=$nic_name
        export HCCL_SOCKET_IFNAME=$nic_name
        export HCCL_ALGO=level0:fullmesh
        export VLLM_RPC_TIMEOUT=3600000
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_EXEC_TIMEOUT=204
        export HCCL_CONNECT_TIMEOUT=180
        export HCCL_BUFFSIZE=300
        export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=10
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
        export ASCEND_RT_VISIBLE_DEVICES=$1

        vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM5-w4a4 \
            --host 0.0.0.0 \
            --port $2 \
            --data-parallel-size $3 \
            --tensor-parallel-size $7 \
            --max-model-len 135000 \
            --max-num-batched-tokens 8192 \
            --served-model-name glm-5 \
            --gpu-memory-utilization 0.95 \
            --enable-expert-parallel \
            --max-num-seqs 8 \
            --enable-prefix-caching \
            --trust-remote-code \
            --enforce-eager \
            --quantization ascend \
            --enable-auto-tool-choice \
            --tool-call-parser glm47 \
            --reasoning-parser glm45 \
            --speculative-config '{"num_speculative_tokens": 1, "method": "deepseek_mtp", "enforce_eager": true}' \
            --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' \
            --kv-transfer-config \
            '{"kv_connector": "MooncakeConnectorV1",
            "kv_role": "kv_producer",
            "kv_port": "30100",
            "engine_id": "1",
            "kv_connector_extra_config": {
                        "prefill": {
                                "dp_size": 1,
                                "tp_size": 8
                        },
                        "decode": {
                                "dp_size": 16,
                                "tp_size": 1
                        },
                        "ascend_local_comm_res_path": "/etc/hixlep"
                }
            }' \
            --additional-config '{"enable_cpu_binding": "True", "multistream_overlap_shared_expert": "True", "recompute_scheduler_enable": "True", "enable_sparse_c8": "True", "enable_dsa_cp": true }'
        ```

    3. Decode node 0

        ```shell
        #!/usr/bin/env bash
        source /root/.bashrc
        export PROMETHEUS_MULTIPROC_DIR=/dev/shm/vllm_metrics && mkdir -p $PROMETHEUS_MULTIPROC_DIR
        export HCCL_DFS_CONFIG="task_exception:off,inconsistent_check:off"

        # this obtained through ifconfig
        # nic_name is the network interface name corresponding to local_ip of the current node
        nic_name="xxx"
        local_ip="xxx"

        export HCCL_IF_IP=$local_ip
        export GLOO_SOCKET_IFNAME=$nic_name
        export TP_SOCKET_IFNAME=$nic_name
        export HCCL_SOCKET_IFNAME=$nic_name
        export HCCL_ALGO=level0:fullmesh
        export VLLM_RPC_TIMEOUT=3600000
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_EXEC_TIMEOUT=200
        export HCCL_CONNECT_TIMEOUT=1800
        export HCCL_BUFFSIZE=1200
        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=10
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
        export ASCEND_RT_VISIBLE_DEVICES=$1

        vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM5-w4a4 \
            --host 0.0.0.0 \
            --port $2 \
            --data-parallel-size $3 \
            --data-parallel-rank $4 \
            --data-parallel-address $5 \
            --data-parallel-rpc-port $6 \
            --tensor-parallel-size $7 \
            --max-model-len 135000 \
            --max-num-batched-tokens 240 \
            --served-model-name glm-5 \
            --gpu-memory-utilization 0.95 \
            --enable-expert-parallel \
            --max-num-seqs 60 \
            --enable-prefix-caching \
            --trust-remote-code \
            --quantization ascend \
            --enable-auto-tool-choice \
            --tool-call-parser glm47 \
            --reasoning-parser glm45 \
            --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
            --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}' \
            --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' \
            --kv-transfer-config \
            '{"kv_connector": "MooncakeConnectorV1",
            "kv_role": "kv_consumer",
            "kv_port": "30300",
            "engine_id": "3",
            "kv_connector_extra_config": {
                        "prefill": {
                                "dp_size": 1,
                                "tp_size": 8
                        },
                        "decode": {
                                "dp_size": 16,
                                "tp_size": 1
                        },
                        "ascend_local_comm_res_path": "/etc/hixlep"
                }
            }' \
            --additional-config '{"enable_cpu_binding": "True", "multistream_overlap_shared_expert": "True", "recompute_scheduler_enable": "True", "enable_sparse_c8": "True", "finegrained_tp_config": {"lmhead_tensor_parallel_size":8}}'
        ```

    4. Decode node 1

        ```shell
        #!/usr/bin/env bash
        source /root/.bashrc
        export PROMETHEUS_MULTIPROC_DIR=/dev/shm/vllm_metrics && mkdir -p $PROMETHEUS_MULTIPROC_DIR
        export HCCL_DFS_CONFIG="task_exception:off,inconsistent_check:off"

        # this obtained through ifconfig
        # nic_name is the network interface name corresponding to local_ip of the current node
        nic_name="xxx"
        local_ip="xxx"

        export HCCL_IF_IP=$local_ip
        export GLOO_SOCKET_IFNAME=$nic_name
        export TP_SOCKET_IFNAME=$nic_name
        export HCCL_SOCKET_IFNAME=$nic_name
        export HCCL_ALGO=level0:fullmesh
        export VLLM_RPC_TIMEOUT=3600000
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
        export HCCL_EXEC_TIMEOUT=200
        export HCCL_CONNECT_TIMEOUT=1800
        export HCCL_BUFFSIZE=1200
        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=10
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
        export ASCEND_RT_VISIBLE_DEVICES=$1

        vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM5-w4a4 \
            --host 0.0.0.0 \
            --port $2 \
            --data-parallel-size $3 \
            --data-parallel-rank $4 \
            --data-parallel-address $5 \
            --data-parallel-rpc-port $6 \
            --tensor-parallel-size $7 \
            --max-model-len 202752 \
            --max-num-batched-tokens 240 \
            --served-model-name glm-5 \
            --gpu-memory-utilization 0.95 \
            --enable-expert-parallel \
            --max-num-seqs 60 \
            --enable-prefix-caching \
            --trust-remote-code \
            --quantization ascend \
            --enable-auto-tool-choice \
            --tool-call-parser glm47 \
            --reasoning-parser glm45 \
            --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
            --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}' \
            --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' \
            --kv-transfer-config \
            '{"kv_connector": "MooncakeConnectorV1",
            "kv_role": "kv_consumer",
            "kv_port": "30300",
            "engine_id": "3",
            "kv_connector_extra_config": {
                        "prefill": {
                                "dp_size": 1,
                                "tp_size": 8
                        },
                        "decode": {
                                "dp_size": 16,
                                "tp_size": 1
                        },
                        "ascend_local_comm_res_path": "/etc/hixlep"
                }
            }' \
            --additional-config '{"enable_cpu_binding": "True", "multistream_overlap_shared_expert": "True", "recompute_scheduler_enable": "True", "enable_sparse_c8": "True", "finegrained_tp_config": {"lmhead_tensor_parallel_size":8}}'
        ```

Once the preparation is done, you can start the server with the following command on each node:

1. Prefill node 0

    ```shell
    # change ip to your own
    python launch_online_dp.py --dp-size 1 --tp-size 8 --dp-size-local 1 --dp-rank-start 0 --dp-address $node_p0_ip --dp-rpc-port 10521 --vllm-start-port 6700
    ```

2. Prefill node 1

    ```shell
    # change ip to your own
    python launch_online_dp.py --dp-size 1 --tp-size 8 --dp-size-local 1 --dp-rank-start 0 --dp-address $node_p1_ip --dp-rpc-port 10521 --vllm-start-port 6700
    ```

3. Decode node 0

    ```shell
    # change ip to your own
    python launch_online_dp.py --dp-size 16 --tp-size 1 --dp-size-local 8 --dp-rank-start 0 --dp-address $node_d0_ip --dp-rpc-port 10523 --vllm-start-port 6721
    ```

4. Decode node 1

    ```shell
    # change ip to your own
    python launch_online_dp.py --dp-size 16 --tp-size 1 --dp-size-local 8 --dp-rank-start 8 --dp-address $node_d0_ip --dp-rpc-port 10523 --vllm-start-port 6721
    ```

#### 5.3.2 Prefill-Decode Disaggregation (A3 series)

The high-throughput (198K context) scenario is validated on 4 Atlas 800 A3 (128GB × 8): 2 prefill nodes (`PP2 TP16`, 78 layers partitioned as `41/37`, one PP rank per node) and 2 decode nodes (`DP8 TP4`, 4 DP ranks per node). The same scripts serve both the high-throughput and low-latency cases.

Before you start, please

prepare the script `launch_online_dp.py` on each node:

```python
import argparse
import multiprocessing
import os
import subprocess
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp-size", type=int, required=True, help="Data parallel size.")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size.")
    parser.add_argument("--pp-size", type=int, default=1, help="Pipeline parallel size.")
    parser.add_argument("--dp-size-local", type=int, default=-1, help="Local data parallel size.")
    parser.add_argument("--dp-rank-start", type=int, default=0, help="Starting rank for data parallel.")
    parser.add_argument("--dp-address", type=str, required=True, help="IP address for data parallel master node.")
    parser.add_argument("--dp-rpc-port", type=str, default="12321", help="Port for data parallel master node.")
    parser.add_argument("--vllm-start-port", type=int, default=8000, help="Starting port for the engine.")
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
# 一个 DP 副本占 tp×pp 张连续卡（pp=1 时退化为现状的 tp）
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
    num_cards = dp_size_local * gpus_per_dp_rank

    for i in range(dp_size_local):
        dp_rank = dp_rank_start + i
        vllm_engine_port = vllm_start_port + i
        visible_devices = ",".join(str(x) for x in range(i * gpus_per_dp_rank, (i + 1) * gpus_per_dp_rank))
        process = multiprocessing.Process(
            target=run_command,
            args=(visible_devices, dp_rank, vllm_engine_port)
        )
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
```

1. prepare the script `run_dp_template.sh` on each node.

    1. Prefill node 0

        The prefill script selects the node via `node_rank`: set `node_rank=0` on prefill node 0 (PP master node, engine port `9081`) and `node_rank=1` on prefill node 1 (non-master node, `--headless`, no API server).

        ```shell
        nic_name="xxxx" # change to your own nic name
        local_ip="xxxx" # change to your own ip
        # pp=2
        export VLLM_PP_LAYER_PARTITION="41,37"
        # prefill node 0: node_rank=0, prefill node 1: node_rank=1
        node_rank=0

        export HCCL_OP_EXPANSION_MODE="AIV"
        export HCCL_TRANSFER_TIMEOUT=600
        export HCCL_EXEC_TIMEOUT=3600
        export HCCL_CONNECT_TIMEOUT=3600

        export HCCL_IF_IP=$local_ip
        export GLOO_SOCKET_IFNAME=$nic_name
        export TP_SOCKET_IFNAME=$nic_name
        export HCCL_SOCKET_IFNAME=$nic_name

        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=1
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export HCCL_BUFFSIZE=400

        export ACL_OP_INIT_MODE=1
        export ASCEND_A3_ENABLE=1

        export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
        export VLLM_ASCEND_ENABLE_FUSED_MC2=1
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

        vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.1-W8A8C8-MTP \
            --host 0.0.0.0 \
            --port 9081 \
            --pipeline-parallel-size 2 \
            --distributed-executor-backend mp \
            --master-addr $local_ip \
            --master-port 7060 \
            --nnodes 2 \
            --node-rank $node_rank \
            --tensor-parallel-size 16 \
            --enable-expert-parallel \
            --speculative-config '{"num_speculative_tokens": 3, "method":"deepseek_mtp","enforce_eager":true}' \
            --seed 1024 \
            --served-model-name glm-5 \
            --max-model-len 202752 \
            --additional-config '{"fuse_muls_add":true, "recompute_scheduler_enable" : false, "multistream_overlap_shared_expert": true, "enable_dsa_cp":true, "enable_sparse_sfa_c8": true, "enable_sparse_li_c8": true, "c8_enable_reshape_optim":true}' \
            --max-num-batched-tokens 16384 \
            --trust-remote-code \
            --enable-prefix-caching \
            --max-num-seqs 64 \
            --quantization ascend \
            --gpu-memory-utilization 0.92 \
            --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' \
            --enforce-eager \
            --enable-auto-tool-choice \
            --tool-call-parser glm47 \
            --reasoning-parser glm45 \
            --kv-transfer-config \
            '{"kv_connector": "MooncakeConnectorV1",
            "kv_role": "kv_producer",
            "kv_port": "30000",
            "engine_id": "0",
            "kv_connector_extra_config": {
                "use_ascend_direct": true,
                "prefill": {"dp_size": 1, "pp_size": 2, "tp_size": 16, "pp_layer_partition": "41,37"},
                "decode": {"dp_size": 8, "tp_size": 4}
            }
        }'
        ```

    2. Prefill node 1

        ```shell
        nic_name="xxxx" # change to your own nic name
        local_ip="xxxx" # change to your own ip
        # IP of prefill node 0 (the PP master node), must be consistent with the local_ip of prefill node 0
        node_p0_ip="xxxx"
        # pp=2
        export VLLM_PP_LAYER_PARTITION="41,37"
        # prefill node 0: node_rank=0, prefill node 1: node_rank=1
        node_rank=1

        export HCCL_OP_EXPANSION_MODE="AIV"
        export HCCL_TRANSFER_TIMEOUT=600
        export HCCL_EXEC_TIMEOUT=3600
        export HCCL_CONNECT_TIMEOUT=3600

        export HCCL_IF_IP=$local_ip
        export GLOO_SOCKET_IFNAME=$nic_name
        export TP_SOCKET_IFNAME=$nic_name
        export HCCL_SOCKET_IFNAME=$nic_name

        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=1
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export HCCL_BUFFSIZE=400

        export ACL_OP_INIT_MODE=1
        export ASCEND_A3_ENABLE=1

        export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
        export VLLM_ASCEND_ENABLE_FUSED_MC2=1
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

        vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.1-W8A8C8-MTP \
            --host 0.0.0.0 \
            --pipeline-parallel-size 2 \
            --distributed-executor-backend mp \
            --master-addr $node_p0_ip \
            --master-port 7060 \
            --nnodes 2 \
            --node-rank $node_rank \
            --headless \
            --tensor-parallel-size 16 \
            --enable-expert-parallel \
            --speculative-config '{"num_speculative_tokens": 3, "method":"deepseek_mtp","enforce_eager":true}' \
            --seed 1024 \
            --served-model-name glm-5 \
            --max-model-len 202752 \
            --additional-config '{"fuse_muls_add":true, "recompute_scheduler_enable" : false, "multistream_overlap_shared_expert": true, "enable_dsa_cp":true, "enable_sparse_sfa_c8": true, "enable_sparse_li_c8": true, "c8_enable_reshape_optim":true}' \
            --max-num-batched-tokens 16384 \
            --trust-remote-code \
            --enable-prefix-caching \
            --max-num-seqs 64 \
            --quantization ascend \
            --gpu-memory-utilization 0.92 \
            --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' \
            --enforce-eager \
            --enable-auto-tool-choice \
            --tool-call-parser glm47 \
            --reasoning-parser glm45 \
            --kv-transfer-config \
            '{"kv_connector": "MooncakeConnectorV1",
            "kv_role": "kv_producer",
            "kv_port": "30000",
            "engine_id": "0",
            "kv_connector_extra_config": {
                "use_ascend_direct": true,
                "prefill": {"dp_size": 1, "pp_size": 2, "tp_size": 16, "pp_layer_partition": "41,37"},
                "decode": {"dp_size": 8, "tp_size": 4}
            }
        }'
        ```

    3. Decode node 0 (ranks 0–3)

        Launch one instance per DP rank via positional parameters: `$1` = visible devices, `$2` = engine port, `$3` = data-parallel-size, `$4` = data-parallel-rank, `$5` = data-parallel-address, `$6` = data-parallel-rpc-port, `$7` = tensor-parallel-size. Prepare `run_dp_template.sh` on decode node 0 with the content below.

        ```shell
        nic_name="xxxx" # change to your own nic name
        local_ip="xxxx" # change to your own ip

        # 每个 DP rank 使用独立的 engine_id,避免 KV 路由混淆
        # $4 = data-parallel-rank; 节点内 rank 偏移 0-3 → engine_id 100-103
        ENGINE_ID=$((100 + $4))

        export HCCL_OP_EXPANSION_MODE="AIV"
        export HCCL_TRANSFER_TIMEOUT=600
        export HCCL_EXEC_TIMEOUT=3600
        export HCCL_CONNECT_TIMEOUT=3600

        export HCCL_IF_IP=$local_ip
        export GLOO_SOCKET_IFNAME=$nic_name
        export TP_SOCKET_IFNAME=$nic_name
        export HCCL_SOCKET_IFNAME=$nic_name

        #Mooncake
        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=1

        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export HCCL_BUFFSIZE=256

        export ACL_OP_INIT_MODE=1
        export ASCEND_A3_ENABLE=1
        export TASK_QUEUE_ENABLE=1
        export ASCEND_RT_VISIBLE_DEVICES=$1

        export VLLM_ASCEND_ENABLE_FUSED_MC2=1
        export VLLM_ASCEND_ENABLE_MLAPO=1
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

        vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.1-W8A8C8-MTP \
            --host 0.0.0.0 \
            --port $2 \
            --data-parallel-size $3 \
            --data-parallel-rank $4 \
            --data-parallel-address $5 \
            --data-parallel-rpc-port $6 \
            --tensor-parallel-size $7 \
            --enable-expert-parallel \
            --speculative-config '{"num_speculative_tokens": 3,  "method":"deepseek_mtp","enforce_eager":true}' \
            --seed 1024 \
            --served-model-name glm-5 \
            --max-model-len 202752 \
            --max-num-batched-tokens 164 \
            --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
            --additional-config '{"fuse_muls_add":true, "recompute_scheduler_enable":true, "multistream_overlap_shared_expert":true, "enable_sparse_sfa_c8": true, "enable_sparse_li_c8": true}' \
            --trust-remote-code \
            --max-num-seqs 32 \
            --gpu-memory-utilization 0.92 \
            --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' \
            --async-scheduling \
            --quantization ascend \
            --enable-auto-tool-choice \
            --tool-call-parser glm47 \
            --reasoning-parser glm45 \
            --kv-transfer-config \
            '{"kv_connector": "MooncakeConnectorV1",
            "kv_role": "kv_consumer",
            "kv_port": "30200",
            "engine_id": "'"$ENGINE_ID"'",
            "kv_connector_extra_config": {
                "use_ascend_direct": true,
                "prefill": {"dp_size": 1, "pp_size": 2, "tp_size": 16, "pp_layer_partition": "41,37"},
                "decode": {"dp_size": 8, "tp_size": 4}
            }
        }'
        ```

    4. Decode node 1 (ranks 4–7)

        Prepare `run_dp_template.sh` on decode node 1 with the content below.

        ```shell
        nic_name="xxxx" # change to your own nic name
        local_ip="xxxx" # change to your own ip

        # 每个 DP rank 使用独立的 engine_id,避免 KV 路由混淆
        # $4 = data-parallel-rank; 节点内 rank 偏移 0-3 → engine_id 100-103
        ENGINE_ID=$((100 + $4))

        export HCCL_OP_EXPANSION_MODE="AIV"
        export HCCL_TRANSFER_TIMEOUT=600
        export HCCL_EXEC_TIMEOUT=3600
        export HCCL_CONNECT_TIMEOUT=3600

        export HCCL_IF_IP=$local_ip
        export GLOO_SOCKET_IFNAME=$nic_name
        export TP_SOCKET_IFNAME=$nic_name
        export HCCL_SOCKET_IFNAME=$nic_name

        #Mooncake
        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=1

        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export HCCL_BUFFSIZE=256

        export ACL_OP_INIT_MODE=1
        export ASCEND_A3_ENABLE=1
        export TASK_QUEUE_ENABLE=1
        export ASCEND_RT_VISIBLE_DEVICES=$1

        export VLLM_ASCEND_ENABLE_FUSED_MC2=1
        export VLLM_ASCEND_ENABLE_MLAPO=1
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

        vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.1-W8A8C8-MTP \
            --host 0.0.0.0 \
            --port $2 \
            --data-parallel-size $3 \
            --data-parallel-rank $4 \
            --data-parallel-address $5 \
            --data-parallel-rpc-port $6 \
            --tensor-parallel-size $7 \
            --enable-expert-parallel \
            --speculative-config '{"num_speculative_tokens": 3,  "method":"deepseek_mtp","enforce_eager":true}' \
            --seed 1024 \
            --served-model-name glm-5 \
            --max-model-len 202752 \
            --max-num-batched-tokens 164 \
            --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
            --additional-config '{"fuse_muls_add":true, "recompute_scheduler_enable":true, "multistream_overlap_shared_expert":true, "enable_sparse_sfa_c8": true, "enable_sparse_li_c8": true}' \
            --trust-remote-code \
            --max-num-seqs 32 \
            --gpu-memory-utilization 0.92 \
            --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' \
            --async-scheduling \
            --quantization ascend \
            --enable-auto-tool-choice \
            --tool-call-parser glm47 \
            --reasoning-parser glm45 \
            --kv-transfer-config \
            '{"kv_connector": "MooncakeConnectorV1",
            "kv_role": "kv_consumer",
            "kv_port": "30200",
            "engine_id": "'"$ENGINE_ID"'",
            "kv_connector_extra_config": {
                "use_ascend_direct": true,
                "prefill": {"dp_size": 1, "pp_size": 2, "tp_size": 16, "pp_layer_partition": "41,37"},
                "decode": {"dp_size": 8, "tp_size": 4}
            }
        }'
        ```

Once the preparation is done, you can start the server with the following command on each node:

1. Prefill node 0

    ```shell
    bash run_dp_template.sh
    ```

2. Prefill node 1

    ```shell
    bash run_dp_template.sh
    ```

3. Decode node 0

    ```shell
    # change ip to your own
    python launch_online_dp.py --dp-size 8 --tp-size 4 --pp-size 1 --dp-size-local 4 --dp-rank-start 0 --dp-address $node_d0_ip --dp-rpc-port 12321 --vllm-start-port 8000
    ```

4. Decode node 1

    ```shell
    # change ip to your own
    python launch_online_dp.py --dp-size 8 --tp-size 4 --pp-size 1 --dp-size-local 4 --dp-rank-start 4 --dp-address $node_d0_ip --dp-rpc-port 12321 --vllm-start-port 8000
    ```

**Notice:**

- When testing with a prefix cache hit rate > 0, add `--enable-prefix-caching` on the prefill nodes (as in the scripts above); when the hit rate is 0, use `--no-enable-prefix-caching` instead.
- `"recompute_scheduler_enable"` is set to `false` on prefill nodes and `true` on decode nodes in this scenario.

#### 5.3.3 Request Forwarding and Key Parameter Descriptions

To set up request forwarding, run the following script on any machine. You can get the proxy program in the repository's examples: [load_balance_proxy_server_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py)

**Ascend950DT series:**

```shell
unset http_proxy
unset https_proxy

python load_balance_proxy_server_example.py \
    --port 8000 \
    --host 0.0.0.0 \
    --prefiller-hosts \
    $node_p0_ip \
    $node_p1_ip \
    --prefiller-ports \
    6700 \
    6700 \
    --decoder-hosts \
    $node_d0_ip \
    $node_d0_ip \
    $node_d0_ip \
    $node_d0_ip \
    $node_d0_ip \
    $node_d0_ip \
    $node_d0_ip \
    $node_d0_ip \
    $node_d1_ip \
    $node_d1_ip \
    $node_d1_ip \
    $node_d1_ip \
    $node_d1_ip \
    $node_d1_ip \
    $node_d1_ip \
    $node_d1_ip \
    --decoder-ports \
    6721 6722 6723 6724 6725 6726 6727 6728 \
    6721 6722 6723 6724 6725 6726 6727 6728
```

**A3 series:**

```shell
unset http_proxy
unset https_proxy

python load_balance_proxy_server_example.py \
    --port 9000 \
    --host 0.0.0.0 \
    --prefiller-hosts \
    $node_p0_ip \
    --prefiller-ports \
    9081 \
    --decoder-hosts \
    $node_d0_ip \
    $node_d0_ip \
    $node_d0_ip \
    $node_d0_ip \
    $node_d1_ip \
    $node_d1_ip \
    $node_d1_ip \
    $node_d1_ip \
    --decoder-ports \
    8000 8001 8002 8003 \
    8000 8001 8002 8003
```

Key Parameter Descriptions for PD separation deployment:

In addition to the single-node and multi-node parameters described above, the following parameters are specific to Prefill-Decode disaggregation:

**Mooncake KV transfer configuration (`--kv-transfer-config`):**

- `"kv_connector": "MooncakeConnectorV1"`: Uses Mooncake as the KV cache transfer connector between prefill and decode nodes.
- `"kv_role": "kv_producer"`: Set on prefill nodes — produces KV cache and sends it to decode nodes. Use `"kv_consumer"` on decode nodes.
- `"kv_port"`: Port for Mooncake KV transfer communication. Each node group should use a distinct port range.
- `"use_ascend_direct": true`: Enables Ascend direct (RDMA-like) transfer for KV cache, reducing latency.
- `"prefill"` / `"decode"` sections: Specify the `dp_size` and `tp_size` of the prefill and decode node groups respectively. These must match the actual deployment topology.

**Prefill node-specific configurations:**

- `VLLM_ASCEND_ENABLE_FUSED_MC2=1`: Enables fused MC2 operators (`dispatch_ffn_combine`/`mega_moe`) to optimize MoE communication. Constraints: `dispatch_ffn_combine` only for w8a8 and EP≤32; `mega_moe` works for w8a8/w4a8/bf16 with EP≤64. Both are incompatible with MTP and dynamic EPLB.
- `--additional-config '{"enable_dsa_cp": true}'`: Enables DSA context parallelism on prefill nodes to accelerate long-context prefill. Required for handling prompts up to 128K tokens.

**Decode node-specific configurations:**

- `VLLM_ASCEND_ENABLE_MLAPO=1`: Enables MLA preprocess operation fusion on decode nodes to significantly improve decode performance. Consumes more NPU memory. In PD scenarios, enable MLAPO only on decode nodes.
- On decode nodes, keep `--max-num-batched-tokens` close to `--max-num-seqs` — decode processes one token per sequence per step (`164` in the A3 scenario, `240` in the Ascend950DT scenario, see the scripts above).
- `--additional-config '{"recompute_scheduler_enable": true}'`: Enables the recomputation scheduler. When decode node KV cache is insufficient, requests are sent back to prefill nodes for KV cache recomputation. In this deployment: `true` on decode nodes; on prefill nodes `true` in the Ascend950DT scenario and `false` in the A3 scenario (see the scripts above).

**Common PD environment variables:**

- `ASCEND_AGGREGATE_ENABLE=1`, `ASCEND_A3_ENABLE=1`: A3-specific optimizations for communication aggregation.
- `ACL_OP_INIT_MODE=1`: ACL operator initialization mode.
- `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=480`: Timeout (in seconds) for automatically releasing the prefill node's KV cache when a request is aborted.
- `LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib`: Required for Mooncake library loading.

**MTP in PD scenarios:**

- Prefill nodes use `"num_speculative_tokens": 3` in the A3 scenario and `1` in the Ascend950DT scenario (see the scripts above).
- Decode nodes use `"num_speculative_tokens": 3` in both scenarios to maximize decode throughput.
- All prefill and decode nodes must use the same `"method": "deepseek_mtp"` and `"enforce_eager": true`.

For further explanation and restrictions of the environment variables above, refer to: [envs.py](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/envs.py).

## 6 Functional Verification

Once your server is started, you can query the model with input prompts:

```shell
curl http://<node0_ip>:<port>/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "glm-5",
        "prompt": "The future of AI is",
        "max_completion_tokens": 15,
        "temperature": 0
    }'
```

Expected Result:

```shell
{"id": "chatcmlib-bc44ad093dec79a2", "object": "chat.completion", "created": "1770903266", "model": "glm-5", "choices": [{ "index": 0, "message": {"role": "assistant", "content": "The future of AI is not one thing, but a convergence of several powerful trends.", "annotations": "null", "audio": "null", "function_call": "null", "tool_calls": [], "reasoning": "null"}, "logprobs": "null", "finish_reason": "length", "stop_reason": "null", "token_ids": null}], "service_tier": "null", "system fingerprint": "null", "usage": {"prompt_tokens": 5, "total_tokens": 20, "completion_tokens": 15, "prompt_tokens_details": null}, "prompt_logprobs": "null", "prompt_token_ids": "null", "kv_transfer_params": null}
```

## 7 Accuracy Evaluation

### 7.1 Using AISBench

1. Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details.

2. After execution, you can get the result.

## 8 Performance Evaluation

### 8.1 Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### 8.2 Using vLLM Benchmark

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

## 9 Performance Tuning

### 9.1 Recommended Configurations

> **Note**: The following configurations are validated in specific test environments and are for reference only. The optimal configuration depends on factors such as maximum input/output length, prefix cache hit rate, precision requirements, and deployment machine ratios. It is recommended to refer to [Tuning Guidelines](#92-tuning-guidelines) for tuning based on actual conditions.

The tables below provide recommended parameter configurations for the `GLM-5.1-w8a8c8` quantized model on Atlas 800 A3, covering three deployment scenarios:

- **Dual-Node Co-Located 198K High Throughput**: `DP8 TP4`, see [Multi-node Deployment](#52-multi-node-deployment).
- **Dual-Node Co-Located 198K Low Latency**: `DP2 TP16`, see [Multi-node Deployment](#52-multi-node-deployment).
- **Prefill-Decode Disaggregation 198K (PP2)**: prefill `PP2 TP16` + decode `DP8 TP4`, see [Prefill-Decode Disaggregation (A3 series)](#532-prefill-decode-disaggregation-a3-series).
- **Prefill-Decode Disaggregation 198K (Ascend950DT)**: prefill `TP8` (DSA CP 8) + decode `DP16 TP1`, see [Prefill-Decode Disaggregation (Ascend950DT series)](#531-prefill-decode-disaggregation-ascend950dt-series).

Test cases use the notation `input/output`, e.g., `128k/1k` means 128K input tokens and 1K output tokens; `@50%/90%` marks the prefix cache hit rate. When the prefix cache hit rate is > 0, add `--enable-prefix-caching`; when the hit rate is 0, add `--no-enable-prefix-caching` instead (for the PD scenario, this applies to the prefill nodes).

#### 9.1.1 Table 1: Detailed Node Configuration

> The TP/DP columns show the values **per node** as configured in the Deployment scripts (a co-located node hosting 4 DP ranks of TP4 uses 16 NPUs; a PD prefill node hosting 1 DP rank of TP16 uses 16 NPUs; a PD decode node hosting 4 DP ranks of TP4 uses 16 NPUs; an Ascend950DT node hosting 8 cards). The 198K PD scenario prefill side uses `PP2 TP16` with the layer partition `41,37`. All A3 scenarios use the `GLM-5.1-w8a8c8` weights; the Ascend950DT scenarios use the `GLM-5.1-w4a4` weights.
>
> When testing with a prefix cache hit rate > 0, keep `--enable-prefix-caching` (as in the deployment scripts); when the hit rate is 0, replace it with `--no-enable-prefix-caching`.

|Scenario|Weight Version|Configuration|NPUs|TP|DP|Max Num Seqs|Max Num Batched Tokens|Max Model Len|MTP Spec Num|
|--------|--------------|-------------|-----|--|--|------------|----------------------|--------------|-------------|
|Dual-Node Co-Located 198K High Throughput (A3)|w8a8c8|Dual-Node Co-Located Node (0/1)|16|4|4|6|4096|202752|3|
|Dual-Node Co-Located 198K Low Latency (A3)|w8a8c8|Dual-Node Co-Located Node (0/1)|16|16|1|16|4096|202752|3|
|PD 198K High Throughput (A3)|w8a8c8|PD — Server-P Node (PP2)|16|16|1|64|16384|202752|3|
|PD 198K High Throughput (A3)|w8a8c8|PD — Server-D Node|16|4|4|32|164|202752|3|
|PD 198K High Throughput (Ascend950DT)|w4a4|PD — Server-P Node (DSA CP 8)|8|8|1|20|8192|202752|3|
|PD 198K High Throughput (Ascend950DT)|w4a4|PD — Server-D Node|8|1|8|60|240|202752|3|
|PD 198K Low Latency (Ascend950DT)|w4a4|PD — Server-P Node (DSA CP 8)|8|8|1|20|8192|202752|3|
|PD 198K Low Latency (Ascend950DT)|w4a4|PD — Server-D Node|8|1|8|60|240|202752|3|

#### 9.1.2 Table 2: Optimizations Requiring Explicit Enablement

The following optimizations must be explicitly enabled to take effect. They apply to the A3 series (w8a8c8) as indicated:

|Optimization|Scenario|Enablement|Principle (Benefits)|Notes|
|------------|--------|----------|---------------------|-----|
|FlashComm_v1|A3 prefill nodes / co-located nodes|`export VLLM_ASCEND_ENABLE_FLASHCOMM1=1`|Splits AllReduce into Reduce-Scatter and All-Gather, improving prefill throughput and reducing communication latency|Not available when `layer_sharding` includes `o_proj`|
|Fused MC2|A3 prefill nodes|`export VLLM_ASCEND_ENABLE_FUSED_MC2=1`|Replaces ALLTOALL+MC2 with the `dispatch_ffn_combine`/`dispatch_gmm_combine_decode` operators, reducing MoE communication overhead and improving MoE inference performance|`dispatch_ffn_combine` only for w8a8, EP≤32, non-MTP, non-dynamic-EPLB; conflicts with `multistream_overlap_shared_expert` (the latter is auto-disabled)|
|MLAPO|A3 co-located high-throughput / PD decode nodes|`export VLLM_ASCEND_ENABLE_MLAPO=1`|Fuses the MLA preprocess operations, significantly improving decode performance|Consumes more NPU memory; in PD scenarios enable on decode nodes only|
|DSA CP|A3 prefill nodes; long context (≥128K)|`--additional-config '{"enable_dsa_cp": true}'`|DSA context parallelism accelerates long-context prefill, reducing TTFT for long prompts|In the reference configs, enabled on co-located nodes and PD prefill nodes|
|Balance Scheduling|A3 single-node / co-located / non-PD scenarios|`--additional-config '{"enable_balance_scheduling": true}'`|Improves output throughput and reduces TPOT in the v1 scheduler|TTFT may degrade; not recommended when Prefill-Decode is separated|
|Sparse SFA C8|A3 (w8a8c8); long-context prefill|`--additional-config '{"enable_sparse_sfa_c8": true}'`|Sparse Flash Attention skips unnecessary attention computation of the C8 quantized model, accelerating long-context prefill|Experimental in v0.23.0. In the reference configs, enabled in the high-throughput and PD scenarios; disabled in the low-latency scenario|
|Sparse LI C8|A3 (w8a8c8)|`--additional-config '{"enable_sparse_li_c8": true}'`|Sparse attention optimization reduces computation of the C8 quantized model, improving throughput|Independent of `enable_sparse_sfa_c8`; the reference low-latency config disables both|
|Recompute Scheduler|A3 decode nodes|`--additional-config '{"recompute_scheduler_enable": true}'`|Recomputes KV cache on prefill nodes when decode KV cache is insufficient, avoiding decode-side OOM and improving throughput|Set to `false` on prefill nodes|
|Multistream Overlap Shared Expert|A3|`--additional-config '{"multistream_overlap_shared_expert": true}'`|Overlaps shared-expert computation on an additional stream, hiding its latency and improving decode performance|Auto-disabled when `VLLM_ASCEND_ENABLE_FUSED_MC2=1`|

> For complete startup commands and detailed parameter descriptions, please refer to the deployment examples and Key Parameter Descriptions in [Online Service Deployment](#5-online-service-deployment).

#### 9.1.3 Table 3: Performance-Related Parameter Tuning Guide

|Parameter|Low Latency|High Throughput|Long Context|Description|
|---------|-----------|---------------|-------------|-----------|
|`--max-num-seqs`|Lower (16)|Higher (6–64)|Higher (32–64)|Limits concurrent sequences. Lower values reduce scheduling latency; higher values increase throughput.|
|`--max-model-len`|198K|Longer (128K–198K)|Maximum (198K)|Maximum context length. Must accommodate your longest input+output. Larger values consume more KV cache memory.|
|`--max-num-batched-tokens`|Lower (4096)|Higher for prefill (4096–16384)|Higher for prefill (16384); small on decode nodes (close to `max-num-seqs`)|Controls batch size per step. Lower values reduce per-step latency; higher values improve prefill throughput.|
|`--gpu-memory-utilization`|0.92|0.92|0.92|NPU memory fraction. The reference configs in this document use 0.92. Reduce if OOM.|
|`--enable-chunked-prefill`|Enable (co-located)|Enable (co-located)|Enable (co-located)|Splits long prompts into chunks to prevent prefill from blocking decode. PD prefill nodes use `--enforce-eager` instead.|
|`num_speculative_tokens` (MTP)|3|3|3 (prefill and decode)|MTP speculation count. Higher values improve decode throughput at the cost of memory for the draft model KV cache. In the reference PD configs, prefill nodes use `3` in the A3 scenario and `1` in the Ascend950DT scenario; decode nodes use `3` in both scenarios.|
|`cudagraph_mode`|FULL_DECODE_ONLY|FULL_DECODE_ONLY (co-located / decode nodes)|FULL_DECODE_ONLY (co-located / decode nodes)|Graph capture for the decode phase only. PD prefill nodes use `--enforce-eager` instead.|

### 9.2 Tuning Guidelines

For general performance tuning methods, refer to the [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md).

For detailed feature descriptions and configuration options, refer to the [Feature Guide](../../user_guide/support_matrix/feature_matrix.md).

For environment variable descriptions and constraints, refer to [envs.py](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/envs.py).

## 10 FAQ

- Common Issues Tip: If you encounter issues, Refer to [FAQs](../../faqs.md).

- **Q: How to solve ValueError: Tokenizer class TokenizersBackend does not exist or is not currently imported?**

  A: Please update the version of transformers to 5.2.0

- **Q: How to enable function calling for GLM-5?**

  A: Please add following configurations in vLLM startup command

  ```shell
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --enable-auto-tool-choice \
  ```
