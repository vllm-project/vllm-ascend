# DeepSeek-V3.2-Exp

## Introduction

DeepSeek-V3.2-Exp is a sparse attention model. The main architecture is similar to DeepSeek-V3.1, but with a sparse attention mechanism, which is designed to explore and validate optimizations for training and inference efficiency in long-context scenarios.

:::{note}
Only machines with AArch64 are supported currently. x86 will be supported soon. This guide takes A3 as the example.
:::

## Supported Features

Refer to [supported models](../user_guide/support_matrix/supported_models.md) to get the model's detail.

Refer to [supported features](../user_guide/support_matrix/supported_features.md) to get the supported features.

## Environment

### Model Weight

- `DeepSeek-V3.2-Exp`: require 2 Atlas 800 A3 (64G × 16) nodes or 4 Atlas 800 A2 (64G × 8). [Model weight link](https://modelers.cn/models/Modelers_Park/DeepSeek-V3.2-Exp-BF16)
- `DeepSeek-V3.2-Exp-w8a8`: require 1 Atlas 800 A3 (64G × 16) node or 2 Atlas 800 A2 (64G × 8). [Model weight link](https://modelers.cn/models/Modelers_Park/DeepSeek-V3.2-Exp-w8a8)

### Verify Multi-node Communication(Optional)

If you want to deploy multi-node environment, you need to verify multi-node communication according to [verify multi-node communication environment](../installation.md#verify-multi-node-communication).

### Installation

Currently, we provide the all-in-one images `quay.io/ascend/vllm-ascend:v0.11.0rc0-deepseek-v3.2-exp`(for Atlas 800 A2) and `quay.io/ascend/vllm-ascend:v0.11.0rc0-a3-deepseek-v3.2-exp`(for Atlas 800 A3). These images include CANN 8.2RC1 + [SparseFlashAttention/LightningIndexer](https://gitcode.com/cann/cann-recipes-infer/tree/master/ops/ascendc) + [MLAPO](https://github.com/vllm-project/vllm-ascend/pull/3226). You can also build your own image by referring to [link](https://github.com/vllm-project/vllm-ascend/issues/3278).

Refer to [installation](../installation.md#set-up-using-docker) to set up environment using Docker.

If you want to deploy multi-node environment, you need to set up envrionment on each node.

## Deployment

### Single-node Deployment

Only the quantized model `DeepSeek-V3.2-Exp-w8a8` can be deployed on 1 Atlas 800 A3.

Run the following script to execute online inference.

```shell
#!/bin/sh
export VLLM_USE_MODELSCOPE=true

vllm serve vllm-ascend/DeepSeek-V3.2-Exp-W8A8 \
--host 0.0.0.0 \
--port 8000 \
--tensor-parallel-size 16 \
--seed 1024 \
--quantization ascend \
--served-model-name deepseek_v3.2 \
--max-num-seqs 16 \
--max-model-len 17450 \
--max-num-batched-tokens 17450 \
--enable-expert-parallel \
--trust-remote-code \
--no-enable-prefix-caching \
--gpu-memory-utilization 0.92 \
--additional-config '{"ascend_scheduler_config":{"enabled":true},"torchair_graph_config":{"enabled":true,"graph_batch_sizes":[16]}}'
```

### Multi-node Deployment

- `DeepSeek-V3.2-Exp`: require 2 Atlas 800 A3 (64G × 16) nodes or 4 Atlas 800 A2 (64G × 8).
- `DeepSeek-V3.2-Exp-w8a8`: require 2 Atlas 800 A2 (64G × 8).

:::::{tab-set}
::::{tab-item} DeepSeek-V3.2-Exp A3 series

Run the following scripts on two nodes respectively.

:::{note}
Before launching the inference server, ensure the following environment variables are set for multi-node communication.
:::

**Node 0**

```shell
#!/bin/sh

# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip of the current node
nic_name="xxxx"
local_ip="xxxx"

export VLLM_USE_MODELSCOPE=True
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100
export HCCL_BUFFSIZE=1024

vllm serve /root/.cache/Modelers_Park/DeepSeek-V3.2-Exp \
--host 0.0.0.0 \
--port 8000 \
--data-parallel-size 2 \
--data-parallel-size-local 1 \
--data-parallel-address $local_ip \
--data-parallel-rpc-port 13389 \
--tensor-parallel-size 16 \
--seed 1024 \
--served-model-name deepseek_v3.2 \
--enable-expert-parallel \
--max-num-seqs 16 \
--max-model-len 17450 \
--max-num-batched-tokens 17450 \
--trust-remote-code \
--no-enable-prefix-caching \
--gpu-memory-utilization 0.9 \
--additional-config '{"ascend_scheduler_config":{"enabled":true},"torchair_graph_config":{"enabled":true,"graph_batch_sizes":[16]}}'
```

**Node 1**

```shell
#!/bin/sh

# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip of the current node
nic_name="xxx"
local_ip="xxx"

# The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
node0_ip="xxxx"

export VLLM_USE_MODELSCOPE=True
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100
export HCCL_BUFFSIZE=1024

vllm serve /root/.cache/Modelers_Park/DeepSeek-V3.2-Exp \
--host 0.0.0.0 \
--port 8000 \
--headless \
--data-parallel-size 2 \
--data-parallel-size-local 1 \
--data-parallel-start-rank 1 \
--data-parallel-address $node0_ip \
--data-parallel-rpc-port 13389 \
--tensor-parallel-size 16 \
--seed 1024 \
--served-model-name deepseek_v3.2 \
--max-num-seqs 16 \
--max-model-len 17450 \
--max-num-batched-tokens 17450 \
--enable-expert-parallel \
--trust-remote-code \
--no-enable-prefix-caching \
--gpu-memory-utilization 0.92 \
--additional-config '{"ascend_scheduler_config":{"enabled":true},"torchair_graph_config":{"enabled":true,"graph_batch_sizes":[16]}}'
```

::::
::::{tab-item} DeepSeek-V3.2-Exp-W8A8 A2 series

Run the following scripts on two nodes respectively.

**Node 0**

```shell
#!/bin/sh

# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip of the current node
nic_name="xxxx"
local_ip="xxxx"

export VLLM_USE_MODELSCOPE=True
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100
export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE="AIV"
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"

vllm serve vllm-ascend/DeepSeek-V3.2-Exp-W8A8 \
--host 0.0.0.0 \
--port 8000 \
--data-parallel-size 2 \
--data-parallel-size-local 1 \
--data-parallel-address $local_ip \
--data-parallel-rpc-port 13389 \
--tensor-parallel-size 8 \
--seed 1024 \
--served-model-name deepseek_v3.2 \
--enable-expert-parallel \
--max-num-seqs 16 \
--max-model-len 17450 \
--max-num-batched-tokens 17450 \
--trust-remote-code \
--quantization ascend \
--no-enable-prefix-caching \
--gpu-memory-utilization 0.9 \
--additional-config '{"ascend_scheduler_config":{"enabled":true},"torchair_graph_config":{"enabled":true,"graph_batch_sizes":[16]}}'
```

**Node 1**

```shell
#!/bin/sh

# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip of the current node
nic_name="xxx"
local_ip="xxx"

# The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
node0_ip="xxxx"

export VLLM_USE_MODELSCOPE=True
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100
export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE="AIV"
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"

vllm serve vllm-ascend/DeepSeek-V3.2-Exp-W8A8 \
--host 0.0.0.0 \
--port 8000 \
--headless \
--data-parallel-size 2 \
--data-parallel-size-local 1 \
--data-parallel-start-rank 1 \
--data-parallel-address $node0_ip \
--data-parallel-rpc-port 13389 \
--tensor-parallel-size 8 \
--seed 1024 \
--served-model-name deepseek_v3.2 \
--max-num-seqs 16 \
--max-model-len 17450 \
--max-num-batched-tokens 17450 \
--enable-expert-parallel \
--trust-remote-code \
--quantization ascend \
--no-enable-prefix-caching \
--gpu-memory-utilization 0.92 \
--additional-config '{"ascend_scheduler_config":{"enabled":true},"torchair_graph_config":{"enabled":true,"graph_batch_sizes":[16]}}'
```

::::
:::::

### Prefill-Decode Disaggregation

TODO

## Functional Verification

Once your server is started, you can query the model with input prompts:

```shell
curl http://<node0_ip>:<port>/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "deepseek_v3.2",
        "prompt": "The future of AI is",
        "max_tokens": 50,
        "temperature": 0
    }'
```

## Accuracy Evaluation

TODO

## Performance

TODO
