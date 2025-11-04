# DeepSeek-V3.2-Exp

## Introduction

DeepSeek-V3.2-Exp is a sparse attention model. The main architecture is similar to DeepSeek-V3.1, but with a sparse attention mechanism, which is designed to explore and validate optimizations for training and inference efficiency in long-context scenarios.

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, single-node and multi-node deployment, accuracy and performance evaluation.

The `DeepSeek-V3.2-Exp` model is first supported in `vllm-ascend:v0.11.0rc0`.

## Supported Features

Refer to [supported features](../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../user_guide/feature_guide/index.md) to get the feature's configuration.

## Environment Preparation

### Model Weight

- `DeepSeek-V3.2-Exp`(BF16 version): require 2 Atlas 800 A3 (64G × 16) nodes or 4 Atlas 800 A2 (64G × 8) nodes. [Download model weight](https://modelers.cn/models/Modelers_Park/DeepSeek-V3.2-Exp-BF16)
- `DeepSeek-V3.2-Exp-w8a8`(Quantized version): require 1 Atlas 800 A3 (64G × 16) node or 2 Atlas 800 A2 (64G × 8) nodes. [Download model weight](https://modelers.cn/models/Modelers_Park/DeepSeek-V3.2-Exp-w8a8)

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`

### Verify Multi-node Communication(Optional)

If you want to deploy multi-node environment, you need to verify multi-node communication according to [verify multi-node communication environment](../installation.md#verify-multi-node-communication).

### Installation

:::::{tab-set}
::::{tab-item} Use deepseek-v3.2 docker image

Currently, we provide the all-in-one images `quay.io/ascend/vllm-ascend:v0.11.0rc0-deepseek-v3.2-exp`(for Atlas 800 A2) and `quay.io/ascend/vllm-ascend:v0.11.0rc0-a3-deepseek-v3.2-exp`(for Atlas 800 A3).

Refer to [using docker](../installation.md#set-up-using-docker) to set up environment using Docker, remember to replace the image with deepseek-v3.2 docker image.

:::{note}
The image is based on a specific version and will not continue to release new version.
Only AArch64 architecture are supported currently due to extra operator's installation limitations.
:::

::::
::::{tab-item} Use vllm-ascend docker image

You can using our official docker image and install extra operator for supporting `DeepSeek-V3.2-Exp`.

:::{note}
Only AArch64 architecture are supported currently due to extra operator's installation limitations.
:::

For `A3` image:

1. Start the docker image on your node, refer to [using docker](../installation.md#set-up-using-docker).

2. Install the package `custom-ops` to make the kernels available.

```shell
wget https://vllm-ascend.obs.cn-north-4.myhuaweicloud.com/vllm-ascend/a3/CANN-custom_ops-sfa-linux.aarch64.run
chmod +x ./CANN-custom_ops-sfa-linux.aarch64.run
./CANN-custom_ops-sfa-linux.aarch64.run --quiet
export ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize:${ASCEND_CUSTOM_OPP_PATH}
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib/:${LD_LIBRARY_PATH}
wget https://vllm-ascend.obs.cn-north-4.myhuaweicloud.com/vllm-ascend/a3/custom_ops-1.0-cp311-cp311-linux_aarch64.whl
pip install custom_ops-1.0-cp311-cp311-linux_aarch64.whl
```

3. Download and install `MLAPO`.

```shell
wget https://vllm-ascend.obs.cn-north-4.myhuaweicloud.com/vllm-ascend/a3/CANN-custom_ops-mlapo-linux.aarch64.run
# please set a custom install-path, here take `/`vllm-workspace/CANN` as example.
chmod +x ./CANN-custom_ops-mlapo-linux.aarch64.run 
./CANN-custom_ops-mlapo-linux.aarch64.run --quiet --install-path=/vllm-workspace/CANN
wget https://vllm-ascend.obs.cn-north-4.myhuaweicloud.com/vllm-ascend/a3/torch_npu-2.7.1%2Bgitb7c90d0-cp311-cp311-linux_aarch64.whl
pip install torch_npu-2.7.1+gitb7c90d0-cp311-cp311-linux_aarch64.whl
wget https://vllm-ascend.obs.cn-north-4.myhuaweicloud.com/vllm-ascend/a3/libopsproto_rt2.0.so
cp libopsproto_rt2.0.so /usr/local/Ascend/ascend-toolkit/8.2.RC1/opp/built-in/op_proto/lib/linux/aarch64/libopsproto_rt2.0.so
# Don't forget to replace `/vllm-workspace/CANN/` to the custom path you set before.
source /vllm-workspace/CANN/vendors/customize/bin/set_env.bash
export LD_PRELOAD=/vllm-workspace/CANN/vendors/customize/op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so:${LD_PRELOAD}
```

For `A2` image, you should change all `wget` commands as above, and replace `A3` with `A2` release file.

1. Start the docker image on your node, refer to [using docker](../installation.md#set-up-using-docker).

2. Install the package `custom-ops` to make the kernels available.

```shell
wget https://vllm-ascend.obs.cn-north-4.myhuaweicloud.com/vllm-ascend/a2/CANN-custom_ops-sfa-linux.aarch64.run
chmod +x ./CANN-custom_ops-sfa-linux.aarch64.run
./CANN-custom_ops-sfa-linux.aarch64.run --quiet
export ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize:${ASCEND_CUSTOM_OPP_PATH}
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib/:${LD_LIBRARY_PATH}
wget https://vllm-ascend.obs.cn-north-4.myhuaweicloud.com/vllm-ascend/a2/custom_ops-1.0-cp311-cp311-linux_aarch64.whl
pip install custom_ops-1.0-cp311-cp311-linux_aarch64.whl
```

3. Download and install `MLAPO`.

```shell
wget https://vllm-ascend.obs.cn-north-4.myhuaweicloud.com/vllm-ascend/a2/CANN-custom_ops-mlapo-linux.aarch64.run
# please set a custom install-path, here take `/`vllm-workspace/CANN` as example.
chmod +x ./CANN-custom_ops-mlapo-linux.aarch64.run 
./CANN-custom_ops-mlapo-linux.aarch64.run --quiet --install-path=/vllm-workspace/CANN
wget https://vllm-ascend.obs.cn-north-4.myhuaweicloud.com/vllm-ascend/a2/torch_npu-2.7.1%2Bgitb7c90d0-cp311-cp311-linux_aarch64.whl
pip install torch_npu-2.7.1+gitb7c90d0-cp311-cp311-linux_aarch64.whl
wget https://vllm-ascend.obs.cn-north-4.myhuaweicloud.com/vllm-ascend/a2/libopsproto_rt2.0.so
cp libopsproto_rt2.0.so /usr/local/Ascend/ascend-toolkit/8.2.RC1/opp/built-in/op_proto/lib/linux/aarch64/libopsproto_rt2.0.so
# Don't forget to replace `/vllm-workspace/CANN/` to the custom path you set before.
source /vllm-workspace/CANN/vendors/customize/bin/set_env.bash
export LD_PRELOAD=/vllm-workspace/CANN/vendors/customize/op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so:${LD_PRELOAD}
```

::::
::::{tab-item} Build from source

You can build all from source.

- Install `vllm-ascend`, refer to [set up using python](../installation.md#set-up-using-python).
- Install extra operator for supporting `DeepSeek-V3.2-Exp`, refer to `Use vllm-ascend docker image` tab.

::::
:::::

If you want to deploy multi-node environment, you need to set up environment on each node.

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

Not supported yet.

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

Here are two accuracy evaluation methods.

### Using AISBench

1. Refer to [Using AISBench](../developer_guide/evaluation/using_ais_bench.md) for details.

2. After execution, you can get the result, here is the result of `DeepSeek-V3.2-Exp-W8A8` in `vllm-ascend:0.11.0rc0` for reference only.

| dataset | version | metric | mode | vllm-api-general-chat |
|----- | ----- | ----- | ----- | -----|
| cevaldataset | - | accuracy | gen | 92.20 |

### Using Language Model Evaluation Harness

As an example, take the `gsm8k` dataset as a test dataset, and run accuracy evaluation of `DeepSeek-V3.2-Exp-W8A8` in online mode.

1. Refer to [Using lm_eval](../developer_guide/evaluation/using_lm_eval.md) for `lm_eval` installation.

2. Run `lm_eval` to execute the accuracy evaluation.

```shell
lm_eval \
  --model local-completions \
  --model_args model=/root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V3.2-Exp-W8A8,base_url=http://127.0.0.1:8000/v1/completions,tokenized_requests=False,trust_remote_code=True \
  --tasks gsm8k \
  --output_path ./
```

3. After execution, you can get the result, here is the result of `DeepSeek-V3.2-Exp-W8A8` in `vllm-ascend:0.11.0rc0` for reference only.

|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9591|±  |0.0055|
|gsm8k|      3|strict-match    |     5|exact_match|↑  |0.9583|±  |0.0055|

## Performance

### Using AISBench

Refer to [Using AISBench for performance evaluation](../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### Using vLLM Benchmark

Run performance evaluation of `DeepSeek-V3.2-Exp-W8A8` as an example.

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/contributing/benchmarks.html) for more details.

There are three `vllm bench` subcommand:
- `latency`: Benchmark the latency of a single batch of requests.
- `serve`: Benchmark the online serving throughput.
- `throughput`: Benchmark offline inference throughput.

Take the `serve` as an example. Run the code as follows.

```shell
export VLLM_USE_MODELSCOPE=true
vllm bench serve --model vllm-ascend/DeepSeek-V3.2-Exp-W8A8  --dataset-name random --random-input 200 --num-prompt 200 --request-rate 1 --save-result --result-dir ./
```

After about several minutes, you can get the performance evaluation result.
