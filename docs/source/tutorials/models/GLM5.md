# GLM-5 & GLM-5.1

## 1 Introduction

This document applies to both `GLM-5` and `GLM-5.1`. Unless otherwise specified, all descriptions, configurations, and deployment procedures for `GLM-5` in this document also apply to `GLM-5.1`. For brevity, `GLM-5` is used hereafter as a unified reference to both `GLM-5` and `GLM-5.1`.

[GLM-5](https://huggingface.co/zai-org/GLM-5) uses a Mixture-of-Experts (MoE) architecture and targets complex systems engineering and long-horizon agentic tasks.

The `GLM-5` model is first supported in `vllm-ascend:v0.17.0rc1`, and all **v0.17.0rc1 and later versions** can run stably. To use the latest features (e.g., PD separation, MTP), it is recommended to use the latest release candidate or official version. The version of transformers need to be upgraded to 5.2.0 or later versions.

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, single-node and multi-node deployment, accuracy and performance evaluation.

## 2 Supported Features

Refer to [Supported Features List](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [Feature Guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## 3 Prerequisites

### 3.1 Model Weight

- `GLM-5`(BF16 version): [Download model weight](https://www.modelscope.cn/models/ZhipuAI/GLM-5).
- `GLM-5-w4a8`(Quantized version): [Download model weight](https://www.modelscope.cn/models/Eco-Tech/GLM-5-w4a8).
- `GLM-5-w8a8`(Quantized version): [Download model weight](https://www.modelscope.cn/models/Eco-Tech/GLM-5-w8a8).
- `GLM-5.1`(BF16 version): [Download model weight](https://huggingface.co/zai-org/GLM-5.1).
- `GLM-5.1-w4a8`(Quantized version): [Download model weight](https://modelers.cn/models/Eco-Tech/GLM-5.1-w4a8).
- `GLM-5.1-w8a8`(Quantized version): [Download model weight](https://modelers.cn/models/Eco-Tech/GLM-5.1-w8a8).

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`

### 3.2 Verify Multi-node Communication (Optional)

If multi-node deployment is required, please follow the [Verify Multi-node Communication Environment](../../installation.md#verify-multi-node-communication) guide for communication verification.

## 4 Installation

### 4.1 Docker Image Installation

You can use our official docker image to run GLM-5/5.1 directly.

=== "A3 series"

    Start the docker image on each node.

    ```shell

    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a3
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

=== "A2 series"

    Start the docker image on each node.

    ```shell

    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
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

### 4.2 Source Code Installation

In addition, if you don't want to use the docker image as above, you can also build all from source:

- Install `vllm-ascend` from source, refer to [installation](../../installation.md).

If you want to deploy multi-node environment, you need to set up environment on each node.

## 5 Online Service Deployment {: #5-online-service-deployment }

### 5.1 Single-Node Online Deployment

=== "A3 series"

    - Quantized model `glm-5-w4a8` and `glm-5.1-w4a8` can be deployed on 1 Atlas 800 A3 (64GB × 16) .

    Run the following script to execute online inference.

    Common Issues Tip: If you encounter issues, Refer to [Public FAQs](../../faqs.md).

    ```shell
    # The version of transformers needs to be upgraded to 5.2.0.
    # pip install transformers==5.2.0 --upgrade

    export HCCL_OP_EXPANSION_MODE="AIV"
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=1
    export HCCL_BUFFSIZE=200
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

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
    --additional-config '{"multistream_overlap_shared_expert":true,"scheduler_config":{"enable_balance_scheduling":true}}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}'
    ```

    - Quantized model `glm-5-w8a8` and `glm-5.1-w8a8` can be deployed on 1 Atlas 800 A3 (64GB × 16) .

    Run the following script to execute online inference.

    ```shell
    export HCCL_OP_EXPANSION_MODE="AIV"
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=1
    export HCCL_BUFFSIZE=200
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM5-w8a8 \
    --host 0.0.0.0 \
    --port 8077 \
    --data-parallel-size 1 \
    --tensor-parallel-size 16 \
    --enable-expert-parallel \
    --seed 1024 \
    --served-model-name glm-5 \
    --max-num-seqs 16 \
    --max-model-len 40960 \
    --max-num-batched-tokens 4096 \
    --trust-remote-code \
    --gpu-memory-utilization 0.95 \
    --quantization ascend \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --additional-config '{"multistream_overlap_shared_expert":true,"scheduler_config":{"enable_balance_scheduling":true},"enable_mlapo":true}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}'
    ```

=== "A2 series"

    - Quantized model `glm-5-w4a8` can be deployed on 1 Atlas 800 A2 (64GB × 8) .

    Run the following script to execute online inference.

    Common Issues Tip: If you encounter issues, Refer to [Public FAQs](../../faqs.md).

    ```shell
    export HCCL_OP_EXPANSION_MODE="AIV"
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=1
    export HCCL_BUFFSIZE=200
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

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
    --additional-config '{"multistream_overlap_shared_expert":true,"scheduler_config":{"enable_balance_scheduling":true}}' \
    --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}'
    ```

Key Parameter Descriptions:

Only the key parameters specific to this model/scenario are described below. `max-model-len` and `max-num-seqs` need to be set according to the actual usage scenario.

**Model-specific parameters:**

- `--enable-expert-parallel`: Must be enabled for the MoE architecture of GLM-5.
- `--tensor-parallel-size 16` / `--tensor-parallel-size 8`: Tensor parallelism within each DP rank. For A3 (16 NPUs), use `tp16`; for A2 (8 NPUs), use `tp8`.
- `--quantization ascend`: Enables Ascend quantization for w4a8/w8a8 quantized weights.
- `--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}'`: Enables Multi-Token Prediction (MTP) speculative decoding with GLM-5's DeepSeek-style MTP draft model. `num_speculative_tokens` (3-5) controls how many tokens are speculated per step; `enforce_eager: true` is required because GLM-5 does not support graph-mode speculative decoding.
- `--enable-chunked-prefill` / `--enable-prefix-caching`: Recommended for long-context and multi-user scenarios — chunked prefill splits long prompts to improve TTFT, prefix caching reuses KV cache for shared prefixes (e.g., system prompts).
- `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'`: Enables graph capture for the decode phase only, improving decode performance by reducing kernel launch overhead.
- `--additional-config '{"multistream_overlap_shared_expert": true}'`: Overlaps shared-expert computation on an additional stream. Note: automatically disabled when `additional_config.enable_fused_mc2=1`, as the two optimizations conflict.

**Key environment variables:**

- `additional_config.enable_mlapo=true`: Enables the MLA preprocess fusion operator (MlaPreprocessOperation). Enabled by default for w8a8 models — significantly improves Decode performance but consumes more NPU memory; set `additional_config.enable_mlapo=false` if memory is a priority. Recommended for w8a8; w4a8 may not benefit.
- `additional_config.scheduler_config.enable_balance_scheduling=true`: Enables balance scheduling to improve output throughput and reduce TPOT in the v1 scheduler.

**Performance tuning notes for single-node:**

- For low-latency scenarios, use `dp1tp16` (data-parallel-size 1, tensor-parallel-size 16) and consider reducing `--max-num-seqs` and `--max-num-batched-tokens`.
- For high-throughput scenarios, increase `--max-num-seqs` and enable `--enable-prefix-caching`.
- For long-context scenarios (e.g., 200K), use w8a8 weight (more memory for KV cache) and set `--max-model-len` to the desired context length. Consider enabling `--enable-chunked-prefill`.
- If you encounter OOM, reduce `--gpu-memory-utilization`, `--max-num-seqs`, or `--max-model-len`. Disabling `additional_config.enable_mlapo` can also reduce memory usage (at the cost of performance).

### 5.2 Multi-node Deployment

If you want to deploy multi-node environment, you need to verify multi-node communication according to [verify multi-node communication environment](../../installation.md#verify-multi-node-communication).

Common Issues Tip: If you encounter issues, Refer to [Public FAQs](../../faqs.md).

=== "A3 series"

    - `glm-5-bf16` and `glm-5.1-bf16`: require at least 2 Atlas 800 A3 (64GB × 16).

    Run the following scripts on two nodes respectively.

    **node 0**

    ```shell
    # this obtained through ifconfig
    # nic_name is the network interface name corresponding to local_ip of the current node
    nic_name="xxx"
    local_ip="xxx"

    # The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
    node0_ip="xxxx"

    export HCCL_OP_EXPANSION_MODE="AIV"
    export HCCL_IF_IP=$local_ip
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export HCCL_SOCKET_IFNAME=$nic_name
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=1
    export HCCL_BUFFSIZE=200
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM5-bf16 --additional-config '{"scheduler_config":{"enable_balance_scheduling":true}}' \
    --host 0.0.0.0 \
    --port 8077 \
    --data-parallel-size 2 \
    --data-parallel-size-local 1 \
    --data-parallel-address $node0_ip \
    --data-parallel-rpc-port 12890 \
    --tensor-parallel-size 16 \
    --seed 1024 \
    --served-model-name glm-5 \
    --enable-expert-parallel \
    --max-num-seqs 16 \
    --max-model-len 8192 \
    --max-num-batched-tokens 4096 \
    --trust-remote-code \
    --gpu-memory-utilization 0.95 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}'
    ```

    **node 1**

    ```shell
    # this obtained through ifconfig
    # nic_name is the network interface name corresponding to local_ip of the current node
    nic_name="xxx"
    local_ip="xxx"

    # The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
    node0_ip="xxxx"

    export HCCL_OP_EXPANSION_MODE="AIV"
    export HCCL_IF_IP=$local_ip
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export HCCL_SOCKET_IFNAME=$nic_name
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=1
    export HCCL_BUFFSIZE=200
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM5-bf16 --additional-config '{"scheduler_config":{"enable_balance_scheduling":true}}' \
    --host 0.0.0.0 \
    --port 8077 \
    --headless \
    --data-parallel-size 2 \
    --data-parallel-size-local 1 \
    --data-parallel-start-rank 1 \
    --data-parallel-address $node0_ip \
    --data-parallel-rpc-port 12890 \
    --tensor-parallel-size 16 \
    --seed 1024 \
    --served-model-name glm-5 \
    --enable-expert-parallel \
    --max-num-seqs 16 \
    --max-model-len 8192 \
    --max-num-batched-tokens 4096 \
    --trust-remote-code \
    --gpu-memory-utilization 0.95 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}'
    ```

=== "A2 series"

    Run the following scripts on two nodes respectively.

    **node 0**

    ```shell
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
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

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
    --additional-config '{"multistream_overlap_shared_expert":true,"scheduler_config":{"enable_balance_scheduling":true}}' \
    --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}'
    ```

    **node 1**

    ```shell
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
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

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
    --additional-config '{"multistream_overlap_shared_expert":true,"scheduler_config":{"enable_balance_scheduling":true}}' \
    --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}'
    ```

- For bf16 weight, use this script on each node to enable [Multi Token Prediction (MTP)](../../user_guide/feature_guide/speculative_decoding.md).

```shell
python adjust_weight.py "path_of_bf16_weight"
```

```python
# adjust_weight.py
from safetensors.torch import safe_open, save_file
import torch
import json
import os
import sys

target_keys = ["model.embed_tokens.weight", "lm_head.weight"]

def get_tensor_info(file_path):
   with safe_open(file_path, framework="pt", device="cpu") as f:
         tensor_names = f.keys()
         tensor_dict = {}
         for name in tensor_names:
            tensor = f.get_tensor(name)
            tensor_dict[name] = tensor
         return tensor_dict


if __name__ == "__main__":
   directory_path = sys.argv[1]
   json_name = "model.safetensors.index.json"
   json_path = os.path.join(directory_path, json_name)
   with open(json_path, 'r', encoding='utf-8') as f:
         json_data = json.load(f)
   weight_map = json_data.get('weight_map', {})
   file_list = []
   for key in target_keys:
         safetensor_file = weight_map.get(key)
         file_list.append(directory_path + safetensor_file)

   new_dict = {}
   for file_path in file_list:
         tensor_dict = get_tensor_info(file_path)
         for key in target_keys:
            if key in tensor_dict:
               if key == "model.embed_tokens.weight":
                     new_key = "model.layers.78.embed_tokens.weight"
               elif key == "lm_head.weight":
                     new_key = "model.layers.78.shared_head.head.weight"
               new_dict[new_key] = tensor_dict[key]

   new_file_name = os.path.join(directory_path, "mtp-others.safetensors")
   new_keys = ["model.layers.78.embed_tokens.weight", "model.layers.78.shared_head.head.weight"]
   save_file(tensors=new_dict, filename=new_file_name)
   for key in new_keys:
         json_data["weight_map"][key] = "mtp-others.safetensors"
   with open(json_path, 'w', encoding='utf-8') as f:
         json.dump(json_data, f, indent=2)
```

- `glm-5-w8a8`: require 2 Atlas 800 A3 (64GB × 16).

Run the following scripts on two nodes respectively.

**node 0**

```shell
# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip of the current node
nic_name="xxx"
local_ip="xxx"

# The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
node0_ip="xxxx"

export HCCL_OP_EXPANSION_MODE="AIV"

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM5-w8a8 \
--host 0.0.0.0 \
--port 8077 \
--data-parallel-size 2 \
--data-parallel-size-local 1 \
--data-parallel-address $node0_ip \
--data-parallel-rpc-port 12890 \
--tensor-parallel-size 16 \
--seed 1024 \
--served-model-name glm-5 \
--enable-expert-parallel \
--max-num-seqs 16 \
--max-model-len 200000 \
--max-num-batched-tokens 4096 \
--trust-remote-code \
--gpu-memory-utilization 0.95 \
--quantization ascend \
--enable-chunked-prefill \
--enable-prefix-caching \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
--additional-config '{"multistream_overlap_shared_expert":true,"scheduler_config":{"enable_balance_scheduling":true},"enable_mlapo":true}' \
--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}'
```

**node 1**

```shell
# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip of the current node
nic_name="xxx"
local_ip="xxx"

# The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
node0_ip="xxxx"

export HCCL_OP_EXPANSION_MODE="AIV"

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM5-w8a8 \
--host 0.0.0.0 \
--port 8077 \
--headless \
--data-parallel-size 2 \
--data-parallel-size-local 1 \
--data-parallel-start-rank 1 \
--data-parallel-address $node0_ip \
--data-parallel-rpc-port 12890 \
--tensor-parallel-size 16 \
--seed 1024 \
--served-model-name glm-5 \
--enable-expert-parallel \
--max-num-seqs 16 \
--max-model-len 200000 \
--max-num-batched-tokens 4096 \
--trust-remote-code \
--gpu-memory-utilization 0.95 \
--quantization ascend \
--enable-chunked-prefill \
--enable-prefix-caching \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
--additional-config '{"multistream_overlap_shared_expert":true,"scheduler_config":{"enable_balance_scheduling":true},"enable_mlapo":true}' \
--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}'
```

Key Parameter Descriptions for multi-node deployment:

In addition to all single-node parameters described in [Single-Node Online Deployment](#51-single-node-online-deployment), the following parameters are specific to multi-node deployment:

**Network and data parallel configuration:**

- `HCCL_IF_IP`, `GLOO_SOCKET_IFNAME`, `TP_SOCKET_IFNAME`, `HCCL_SOCKET_IFNAME`: Network interface configuration for multi-node communication. Set `nic_name` to the network interface name (obtained via `ifconfig`) and `local_ip` to the current node's IP address. These must be correctly configured on each node for successful multi-node communication.
- `--data-parallel-size`: Total number of data parallel ranks across all nodes. For 2-node deployment, typically set to `2`.
- `--data-parallel-size-local`: Number of data parallel ranks on the current node. Usually set to `1` (one DP rank per node).
- `--data-parallel-address`: IP address of the data parallel master node (node 0). Must match the `local_ip` of the master node.
- `--data-parallel-rpc-port`: RPC port for data parallel master communication. Must be the same across all nodes.
- `--headless`: Indicates this is a non-master node. Do not use on node 0.
- `--data-parallel-start-rank`: Starting rank offset for data parallel ranks on this node. Node 0 uses `0`, node 1 uses `1`.

**Multi-node performance tuning:**

- For low-latency multi-node scenarios, keep `--data-parallel-size-local 1` to minimize cross-node communication.
- `--max-num-seqs` should be tuned based on available KV cache memory after model loading. For w8a8 on A3 multi-node, `16` is recommended. For w4a8 on A2 multi-node with long context, start with `2` and increase if memory permits.
- All nodes in a multi-node deployment must use identical `--tensor-parallel-size`, `--enable-expert-parallel`, and model weight path configurations.

### 5.3 Prefill-Decode Disaggregation

We'd like to show the deployment guide of `GLM-5` on multi-node environment with 1P1D for better performance. *Prefill-Decode Disaggregation* refers to the separation of the prefill stage and the decode stage across different nodes to improve throughput and latency.

Before you start, please

1. prepare the script `launch_online_dp.py` on each node:

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

2. prepare the script `run_dp_template.sh` on each node.

    1. Prefill node 0

        ```shell
        nic_name="xxxx" # change to your own nic name
        local_ip="xxxx" # change to your own ip

        export HCCL_OP_EXPANSION_MODE="AIV"
        export HCCL_IF_IP=$local_ip
        export GLOO_SOCKET_IFNAME=$nic_name
        export TP_SOCKET_IFNAME=$nic_name
        export HCCL_SOCKET_IFNAME=$nic_name
        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=1
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export HCCL_BUFFSIZE=256
        export ASCEND_AGGREGATE_ENABLE=1
        export ASCEND_TRANSPORT_PRINT=1
        export ACL_OP_INIT_MODE=1
        export ASCEND_A3_ENABLE=1
        # Timeout (in seconds) for automatically releasing the prefiller’s KV cache for a particular request.
        export VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=480
        export ASCEND_RT_VISIBLE_DEVICES=$1
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

        vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM5-w8a8 \
            --host 0.0.0.0 \
            --port $2 \
            --data-parallel-size $3 \
            --data-parallel-rank $4 \
            --data-parallel-address $5 \
            --data-parallel-rpc-port $6 \
            --tensor-parallel-size $7 \
            --enable-expert-parallel \
            --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp", "enforce_eager": true}' \
            --seed 1024 \
            --served-model-name glm-5 \
            --max-model-len 131072 \
            --additional-config '{"enable_dsa_cp": true,"enable_fused_mc2":1}' \
            --max-num-batched-tokens 4096 \
            --trust-remote-code \
            --max-num-seqs 64 \
            --enable-chunked-prefill \
            --quantization ascend \
            --gpu-memory-utilization 0.95 \
            --enforce-eager \
            --enable-auto-tool-choice \
            --tool-call-parser glm47 \
            --reasoning-parser glm45 \
            --kv-transfer-config \
            '{"kv_connector": "MooncakeConnectorV1",
            "kv_role": "kv_producer",
            "kv_port": "30000",
            "kv_connector_extra_config": {
                        "use_ascend_direct": true,
                        "prefill": {
                                "dp_size": 2,
                                "tp_size": 16
                        },
                        "decode": {
                                "dp_size": 16,
                                "tp_size": 4
                        }
                }
            }'

        ```

    2. Prefill node 1

        ```shell
        nic_name="xxxx" # change to your own nic name
        local_ip="xxxx" # change to your own ip

        export HCCL_OP_EXPANSION_MODE="AIV"
        export HCCL_IF_IP=$local_ip
        export GLOO_SOCKET_IFNAME=$nic_name
        export TP_SOCKET_IFNAME=$nic_name
        export HCCL_SOCKET_IFNAME=$nic_name
        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=1
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export HCCL_BUFFSIZE=256
        export ASCEND_AGGREGATE_ENABLE=1
        export ASCEND_TRANSPORT_PRINT=1
        export ACL_OP_INIT_MODE=1
        export ASCEND_A3_ENABLE=1
        # Timeout (in seconds) for automatically releasing the prefiller’s KV cache for a particular request.
        export VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=480
        export ASCEND_RT_VISIBLE_DEVICES=$1
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

        vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM5-w8a8 \
            --host 0.0.0.0 \
            --port $2 \
            --data-parallel-size $3 \
            --data-parallel-rank $4 \
            --data-parallel-address $5 \
            --data-parallel-rpc-port $6 \
            --tensor-parallel-size $7 \
            --enable-expert-parallel \
            --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp", "enforce_eager": true}' \
            --seed 1024 \
            --served-model-name glm-5 \
            --max-model-len 131072 \
            --additional-config '{"enable_dsa_cp": true,"enable_fused_mc2":1}' \
            --max-num-batched-tokens 4096 \
            --trust-remote-code \
            --max-num-seqs 64 \
            --enable-chunked-prefill \
            --gpu-memory-utilization 0.95 \
            --quantization ascend \
            --enforce-eager \
            --enable-auto-tool-choice \
            --tool-call-parser glm47 \
            --reasoning-parser glm45 \
            --kv-transfer-config \
            '{"kv_connector": "MooncakeConnectorV1",
            "kv_role": "kv_producer",
            "kv_port": "30000",
            "kv_connector_extra_config": {
                        "use_ascend_direct": true,
                        "prefill": {
                                "dp_size": 2,
                                "tp_size": 16
                        },
                        "decode": {
                                "dp_size": 16,
                                "tp_size": 4
                        }
                }
            }'
        ```

    3. Decode node 0

        ```shell
        nic_name="xxxx" # change to your own nic name
        local_ip="xxxx" # change to your own ip

        export HCCL_OP_EXPANSION_MODE="AIV"
        export HCCL_IF_IP=$local_ip
        export GLOO_SOCKET_IFNAME=$nic_name
        export TP_SOCKET_IFNAME=$nic_name
        export HCCL_SOCKET_IFNAME=$nic_name
        #Mooncake
        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=1
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export HCCL_BUFFSIZE=256
        export ASCEND_AGGREGATE_ENABLE=1
        export ASCEND_TRANSPORT_PRINT=1
        export ACL_OP_INIT_MODE=1
        export ASCEND_A3_ENABLE=1
        # Timeout (in seconds) for automatically releasing the prefiller’s KV cache for a particular request.
        export VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=480
        export TASK_QUEUE_ENABLE=1
        export ASCEND_RT_VISIBLE_DEVICES=$1
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

        vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM5-w8a8 \
            --host 0.0.0.0 \
            --port $2 \
            --data-parallel-size $3 \
            --data-parallel-rank $4 \
            --data-parallel-address $5 \
            --data-parallel-rpc-port $6 \
            --tensor-parallel-size $7 \
            --enable-expert-parallel \
            --speculative-config '{"num_speculative_tokens": 3,  "method":"deepseek_mtp", "enforce_eager": true}' \
            --seed 1024 \
            --served-model-name glm-5 \
            --max-model-len 200000 \
            --max-num-batched-tokens 32 \
            --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
            --additional-config '{"recompute_scheduler_enable": true,"enable_fused_mc2":1,"enable_mlapo":true}' \
            --trust-remote-code \
            --max-num-seqs 8 \
            --gpu-memory-utilization 0.92 \
            --quantization ascend \
            --enable-auto-tool-choice \
            --tool-call-parser glm47 \
            --reasoning-parser glm45 \
            --kv-transfer-config \
            '{"kv_connector": "MooncakeConnectorV1",
            "kv_role": "kv_consumer",
            "kv_port": "30100",
            "kv_connector_extra_config": {
                        "use_ascend_direct": true,
                        "prefill": {
                                "dp_size": 2,
                                "tp_size": 16
                        },
                        "decode": {
                                "dp_size": 16,
                                "tp_size": 4
                        }
                }
            }'
        ```

    4. Decode node 1

         ```shell
         nic_name="xxxx" # change to your own nic name
         local_ip="xxxx" # change to your own ip

         export HCCL_OP_EXPANSION_MODE="AIV"
         export HCCL_IF_IP=$local_ip
         export GLOO_SOCKET_IFNAME=$nic_name
         export TP_SOCKET_IFNAME=$nic_name
         export HCCL_SOCKET_IFNAME=$nic_name
         #Mooncake
         export OMP_PROC_BIND=false
         export OMP_NUM_THREADS=1
         export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
         export HCCL_BUFFSIZE=256
         export ASCEND_AGGREGATE_ENABLE=1
         export ASCEND_TRANSPORT_PRINT=1
         export ACL_OP_INIT_MODE=1
         export ASCEND_A3_ENABLE=1
         # Timeout (in seconds) for automatically releasing the prefiller’s KV cache for a particular request.
         export VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=480
         export TASK_QUEUE_ENABLE=1
         export ASCEND_RT_VISIBLE_DEVICES=$1
         export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

         vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM5-w8a8 \
             --host 0.0.0.0 \
             --port $2 \
             --data-parallel-size $3 \
             --data-parallel-rank $4 \
             --data-parallel-address $5 \
             --data-parallel-rpc-port $6 \
             --tensor-parallel-size $7 \
             --enable-expert-parallel \
             --speculative-config '{"num_speculative_tokens": 3,  "method":"deepseek_mtp", "enforce_eager": true}' \
             --seed 1024 \
             --served-model-name glm-5 \
             --max-model-len 200000 \
             --max-num-batched-tokens 32 \
             --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
             --additional-config '{"recompute_scheduler_enable": true,"enable_fused_mc2":1,"enable_mlapo":true}' \
             --trust-remote-code \
             --max-num-seqs 8 \
             --gpu-memory-utilization 0.92 \
             --quantization ascend \
             --enable-auto-tool-choice \
             --tool-call-parser glm47 \
             --reasoning-parser glm45 \
             --kv-transfer-config \
             '{"kv_connector": "MooncakeConnectorV1",
             "kv_role": "kv_consumer",
             "kv_port": "30100",
             "kv_connector_extra_config": {
                         "use_ascend_direct": true,
                         "prefill": {
                                 "dp_size": 2,
                                 "tp_size": 16
                         },
                         "decode": {
                                 "dp_size": 16,
                                 "tp_size": 4
                         }
                 }
             }'
         ```

    5. Decode node 2

         ```shell
         nic_name="xxxx" # change to your own nic name
         local_ip="xxxx" # change to your own ip

         export HCCL_OP_EXPANSION_MODE="AIV"
         export HCCL_IF_IP=$local_ip
         export GLOO_SOCKET_IFNAME=$nic_name
         export TP_SOCKET_IFNAME=$nic_name
         export HCCL_SOCKET_IFNAME=$nic_name
         #Mooncake
         export OMP_PROC_BIND=false
         export OMP_NUM_THREADS=1
         export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
         export HCCL_BUFFSIZE=256
         export ASCEND_AGGREGATE_ENABLE=1
         export ASCEND_TRANSPORT_PRINT=1
         export ACL_OP_INIT_MODE=1
         export ASCEND_A3_ENABLE=1
         # Timeout (in seconds) for automatically releasing the prefiller’s KV cache for a particular request.
         export VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=480
         export TASK_QUEUE_ENABLE=1
         export ASCEND_RT_VISIBLE_DEVICES=$1
         export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

         vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM5-w8a8 \
             --host 0.0.0.0 \
             --port $2 \
             --data-parallel-size $3 \
             --data-parallel-rank $4 \
             --data-parallel-address $5 \
             --data-parallel-rpc-port $6 \
             --tensor-parallel-size $7 \
             --enable-expert-parallel \
             --speculative-config '{"num_speculative_tokens": 3,  "method":"deepseek_mtp", "enforce_eager": true}' \
             --seed 1024 \
             --served-model-name glm-5 \
             --max-model-len 200000 \
             --max-num-batched-tokens 32 \
             --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
             --additional-config '{"recompute_scheduler_enable": true,"enable_fused_mc2":1,"enable_mlapo":true}' \
             --trust-remote-code \
             --max-num-seqs 8 \
             --gpu-memory-utilization 0.92 \
             --quantization ascend \
             --enable-auto-tool-choice \
             --tool-call-parser glm47 \
             --reasoning-parser glm45 \
             --kv-transfer-config \
             '{"kv_connector": "MooncakeConnectorV1",
             "kv_role": "kv_consumer",
             "kv_port": "30100",
             "kv_connector_extra_config": {
                         "use_ascend_direct": true,
                         "prefill": {
                                 "dp_size": 2,
                                 "tp_size": 16
                         },
                         "decode": {
                                 "dp_size": 16,
                                 "tp_size": 4
                         }
                 }
             }'
         ```

    6. Decode node 3

         ```shell
         nic_name="xxxx" # change to your own nic name
         local_ip="xxxx" # change to your own ip

         export HCCL_OP_EXPANSION_MODE="AIV"
         export HCCL_IF_IP=$local_ip
         export GLOO_SOCKET_IFNAME=$nic_name
         export TP_SOCKET_IFNAME=$nic_name
         export HCCL_SOCKET_IFNAME=$nic_name
         #Mooncake
         export OMP_PROC_BIND=false
         export OMP_NUM_THREADS=1
         export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
         export HCCL_BUFFSIZE=256
         export ASCEND_AGGREGATE_ENABLE=1
         export ASCEND_TRANSPORT_PRINT=1
         export ACL_OP_INIT_MODE=1
         export ASCEND_A3_ENABLE=1
         # Timeout (in seconds) for automatically releasing the prefiller’s KV cache for a particular request.
         export VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=480
         export TASK_QUEUE_ENABLE=1
         export ASCEND_RT_VISIBLE_DEVICES=$1
         export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

         vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM5-w8a8 \
             --host 0.0.0.0 \
             --port $2 \
             --data-parallel-size $3 \
             --data-parallel-rank $4 \
             --data-parallel-address $5 \
             --data-parallel-rpc-port $6 \
             --tensor-parallel-size $7 \
             --enable-expert-parallel \
             --speculative-config '{"num_speculative_tokens": 3,  "method":"deepseek_mtp", "enforce_eager": true}' \
             --seed 1024 \
             --served-model-name glm-5 \
             --max-model-len 200000 \
             --max-num-batched-tokens 32 \
             --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
             --additional-config '{"recompute_scheduler_enable": true,"enable_fused_mc2":1,"enable_mlapo":true}' \
             --trust-remote-code \
             --max-num-seqs 8 \
             --gpu-memory-utilization 0.92 \
             --quantization ascend \
             --enable-auto-tool-choice \
             --tool-call-parser glm47 \
             --reasoning-parser glm45 \
             --kv-transfer-config \
             '{"kv_connector": "MooncakeConnectorV1",
             "kv_role": "kv_consumer",
             "kv_port": "30100",
             "kv_connector_extra_config": {
                         "use_ascend_direct": true,
                         "prefill": {
                                 "dp_size": 2,
                                 "tp_size": 16
                         },
                         "decode": {
                                 "dp_size": 16,
                                 "tp_size": 4
                         }
                 }
             }'
         ```

Once the preparation is done, you can start the server with the following command on each node:

1. Prefill node 0

    ```shell
    # change ip to your own
    python launch_online_dp.py --dp-size 2 --tp-size 16 --dp-size-local 1 --dp-rank-start 0 --dp-address $node_p0_ip --dp-rpc-port 10521 --vllm-start-port 6700
    ```

2. Prefill node 1

    ```shell
    # change ip to your own
    python launch_online_dp.py --dp-size 2 --tp-size 16 --dp-size-local 1 --dp-rank-start 1 --dp-address $node_p0_ip --dp-rpc-port 10521 --vllm-start-port 6700
    ```

3. Decode node 0

    ```shell
    # change ip to your own
    python launch_online_dp.py --dp-size 16 --tp-size 4 --dp-size-local 4 --dp-rank-start 0 --dp-address $node_d0_ip --dp-rpc-port 10523 --vllm-start-port 6721
    ```

4. Decode node 1

    ```shell
    # change ip to your own
    python launch_online_dp.py --dp-size 16 --tp-size 4 --dp-size-local 4 --dp-rank-start 4 --dp-address $node_d0_ip --dp-rpc-port 10523 --vllm-start-port 6721
    ```

5. Decode node 2

    ```shell
    # change ip to your own
    python launch_online_dp.py --dp-size 16 --tp-size 4 --dp-size-local 4 --dp-rank-start 8 --dp-address $node_d0_ip --dp-rpc-port 10523 --vllm-start-port 6721
    ```

6. Decode node 3

    ```shell
    # change ip to your own
    python launch_online_dp.py --dp-size 16 --tp-size 4 --dp-size-local 4 --dp-rank-start 12 --dp-address $node_d0_ip --dp-rpc-port 10523 --vllm-start-port 6721
    ```

### 5.4 Request Forwarding

To set up request forwarding, run the following script on any machine. You can get the proxy program in the repository's examples: [load_balance_proxy_server_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py)

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
      $node_d1_ip \
      $node_d1_ip \
      $node_d1_ip \
      $node_d1_ip \
      $node_d2_ip \
      $node_d2_ip \
      $node_d2_ip \
      $node_d2_ip \
      $node_d3_ip \
      $node_d3_ip \
      $node_d3_ip \
      $node_d3_ip \
    --decoder-ports \
      6721 6722 6723 6724 \
      6721 6722 6723 6724 \
      6721 6722 6723 6724 \
      6721 6722 6723 6724
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

- `additional_config.enable_fused_mc2=1`: Enables fused MC2 operators (`dispatch_ffn_combine`/`mega_moe`) to optimize MoE communication. Constraints: `dispatch_ffn_combine` only for w8a8 and EP≤32; `mega_moe` works for w8a8/w4a8/bf16 with EP≤64. Both are incompatible with MTP and dynamic EPLB.
- `--additional-config '{"enable_dsa_cp": true}'`: Enables DSA context parallelism on prefill nodes to accelerate long-context prefill. Required for handling prompts up to 128K tokens.

**Decode node-specific configurations:**

- `additional_config.enable_mlapo=true`: Enables MLA preprocess operation fusion on decode nodes to significantly improve decode performance. Consumes more NPU memory. In PD scenarios, enable MLAPO only on decode nodes.
- `--max-num-batched-tokens 32`: Small batch token limit on decode nodes — decode processes one token per sequence per step, so batch tokens should be close to `max-num-seqs`.
- `--additional-config '{"recompute_scheduler_enable": true}'`: Enables the recomputation scheduler. When decode node KV cache is insufficient, requests are sent back to prefill nodes for KV cache recomputation. Recommended on both prefill and decode nodes in PD scenarios.

**Common PD environment variables:**

- `ASCEND_AGGREGATE_ENABLE=1`, `ASCEND_A3_ENABLE=1`: A3-specific optimizations for communication aggregation.
- `ACL_OP_INIT_MODE=1`: ACL operator initialization mode.
- `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=480`: Timeout (in seconds) for automatically releasing the prefill node's KV cache when a request is aborted.
- `LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib`: Required for Mooncake library loading.

**MTP in PD scenarios:**

- Prefill nodes typically use `"num_speculative_tokens": 1` for MTP (minimal speculation during prefill).
- Decode nodes use `"num_speculative_tokens": 3` for MTP to maximize decode throughput.
- Both prefill and decode nodes must use the same `"method": "deepseek_mtp"` and `"enforce_eager": true`.

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

The tables below provide recommended parameter configurations for different deployment scenarios. All scenarios are categorized by use case (High Throughput, Low Latency, Long Context) and correspond to the deployment modes documented in [Online Service Deployment](#5-online-service-deployment).

#### 9.1.1 Table 1: Scenario Overview

> `*Total NPUs` indicates the total number of NPUs used across all nodes. 1 node = 1 Atlas 800 A3 server (64GB × 16 NPUs) or 1 Atlas 800 A2 server (64GB × 8 NPUs).

|Scenario|Deployment Mode|*Total NPUs|Weight Version|Key Considerations|
|--------|---------------|-----------|--------------|------------------|
|Low Latency<br>(64K input)|PD Disaggregation, [Prefill-Decode Disaggregation](#53-prefill-decode-disaggregation)|4 nodes (A3)|w8a8c8|P: dp4 tp8 (max-num-seqs 64, max-num-batched-tokens 8192); D: dp8 tp4 (max-num-seqs 32, max-num-batched-tokens 164); MTP3, max-model-len 202752, Mooncake KV transfer|
|Low Latency<br>(128K input)|PD Disaggregation, [Prefill-Decode Disaggregation](#53-prefill-decode-disaggregation)|4 nodes (A3)|w8a8c8|P: dp4 tp8 (max-num-seqs 64, max-num-batched-tokens 8192); D: dp8 tp4 (max-num-seqs 32, max-num-batched-tokens 164); MTP3, max-model-len 202752, Mooncake KV transfer|
|High Throughput<br>(64K input)|PD Disaggregation, [Prefill-Decode Disaggregation](#53-prefill-decode-disaggregation)|4 nodes (A3)|w8a8c8|P: dp4 tp8 (max-num-seqs 64, max-num-batched-tokens 8192); D: dp8 tp4 (max-num-seqs 32, max-num-batched-tokens 164); MTP3, max-model-len 202752, Mooncake KV transfer|
|High Throughput<br>(128K input)|PD Disaggregation, [Prefill-Decode Disaggregation](#53-prefill-decode-disaggregation)|4 nodes (A3)|w8a8c8|P: dp4 tp8 (max-num-seqs 64, max-num-batched-tokens 8192); D: dp8 tp4 (max-num-seqs 32, max-num-batched-tokens 164); MTP3, max-model-len 202752, Mooncake KV transfer|
|Long Context<br>(198K input)|PD Disaggregation, [Prefill-Decode Disaggregation](#53-prefill-decode-disaggregation)|4 nodes (A3)|w8a8c8|P: dp4 tp8 (max-num-seqs 64, max-num-batched-tokens 8192); D: dp8 tp4 (max-num-seqs 32, max-num-batched-tokens 164); MTP3, max-model-len 202752, Mooncake KV transfer|

#### 9.1.2 Table 2: Detailed Node Configuration

> The TP/DP columns show the values **per node** as configured in the Deployment scripts (a prefill node hosting 2 DP ranks of TP8 uses 16 NPUs; a decode node hosting 4 DP ranks of TP4 uses 16 NPUs).

|Scenario|Configuration|NPUs|TP|DP|Max Num Seqs|Max Num Batched Tokens|Max Model Len|MTP Spec Num|
|--------|-------------|-----|--|--|------------|----------------------|--------------|-------------|
|Low Latency 64K (A3)|PD — Server-P Node|16|8|2|64|8192|202752|3|
|Low Latency 64K (A3)|PD — Server-D Node|16|4|4|32|164|202752|3|
|Low Latency 128K (A3)|PD — Server-P Node|16|8|2|64|8192|202752|3|
|Low Latency 128K (A3)|PD — Server-D Node|16|4|4|32|164|202752|3|
|High Throughput 64K (A3)|PD — Server-P Node|16|8|2|64|8192|202752|3|
|High Throughput 64K (A3)|PD — Server-D Node|16|4|4|32|164|202752|3|
|High Throughput 128K (A3)|PD — Server-P Node|16|8|2|64|8192|202752|3|
|High Throughput 128K (A3)|PD — Server-D Node|16|4|4|32|164|202752|3|
|Long Context 198K (A3)|PD — Server-P Node|16|8|2|64|8192|202752|3|
|Long Context 198K (A3)|PD — Server-D Node|16|4|4|32|164|202752|3|

> For complete startup commands and detailed parameter descriptions, please refer to the deployment examples and Key Parameter Descriptions in [Online Service Deployment](#5-online-service-deployment).

#### 9.1.3 Table 3: Performance-Related Parameter Tuning Guide

|Parameter|Low Latency|High Throughput|Long Context|Description|
|---------|-----------|---------------|-------------|-----------|
|`--max-num-seqs`|Lower (4–8)|Higher (16–64)|Controlled (2–8)|Limits concurrent sequences. Lower values reduce scheduling latency; higher values increase throughput.|
|`--max-model-len`|Shorter (32K–40K)|Longer (128K–200K)|Maximum (128K–200K)|Maximum context length. Must accommodate your longest input+output. Larger values consume more KV cache memory.|
|`--max-num-batched-tokens`|Lower (2048–4096)|Higher (4096–8192)|Higher for prefill (4096)|Controls batch size per step. Lower values reduce per-step latency; higher values improve prefill throughput.|
|`--gpu-memory-utilization`|0.92–0.95|0.95|0.92–0.95|NPU memory fraction. Higher values leave more memory for KV cache. Reduce if OOM.|
|`--enable-chunked-prefill`|Enable|Enable|Enable|Splits long prompts into chunks to prevent prefill from blocking decode. Recommended in all scenarios.|
|`--enable-prefix-caching`|Optional|Enable|Optional|Reuses KV cache for shared prefixes (e.g., system prompts). Improves throughput when cache hit rate is high but may reduce available KV cache memory.|
|`num_speculative_tokens`|3|3|3|MTP speculation count. Higher values improve decode throughput at the cost of memory for draft model KV cache. Use `1` on prefill nodes in PD mode.|
|`additional_config.enable_mlapo`|1 (w8a8)|1 (w8a8)|0 or 1|Enables MLA fusion on w8a8 models. Improves decode performance but consumes more NPU memory. Disable for long-context if memory is insufficient.|

### 9.2 Tuning Guidelines

For general performance tuning methods, refer to the [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md).

For detailed feature descriptions and configuration options, refer to the [Feature Matrix](../../user_guide/support_matrix/feature_matrix.md).

For environment variable descriptions and constraints, refer to [envs.py](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/envs.py).

## 10 FAQ

- Common Issues Tip: If you encounter issues, Refer to [Public FAQs](../../faqs.md).

- **Q: How to solve ValueError: Tokenizer class TokenizersBackend does not exist or is not currently imported?**

  A: Please update the version of transformers to 5.2.0

- **Q: How to enable function calling for GLM-5?**

  A: Please add following configurations in vLLM startup command

  ```shell
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --enable-auto-tool-choice \
  ```
