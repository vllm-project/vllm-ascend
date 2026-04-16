# GLM-5

## Introduction

[GLM-5](https://huggingface.co/zai-org/GLM-5) use a Mixture-of-Experts (MoE) architecture and targets complex systems engineering and long-horizon agentic tasks.

The `GLM-5` model is first supported in `vllm-ascend:v0.17.0rc1`. In `vllm-ascend:v0.17.0rc1` and `vllm-ascend:v0.18.0rc1` , the version of transformers need to be upgraded to 5.2.0.

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, single-node and multi-node deployment, accuracy and performance evaluation.

## Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## Environment Preparation

### Model Weight

- `GLM-5`(BF16 version): [Download model weight](https://www.modelscope.cn/models/ZhipuAI/GLM-5).
- `GLM-5-w4a8`: [Download model weight](https://modelscope.cn/models/Eco-Tech/GLM-5-w4a8).
- `GLM-5-w8a8`: [Download model weight](https://www.modelscope.cn/models/Eco-Tech/GLM-5-w8a8).
- You can use [msmodelslim](https://gitcode.com/Ascend/msmodelslim) to quantify the model naively.

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`

### Installation

You can use our official docker image to run GLM-5 directly.

:::::{tab-set}
:sync-group: install

::::{tab-item} A3 series
:sync: A3

Start the docker image on your each node.

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

Start the docker image on your each node.

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

::::
:::::

In addition, if you don't want to use the docker image as above, you can also build all from source:

- Install `vllm-ascend` from source, refer to [installation](https://docs.vllm.ai/projects/ascend/en/latest/installation.html).

If you want to deploy multi-node environment, you need to set up environment on each node.

## Deployment

### Single-node Deployment

:::::{tab-set}
:sync-group: install

::::{tab-item} A3 series
:sync: A3

- Quantized model `glm-5-w4a8` can be deployed on 1 Atlas 800 A3 (64G × 16) .

Run the following script to execute online inference.

```{code-block} bash
   :substitutions:
export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_BALANCE_SCHEDULING=1

vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM5-w4a8 \
--host 0.0.0.0 \
--port 8077 \
--data-parallel-size 1 \
--tensor-parallel-size 16 \
--enable-expert-parallel \
--seed 1024 \
--served-model-name glm-5 \
--max-num-seqs 8 \
--max-model-len 200000 \
--max-num-batched-tokens 4096 \
--trust-remote-code \
--gpu-memory-utilization 0.95 \
--quantization ascend \
--enable-chunked-prefill \
--enable-prefix-caching \
--async-scheduling \
--additional-config '{"fuse_muls_add": true, "multistream_overlap_shared_expert": true, "ascend_compilation_config": {"enable_npugraph_ex": true}}' \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}' 
```

- Quantized model `glm-5-w8a8` can be deployed on 1 Atlas 800 A3 (64G × 16) .

Run the following script to execute online inference.

```{code-block} bash
   :substitutions:
export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_BALANCE_SCHEDULING=1
export VLLM_ASCEND_ENABLE_MLAPO=1

vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM5-w8a8 \
--host 0.0.0.0 \
--port 8077 \
--data-parallel-size 1 \
--tensor-parallel-size 16 \
--enable-expert-parallel \
--seed 1024 \
--served-model-name glm-5 \
--max-num-seqs 8 \
--max-model-len 40960 \
--max-num-batched-tokens 4096 \
--trust-remote-code \
--gpu-memory-utilization 0.95 \
--quantization ascend \
--enable-chunked-prefill \
--enable-prefix-caching \
--async-scheduling \
--additional-config '{"fuse_muls_add": true, "multistream_overlap_shared_expert": true, "ascend_compilation_config": {"enable_npugraph_ex": true}}' \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}' 
```

::::
::::{tab-item} A2 series
:sync: A2

- Quantized model `glm-5-w4a8` can be deployed on 1 Atlas 800 A2 (64G × 8) .

Run the following script to execute online inference.

```{code-block} bash
   :substitutions:
export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_BALANCE_SCHEDULING=1

vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5-w4a8 \
--host 0.0.0.0 \
--port 8077 \
--data-parallel-size 1 \
--tensor-parallel-size 8 \
--enable-expert-parallel \
--seed 1024 \
--served-model-name glm-5 \
--max-num-seqs 2 \
--max-model-len 32768 \
--max-num-batched-tokens 4096 \
--trust-remote-code \
--gpu-memory-utilization 0.95 \
--quantization ascend \
--enable-chunked-prefill \
--enable-prefix-caching \
--async-scheduling \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
--additional-config '{"fuse_muls_add": true, "multistream_overlap_shared_expert": true, "ascend_compilation_config": {"enable_npugraph_ex": true}}' \
--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}'
```

::::
:::::

**Notice:**
The parameters are explained as follows:

- For single-node deployment, we recommend using `dp1tp16` and turn off expert parallel in low-latency scenarios.
- `--async-scheduling` Asynchronous scheduling is a technique used to optimize inference efficiency. It allows non-blocking task scheduling to improve concurrency and throughput, especially when processing large-scale models.

### Multi-node Deployment

If you want to deploy multi-node environment, you need to verify multi-node communication according to [verify multi-node communication environment](../../installation.md#verify-multi-node-communication).

:::::{tab-set}
:sync-group: install

::::{tab-item} A3 series
:sync: A3

- `glm-5-bf16`: require at least 2 Atlas 800 A3 (64G × 16).

Run the following scripts on two nodes respectively.

**node 0**

```{code-block} bash
   :substitutions:
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
export VLLM_ASCEND_BALANCE_SCHEDULING=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM5-bf16 \
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
--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}'
```

**node 1**

```{code-block} bash
   :substitutions:
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
export VLLM_ASCEND_BALANCE_SCHEDULING=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM5-bf16 \
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
--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}'
```

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

vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5-w4a8 \
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
--additional-config '{"fuse_muls_add": true, "multistream_overlap_shared_expert": true, "ascend_compilation_config": {"enable_npugraph_ex": true}}' \
--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}'
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

vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5-w4a8 \
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
--additional-config '{"fuse_muls_add": true, "multistream_overlap_shared_expert": true, "ascend_compilation_config": {"enable_npugraph_ex": true}}' \
--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}'
```

::::
:::::

- For bf16 weight, use this script on each node to enable [Multi Token Prediction (MTP)](../../user_guide/feature_guide/Multi_Token_Prediction.md).

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
   new_key = ["model.layers.78.embed_tokens.weight", "model.layers.78.shared_head.head.weight"]
   save_file(tensors=new_dict, filename=new_file_name)
   for key in new_key:
         json_data["weight_map"][key] = "mtp-others.safetensors"
   with open(json_path, 'w', encoding='utf-8') as f:
         json.dump(json_data, f, indent=2)
```

:::::{tab-set}
:sync-group: install

::::{tab-item} A3 series
:sync: A3

- `glm-5-w8a8`: require 2 Atlas 800 A3 (64G × 16).

Run the following scripts on two nodes respectively.

**node 0**

```{code-block} bash
   :substitutions:
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
export VLLM_ASCEND_BALANCE_SCHEDULING=1
export VLLM_ASCEND_ENABLE_MLAPO=1

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
--async-scheduling \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
--additional-config '{"fuse_muls_add": true, "multistream_overlap_shared_expert": true, "ascend_compilation_config": {"enable_npugraph_ex": true}}' \
--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}'
```

**node 1**

```{code-block} bash
   :substitutions:
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
export VLLM_ASCEND_BALANCE_SCHEDULING=1
export VLLM_ASCEND_ENABLE_MLAPO=1

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
--async-scheduling \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
--additional-config '{"fuse_muls_add": true, "multistream_overlap_shared_expert": true, "ascend_compilation_config": {"enable_npugraph_ex": true}}' \
--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}'
```

::::
:::::

### Prefill-Decode Disaggregation

We'd like to show the deployment guide of `GLM-5` on multi-node environment with 1P1D for better performance.

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

    To support a 200k context window on the stage of prefill, the parameter `"layer_sharding": ["q_b_proj"]` needs to be added to `--additional_config` on each prefill node.
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
        export VLLM_NIXL_ABORT_REQUEST_TIMEOUT=300000

        export ASCEND_RT_VISIBLE_DEVICES=$1
        export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
          
        export VLLM_ASCEND_ENABLE_FUSED_MC2=1
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

        vllm serve /root/.cache/glm5-w8a8 \
            --host 0.0.0.0 \
            --port $2 \
            --data-parallel-size $3 \
            --data-parallel-rank $4 \
            --data-parallel-address $5 \
            --data-parallel-rpc-port $6 \
            --tensor-parallel-size $7 \
            --enable-expert-parallel \
            --speculative-config '{"num_speculative_tokens": 3, "method":"deepseek_mtp"}' \
            --profiler-config \
            '{"profiler": "torch",
            "torch_profiler_dir": "./vllm_profile",
            "torch_profiler_with_stack": false}' \
            --seed 1024 \
            --served-model-name glm-5 \
            --max-model-len 131072 \
            --additional-config '{"fuse_muls_add": true, "multistream_overlap_shared_expert": true, "recompute_scheduler_enable": true, "ascend_compilation_config": {"enable_npugraph_ex": true}}' \
            --max-num-batched-tokens 4096 \
            --trust-remote-code \
            --max-num-seqs 64 \
            --async-scheduling \
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
            "engine_id": "0",
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
        export VLLM_NIXL_ABORT_REQUEST_TIMEOUT=300000

        export ASCEND_RT_VISIBLE_DEVICES=$1
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

        export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
       
        export VLLM_ASCEND_ENABLE_FUSED_MC2=1
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

        vllm serve /root/.cache/glm5-w8a8 \
            --host 0.0.0.0 \
            --port $2 \
            --data-parallel-size $3 \
            --data-parallel-rank $4 \
            --data-parallel-address $5 \
            --data-parallel-rpc-port $6 \
            --tensor-parallel-size $7 \
            --enable-expert-parallel \
            --speculative-config '{"num_speculative_tokens": 3, "method":"deepseek_mtp"}' \
            --profiler-config \
            '{"profiler": "torch",
            "torch_profiler_dir": "./vllm_profile",
            "torch_profiler_with_stack": false}' \
            --seed 1024 \
            --served-model-name glm-5 \
            --max-model-len 131072 \
            --additional-config '{"fuse_muls_add": true, "multistream_overlap_shared_expert": true, "recompute_scheduler_enable": true, "ascend_compilation_config": {"enable_npugraph_ex": true}}' \
            --max-num-batched-tokens 4096 \
            --trust-remote-code \
            --max-num-seqs 64 \
            --async-scheduling \
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
            "engine_id": "0",
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
        export VLLM_NIXL_ABORT_REQUEST_TIMEOUT=300000
    
        export TASK_QUEUE_ENABLE=1
    
        export ASCEND_RT_VISIBLE_DEVICES=$1
          
        export VLLM_ASCEND_ENABLE_FUSED_MC2=1
        export VLLM_ASCEND_ENABLE_MLAPO=1
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
    
        vllm serve /root/.cache/glm5-w8a8 \
            --host 0.0.0.0 \
            --port $2 \
            --data-parallel-size $3 \
            --data-parallel-rank $4 \
            --data-parallel-address $5 \
            --data-parallel-rpc-port $6 \
            --tensor-parallel-size $7 \
            --enable-expert-parallel \
            --speculative-config '{"num_speculative_tokens": 3,  "method":"deepseek_mtp"}' \
            --profiler-config \
            '{"profiler": "torch",
            "torch_profiler_dir": "./vllm_profile",
            "torch_profiler_with_stack": false}' \
            --seed 1024 \
            --served-model-name glm-5 \
            --max-model-len 200000 \
            --max-num-batched-tokens 32 \
            --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY", "cudagraph_capture_sizes":[4, 8, 12, 16,20,24,28, 32]}' \
            --additional-config '{"fuse_muls_add": true, "multistream_overlap_shared_expert": true, "recompute_scheduler_enable": true, "ascend_compilation_config": {"enable_npugraph_ex": true}}' \
            --trust-remote-code \
            --max-num-seqs 8 \
            --gpu-memory-utilization 0.92 \
            --async-scheduling \
            --quantization ascend \
            --enable-auto-tool-choice \
            --tool-call-parser glm47 \
            --reasoning-parser glm45 \
            --kv-transfer-config \
            '{"kv_connector": "MooncakeConnectorV1",
            "kv_role": "kv_consumer",
            "kv_port": "30100",
            "engine_id": "1",
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
         export VLLM_NIXL_ABORT_REQUEST_TIMEOUT=300000
            
         export TASK_QUEUE_ENABLE=1
            
         export ASCEND_RT_VISIBLE_DEVICES=$1
                     
         export VLLM_ASCEND_ENABLE_FUSED_MC2=1
         export VLLM_ASCEND_ENABLE_MLAPO=1
         export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
            
         vllm serve /root/.cache/glm5-w8a8 \
             --host 0.0.0.0 \
             --port $2 \
             --data-parallel-size $3 \
             --data-parallel-rank $4 \
             --data-parallel-address $5 \
             --data-parallel-rpc-port $6 \
             --tensor-parallel-size $7 \
             --enable-expert-parallel \
             --speculative-config '{"num_speculative_tokens": 3,  "method":"deepseek_mtp"}' \
             --profiler-config \
             '{"profiler": "torch",
             "torch_profiler_dir": "./vllm_profile",
             "torch_profiler_with_stack": false}' \
             --seed 1024 \
             --served-model-name glm-5 \
             --max-model-len 200000 \
             --max-num-batched-tokens 32 \
             --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY", "cudagraph_capture_sizes":[4, 8, 12, 16,20,24,28, 32]}' \
             --additional-config '{"fuse_muls_add": true, "multistream_overlap_shared_expert": true, "recompute_scheduler_enable": true, "ascend_compilation_config": {"enable_npugraph_ex": true}}' \
             --trust-remote-code \
             --max-num-seqs 8 \
             --gpu-memory-utilization 0.92 \
             --async-scheduling \
             --quantization ascend \
             --enable-auto-tool-choice \
             --tool-call-parser glm47 \
             --reasoning-parser glm45 \
             --kv-transfer-config \
             '{"kv_connector": "MooncakeConnectorV1",
             "kv_role": "kv_consumer",
             "kv_port": "30100",
             "engine_id": "1",
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
         export VLLM_NIXL_ABORT_REQUEST_TIMEOUT=300000
            
         export TASK_QUEUE_ENABLE=1
            
         export ASCEND_RT_VISIBLE_DEVICES=$1
                     
         export VLLM_ASCEND_ENABLE_FUSED_MC2=1
         export VLLM_ASCEND_ENABLE_MLAPO=1
         export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
            
         vllm serve /root/.cache/glm5-w8a8 \
             --host 0.0.0.0 \
             --port $2 \
             --data-parallel-size $3 \
             --data-parallel-rank $4 \
             --data-parallel-address $5 \
             --data-parallel-rpc-port $6 \
             --tensor-parallel-size $7 \
             --enable-expert-parallel \
             --speculative-config '{"num_speculative_tokens": 3,  "method":"deepseek_mtp"}' \
             --profiler-config \
             '{"profiler": "torch",
             "torch_profiler_dir": "./vllm_profile",
             "torch_profiler_with_stack": false}' \
             --seed 1024 \
             --served-model-name glm-5 \
             --max-model-len 200000 \
             --max-num-batched-tokens 32 \
             --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY", "cudagraph_capture_sizes":[4, 8, 12, 16,20,24,28, 32]}' \
             --additional-config '{"fuse_muls_add": true, "multistream_overlap_shared_expert": true, "recompute_scheduler_enable": true, "ascend_compilation_config": {"enable_npugraph_ex": true}}' \
             --trust-remote-code \
             --max-num-seqs 8 \
             --gpu-memory-utilization 0.92 \
             --async-scheduling \
             --quantization ascend \
             --enable-auto-tool-choice \
             --tool-call-parser glm47 \
             --reasoning-parser glm45 \
             --kv-transfer-config \
             '{"kv_connector": "MooncakeConnectorV1",
             "kv_role": "kv_consumer",
             "kv_port": "30100",
             "engine_id": "1",
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
         export VLLM_NIXL_ABORT_REQUEST_TIMEOUT=300000
            
         export TASK_QUEUE_ENABLE=1
            
         export ASCEND_RT_VISIBLE_DEVICES=$1
                     
         export VLLM_ASCEND_ENABLE_FUSED_MC2=1
         export VLLM_ASCEND_ENABLE_MLAPO=1
         export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
            
         vllm serve /root/.cache/glm5-w8a8 \
             --host 0.0.0.0 \
             --port $2 \
             --data-parallel-size $3 \
             --data-parallel-rank $4 \
             --data-parallel-address $5 \
             --data-parallel-rpc-port $6 \
             --tensor-parallel-size $7 \
             --enable-expert-parallel \
             --speculative-config '{"num_speculative_tokens": 3,  "method":"deepseek_mtp"}' \
             --profiler-config \
             '{"profiler": "torch",
             "torch_profiler_dir": "./vllm_profile",
             "torch_profiler_with_stack": false}' \
             --seed 1024 \
             --served-model-name glm-5 \
             --max-model-len 200000 \
             --max-num-batched-tokens 32 \
             --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY", "cudagraph_capture_sizes":[4, 8, 12, 16,20,24,28, 32]}' \
             --additional-config '{"fuse_muls_add": true, "multistream_overlap_shared_expert": true, "recompute_scheduler_enable": true, "ascend_compilation_config": {"enable_npugraph_ex": true}}' \
             --trust-remote-code \
             --max-num-seqs 8 \
             --gpu-memory-utilization 0.92 \
             --async-scheduling \
             --quantization ascend \
             --enable-auto-tool-choice \
             --tool-call-parser glm47 \
             --reasoning-parser glm45 \
             --kv-transfer-config \
             '{"kv_connector": "MooncakeConnectorV1",
             "kv_role": "kv_consumer",
             "kv_port": "30100",
             "engine_id": "1",
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

### Request Forwarding

To set up request forwarding, run the following script on any machine. You can get the proxy program in the repository's examples: [load_balance_proxy_server_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py)

```shell
unset http_proxy
unset https_proxy

python load_balance_proxy_server_example.py \
    --port 8000 \
    --host 0.0.0.0 \
    --prefiller-hosts \
       $node_p0_ip \
       $node_p0_ip \
       $node_p1_ip \
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

**Notice:**

Some configurations for optimization are shown below:

- `VLLM_ASCEND_ENABLE_FLASHCOMM1`: Enable FlashComm optimization to reduce communication and computation overhead on prefill node. With FlashComm enabled, layer_sharding list cannot include o_proj as an element.
- `VLLM_ASCEND_ENABLE_FUSED_MC2`: Enable following fused operators: dispatch_gmm_combine_decode and dispatch_ffn_combine operator.
- `VLLM_ASCEND_ENABLE_MLAPO`: Enable fused operator MlaPreprocessOperation.

Please refer to the following python file for further explanation and restrictions of the environment variables above: [envs.py](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/envs.py)

## Functional Verification

Once your server is started, you can query the model with input prompts:

```shell
curl http://<node0_ip>:<port>/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "glm-5",
        "prompt": "The future of AI is",
        "max_completion_tokens": 50,
        "temperature": 0
    }'
```

## Accuracy Evaluation

Here are two accuracy evaluation methods.

### Using AISBench

1. Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details.

2. After execution, you can get the result.

### Using Language Model Evaluation Harness

Not tested yet.

## Performance

### Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### Using vLLM Benchmark

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/contributing/benchmarks.html) for more details.

## Best Practices

In this chapter, we recommend best practices in prefill-decode disaggregation scenario with 1P1D architecture using 4 Atlas 800 A3 (64G × 16):

- Low-latency: We recommend setting `dp4 tp8` on prefill nodes and `dp4 tp8` on decode nodes for low latency situation.
- High-throughput: `dp4 tp8` on prefill nodes and `dp8 tp4` on decode nodes is recommended for high throughput situation.

**Notice:**
`max-model-len` and `max-num-seqs` need to be set according to the actual usage scenario. For other settings, please refer to the **[Deployment](#deployment)** chapter.

## FAQ

- **Q: How to solve ValueError: Tokenizer class TokenizersBackend does not exist or is not currently imported?**

  A: Please update the version of transformers to 5.2.0

- **Q: How to enable function calling for GLM-5?**

  A: Please add following configurations in vLLM startup command

  ```shell
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --enable-auto-tool-choice \
  ```
