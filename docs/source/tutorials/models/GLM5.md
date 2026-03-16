# GLM-5

## Introduction

[GLM-5](https://huggingface.co/zai-org/GLM-5) use a Mixture-of-Experts (MoE) architecture and targeting at complex systems engineering and long-horizon agentic tasks.

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, single-node and multi-node deployment, accuracy and performance evaluation.

## Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## Environment Preparation

### Model Weight

- `GLM-5`(BF16 version): [Download model weight](https://www.modelscope.cn/models/ZhipuAI/GLM-5).
- `GLM-5-w4a8`(Quantized version without MTP quant): [Download model weight](https://modelscope.cn/models/Eco-Tech/GLM-5-w4a8).
- `GLM-5-w4a8`(Quantized version with MTP quant): [Download model weight](https://modelscope.cn/models/Eco-Tech/GLM-5-w4a8-mtp-QuaRot).
- You can use [msmodelslim](https://gitcode.com/Ascend/msmodelslim) to quantify the model naively.

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`

### Installation

vLLM and vLLM-ascend only support GLM-5 on our main branches. you can use our glm5 docker images for inference.

:::::{tab-set}
:sync-group: install

::::{tab-item} A3 series
:sync: A3

Start the docker image on your each node.

```{code-block} bash
   :substitutions:

# Update --device according to your device (Atlas A3:/dev/davinci[0-15]).
# Update the vllm-ascend image according to your environment.
# Note you should download the weight to /root/.cache in advance.
# Update the vllm-ascend image, glm5-a3 can be replaced by: glm5;glm5-openeuler;glm5-a3-openeuler
export IMAGE=m.daocloud.io/quay.io/ascend/vllm-ascend:glm5-a3
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

export IMAGE=quay.io/ascend/vllm-ascend:glm5
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

- After install `vllm-ascend`  from source, you should upgrade vllm、vllm-ascend、transformers to main branches:

```shell
# upgrade vllm
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout 978a37c82387ce4a40aaadddcdbaf4a06fc4d590
VLLM_TARGET_DEVICE=empty pip install -v .

# upgrade vllm-ascend
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git checkout ff3a50d011dcbea08f87ebed69ff1bf156dbb01e
git submodule update --init --recursive
pip install -v .

# reinstall transformers
pip install git+https://github.com/huggingface/transformers.git
```

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
export OMP_NUM_THREADS=10
export VLLM_USE_V1=1
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
--max-model-len 66600 \
--max-num-batched-tokens 4096 \
--trust-remote-code \
--gpu-memory-utilization 0.95 \
--quantization ascend \
--enable-chunked-prefill \
--enable-prefix-caching \
--async-scheduling \
--additional-config '{"multistream_overlap_shared_expert":true}' \
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
export OMP_NUM_THREADS=10
export VLLM_USE_V1=1
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
--additional-config '{"multistream_overlap_shared_expert":true}' \
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
export OMP_NUM_THREADS=10
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=200
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
--no-enable-prefix-caching \
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
export OMP_NUM_THREADS=10
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=200
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
--no-enable-prefix-caching \
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
export OMP_NUM_THREADS=10
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=200
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
--no-enable-prefix-caching \
--gpu-memory-utilization 0.95 \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
--additional-config '{"multistream_overlap_shared_expert":true}' \
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
export OMP_NUM_THREADS=10
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=200
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
--no-enable-prefix-caching \
--gpu-memory-utilization 0.95 \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
--additional-config '{"multistream_overlap_shared_expert":true}' \
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

    1. Prefill node 0

        ```shell
        nic_name="enp48s3u1u1" # change to your own nic name
        local_ip=141.61.39.129 # change to your own ip

        export HCCL_OP_EXPANSION_MODE="AIV"

        export HCCL_IF_IP=$local_ip
        export GLOO_SOCKET_IFNAME=$nic_name
        export TP_SOCKET_IFNAME=$nic_name
        export HCCL_SOCKET_IFNAME=$nic_name

        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=10
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export VLLM_USE_V1=1
        export HCCL_BUFFSIZE=256

        export ASCEND_AGGREGATE_ENABLE=1
        export ASCEND_TRANSPORT_PRINT=1
        export ACL_OP_INIT_MODE=1
        export ASCEND_A3_ENABLE=1
        export VLLM_NIXL_ABORT_REQUEST_TIMEOUT=300000

        export ASCEND_RT_VISIBLE_DEVICES=$1

        export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
          
        export VLLM_ASCEND_ENABLE_FUSED_MC2=1
        export VLLM_ASCEND_ENABLE_MLAPO=1


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
            --additional-config '{"enable_npugraph_ex": true, "fuse_qknorm_rope": true, "fuse_muls_add":true,"multistream_overlap_shared_expert":true,"recompute_scheduler_enable" : true, "rot_path": "/mnt/share/rot.safetensors"}' \
            --max-num-batched-tokens 4096 \
            --trust-remote-code \
            --max-num-seqs 64 \
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
                                "dp_size": 4,
                                "tp_size": 8
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
        nic_name="enp48s3u1u1" # change to your own nic name
        local_ip=141.61.39.133 # change to your own ip

        export HCCL_OP_EXPANSION_MODE="AIV"

        export HCCL_IF_IP=$local_ip
        export GLOO_SOCKET_IFNAME=$nic_name
        export TP_SOCKET_IFNAME=$nic_name
        export HCCL_SOCKET_IFNAME=$nic_name

        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=10
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export VLLM_USE_V1=1
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
        export VLLM_ASCEND_ENABLE_MLAPO=1


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
            --additional-config '{"enable_npugraph_ex": true, "fuse_qknorm_rope": true, "fuse_muls_add":true,"multistream_overlap_shared_expert":true,"recompute_scheduler_enable" : true, "rot_path": "/mnt/share/rot.safetensors"}' \
            --max-num-batched-tokens 4096 \
            --trust-remote-code \
            --max-num-seqs 64 \
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
                                "dp_size": 4,
                                "tp_size": 8
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
        nic_name="enp48s3u1u1" # change to your own nic name
        local_ip=141.61.39.101 # change to your own ip
    
        export HCCL_OP_EXPANSION_MODE="AIV"
    
        export HCCL_IF_IP=$local_ip
        export GLOO_SOCKET_IFNAME=$nic_name
        export TP_SOCKET_IFNAME=$nic_name
        export HCCL_SOCKET_IFNAME=$nic_name
    
        #Mooncake
        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=10
    
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export VLLM_USE_V1=1
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
            --additional-config '{"enable_npugraph_ex": true, "fuse_qknorm_rope": true, "fuse_muls_add":true,"multistream_overlap_shared_expert":true,"recompute_scheduler_enable" : true, "rot_path": "/mnt/share/rot.safetensors"}' \
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
                                "dp_size": 4,
                                "tp_size": 8
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
         nic_name="enp48s3u1u1" # change to your own nic name
         local_ip=141.61.39.109 # change to your own ip
            
         export HCCL_OP_EXPANSION_MODE="AIV"
            
         export HCCL_IF_IP=$local_ip
         export GLOO_SOCKET_IFNAME=$nic_name
         export TP_SOCKET_IFNAME=$nic_name
         export HCCL_SOCKET_IFNAME=$nic_name
            
         #Mooncake
         export OMP_PROC_BIND=false
         export OMP_NUM_THREADS=10
            
         export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
         export VLLM_USE_V1=1
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
             --additional-config '{"enable_npugraph_ex": true, "fuse_qknorm_rope": true, "fuse_muls_add":true,"multistream_overlap_shared_expert":true,"recompute_scheduler_enable" : true, "rot_path": "/mnt/share/rot.safetensors"}' \
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
                                 "dp_size": 4,
                                 "tp_size": 8
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
         nic_name="enp48s3u1u1" # change to your own nic name
         local_ip=141.61.39.185 # change to your own ip
            
         export HCCL_OP_EXPANSION_MODE="AIV"
            
         export HCCL_IF_IP=$local_ip
         export GLOO_SOCKET_IFNAME=$nic_name
         export TP_SOCKET_IFNAME=$nic_name
         export HCCL_SOCKET_IFNAME=$nic_name
            
         #Mooncake
         export OMP_PROC_BIND=false
         export OMP_NUM_THREADS=10
            
         export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
         export VLLM_USE_V1=1
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
             --additional-config '{"enable_npugraph_ex": true, "fuse_qknorm_rope": true, "fuse_muls_add":true,"multistream_overlap_shared_expert":true,"recompute_scheduler_enable" : true, "rot_path": "/mnt/share/rot.safetensors"}' \
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
                                 "dp_size": 4,
                                 "tp_size": 8
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
         nic_name="enp48s3u1u1" # change to your own nic name
         local_ip=141.61.39.173 # change to your own ip
            
         export HCCL_OP_EXPANSION_MODE="AIV"
            
         export HCCL_IF_IP=$local_ip
         export GLOO_SOCKET_IFNAME=$nic_name
         export TP_SOCKET_IFNAME=$nic_name
         export HCCL_SOCKET_IFNAME=$nic_name
            
         #Mooncake
         export OMP_PROC_BIND=false
         export OMP_NUM_THREADS=10
            
         export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
         export VLLM_USE_V1=1
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
             --additional-config '{"enable_npugraph_ex": true, "fuse_qknorm_rope": true, "fuse_muls_add":true,"multistream_overlap_shared_expert":true,"recompute_scheduler_enable" : true, "rot_path": "/mnt/share/rot.safetensors"}' \
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
                                 "dp_size": 4,
                                 "tp_size": 8
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
python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 2 --dp-rank-start 0 --dp-address 141.61.39.129 --dp-rpc-port 10521 --vllm-start-port 6700
```

2. Prefill node 1

```shell
# change ip to your own
python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 2 --dp-rank-start 2 --dp-address 141.61.39.129 --dp-rpc-port 10521 --vllm-start-port 6700
```

3. Decode node 0

```shell
# change ip to your own
python launch_online_dp.py --dp-size 16 --tp-size 4 --dp-size-local 4 --dp-rank-start 0 --dp-address 141.61.39.101 --dp-rpc-port 10523 --vllm-start-port 6721
```

4. Decode node 1

```shell
# change ip to your own
python launch_online_dp.py --dp-size 16 --tp-size 4 --dp-size-local 4 --dp-rank-start 4 --dp-address 141.61.39.101 --dp-rpc-port 10523 --vllm-start-port 6721
```

5. Decode node 2

```shell
# change ip to your own
python launch_online_dp.py --dp-size 16 --tp-size 4 --dp-size-local 4 --dp-rank-start 8 --dp-address 141.61.39.101 --dp-rpc-port 10523 --vllm-start-port 6721
```

6. Decode node 3

```shell
# change ip to your own
python launch_online_dp.py --dp-size 16 --tp-size 4 --dp-size-local 4 --dp-rank-start 12 --dp-address 141.61.39.101 --dp-rpc-port 10523 --vllm-start-port 6721
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
       141.61.39.129 \
       141.61.39.129 \
       141.61.39.133 \
       141.61.39.133 \
    --prefiller-ports \
       6700 6701 \
       6700 6701 \
    --decoder-hosts \
      141.61.39.101 \
      141.61.39.101 \
      141.61.39.101 \
      141.61.39.101 \
      141.61.39.109 \
      141.61.39.109 \
      141.61.39.109 \
      141.61.39.109 \
      141.61.39.185 \
      141.61.39.185 \
      141.61.39.185 \
      141.61.39.185 \
      141.61.39.173 \
      141.61.39.173 \
      141.61.39.173 \
      141.61.39.173 \
    --decoder-ports \
      6721 6722 6723 6724 \
      6721 6722 6723 6724 \
      6721 6722 6723 6724 \
      6721 6722 6723 6724      
```

## Accuracy Evaluation

Here are two accuracy evaluation methods.

### Using AISBench

1. Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details.

2. After execution, you can get the result.

### Using Language Model Evaluation Harness

Not test yet.

## Performance

### Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### Using vLLM Benchmark

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/contributing/benchmarks.html) for more details.
