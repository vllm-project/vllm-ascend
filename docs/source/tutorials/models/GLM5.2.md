# GLM-5.2

## 1 Introduction

[GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) uses a Mixture-of-Experts (MoE) architecture and targets complex systems engineering and long-horizon agentic tasks.

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, single-node and multi-node deployment, accuracy and performance evaluation.

## 2 Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## 3 Prerequisites

### 3.1 Model Weight

- `GLM-5.2-w8a8c8`: requires 2 Atlas 800 A3 (128GB × 8) node.[Download model weight](https://modelers.cn/models/Eco-Tech/GLM-5.2-w8a8c8).
- `GLM-5.2-w4a8c8` (experimental): requires 1 Atlas 800 A3 (128GB × 8) node or 2 Atlas 800 A2 (64GB × 8) node. This experimental feature has known accuracy issues in Prefill-Decode (PD) disaggregation scenarios. [Download model weight](https://www.modelscope.cn/models/Eco-Tech/GLM-5.2-w4a8c8).
- You can use [msmodelslim](https://gitcode.com/Ascend/msmodelslim) to quantize the model directly.

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`

### 3.2 Verify Multi-node Communication (Optional)

If you want to deploy multi-node environment, you need to verify multi-node communication according to [verify multi-node communication environment](../../installation.md#verify-multi-node-communication).

## 4 Installation

### 4.1 Docker Image Installation

- You can use our official docker image to run GLM-5.2 directly.

:::::{tab-set}
:sync-group: install

::::{tab-item} A3 series
:sync: A3

Start the docker image on each node.

```{code-block} bash
   :substitutions:

export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|-a3
docker run --rm \
    --name vllm-ascend \
    --shm-size=512g \
    --net=host \
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
    -v /root/.cache:/root/.cache \
    -it $IMAGE bash
```

::::
::::{tab-item} A2 series
:sync: A2

Start the docker image on each of your nodes.

```{code-block} bash
   :substitutions:

export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
docker run --rm \
    --name vllm-ascend \
    --shm-size=512g \
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
    -v /root/.cache:/root/.cache \
    -it $IMAGE bash
```

If you want to deploy multi-node environment, you need to set up environment on each node.

::::
:::::

### 4.2 Source Code Installation

If you don't want to use the docker image as above, you can also build all from source:

- Install `vllm-ascend` from source, refer to [installation](../../installation.md).

## 5 Deployment

The deployment scenarios validated for this release are organized by context window size (below 1M / 1M), hardware (Atlas 800 A3 / A2), and deployment mode (single-node, multi-node co-located, Prefill-Decode disaggregation). All startup scripts below are the verified reference commands; key parameters are explained after each scenario.

### 5.1 Context Below 1M

#### 5.1.1 Atlas 800 A3

##### 5.1.1.1 Single-Node Deployment

- Quantized model `GLM-5.2-w4a8c8` (experimental, see [Model Weight](#31-model-weight)) can be deployed on 1 Atlas 800 A3 (128GB × 8) .

Run the following script to execute online inference.

```{code-block} bash
   :substitutions:
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_TRANSFER_TIMEOUT=600
export HCCL_EXEC_TIMEOUT=3600
export HCCL_CONNECT_TIMEOUT=3600
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=0

vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.2-w4a8c8 \
--host 0.0.0.0 \
--port 8077 \
--api-server-count 1 \
--data-parallel-size 2 \
--enable-expert-parallel \
--tensor-parallel-size 8 \
--seed 1024 \
--served-model-name glm-5 \
--tool-call-parser glm47 \
--reasoning-parser glm45 \
--enable-auto-tool-choice \
--max-num-seqs 12 \
--max-model-len 135000 \
--max-num-batched-tokens 8192 \
--trust-remote-code \
--gpu-memory-utilization 0.92 \
--quantization ascend \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
--additional-config '{"enable_dsa_cp": true, "enable_sparse_li_c8": true,"enable_balance_scheduling": true,"multistream_overlap_shared_expert":true}' \
--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp","enforce_eager":true}'

```

Key Parameter Descriptions:

Only the key parameters specific to this model/scenario are described below. `max-model-len` and `max-num-seqs` need to be set according to the actual usage scenario.

**Model-specific parameters:**

- `--enable-expert-parallel`: Must be enabled for the MoE architecture of GLM-5.2.
- `--quantization ascend`: Enables Ascend quantization for the w4a8c8 quantized weights.
- `--data-parallel-size 2` / `--tensor-parallel-size 8`: DP2 TP8 parallelism layout, recommended to balance memory capacity and compute efficiency for the w4a8c8 weights. For low-latency scenarios, use `dp1tp16` instead and turn off expert parallel, at the cost of lower throughput.
- `--tool-call-parser glm47` / `--reasoning-parser glm45` / `--enable-auto-tool-choice`: Enable function calling for GLM-5.2.
- `--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}'`: Enables Multi-Token Prediction (MTP) speculative decoding with the DeepSeek-style MTP draft head of GLM-5.2. `num_speculative_tokens` (3-5) controls how many tokens are speculated per step; `enforce_eager: true` is required because GLM-5.2 does not support graph-mode speculative decoding.
- `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'`: Enables graph capture for the decode phase only, improving decode performance by reducing kernel launch overhead.
- `VLLM_ASCEND_ENABLE_FUSED_MC2=0`: Disables the fused `dispatch_ffn_combine`/`mega_moe` operators in this scenario because they conflict with `multistream_overlap_shared_expert`; turn them on in multi-node scenarios where they are beneficial.

**`--additional-config` fields (Ascend-specific optimizations):**

- `"enable_dsa_cp": true`: Enables DSA context parallelism to accelerate long-context prefill. Since v0.21.0, DSA-CP is decoupled from FlashComm1 and must be enabled explicitly. With DSA-CP enabled, `layer_sharding` cannot include `o_proj`.
- `"enable_sparse_li_c8": true`: Sparse attention optimizations of the C8 quantized model. `enable_sparse_li_c8` accelerates the layer-index (LI) sparse attention and is recommended to keep `true`; If the GPU memory is insufficient due to a long sequence length, you are advised to enable `enable_sparse_sfa_c8`.
- `"enable_balance_scheduling": true`: Balance scheduling improves output throughput and reduces TPOT in the v1 scheduler. It replaces the deprecated environment variable `VLLM_ASCEND_BALANCE_SCHEDULING`; TTFT may degrade in some scenarios, and it is not recommended when Prefill-Decode is separated.
- `"multistream_overlap_shared_expert": true`: Overlaps shared-expert computation on an additional stream, improving decode efficiency.

##### 5.1.1.2 Multi-Node Co-Located Deployment

**GLM-5.2-w8a8c8 (198K context, dual-node co-located):**

`GLM-5.2-w8a8c8` can be deployed on 2 Atlas 800 A3 (128GB × 8) for the 198K high-throughput scenario (`DP8 TP4`, 4 DP ranks per node).

Run the following scripts on two nodes respectively.

**node 0**

```{code-block} bash
    :substitutions:
    nic_name="xxxx" # change to your own nic name
    local_ip="xxxx" # change to your own ip

    # The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
    node0_ip="xxxx"

    export HCCL_OP_EXPANSION_MODE="AIV"
    export HCCL_IF_IP=$local_ip
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export HCCL_SOCKET_IFNAME=$nic_name
    export HCCL_TRANSFER_TIMEOUT=600
    export HCCL_EXEC_TIMEOUT=3600
    export HCCL_CONNECT_TIMEOUT=3600
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=1
    export HCCL_BUFFSIZE=400
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export VLLM_ASCEND_ENABLE_MLAPO=1
    export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
    export VLLM_ASCEND_ENABLE_FUSED_MC2=1

    vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.2-w8a8c8 \
    --host 0.0.0.0 \
    --port 8077 \
    --api-server-count 1 \
    --data-parallel-size 8 \
    --data-parallel-start-rank 0 \
    --data-parallel-size-local 4 \
    --data-parallel-address $node0_ip \
    --data-parallel-rpc-port 12980 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --seed 1024 \
    --served-model-name glm-5 \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --enable-auto-tool-choice \
    --max-num-seqs 6 \
    --max-model-len 202752 \
    --max-num-batched-tokens 4096 \
    --trust-remote-code \
    --gpu-memory-utilization 0.92 \
    --quantization ascend \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --async-scheduling \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_dsa_cp": true, "enable_sparse_sfa_c8": true, "enable_sparse_li_c8": true, "enable_balance_scheduling": true, "fuse_muls_add": true}' \
    --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp","enforce_eager":true}'
```

**node 1**

```{code-block} bash
    :substitutions:
    nic_name="xxxx" # change to your own nic name
    local_ip="xxxx" # change to your own ip

    # The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
    node0_ip="xxxx"

    export HCCL_OP_EXPANSION_MODE="AIV"
    export HCCL_IF_IP=$local_ip
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export HCCL_SOCKET_IFNAME=$nic_name
    export HCCL_TRANSFER_TIMEOUT=600
    export HCCL_EXEC_TIMEOUT=3600
    export HCCL_CONNECT_TIMEOUT=3600
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=1
    export HCCL_BUFFSIZE=400
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export VLLM_ASCEND_ENABLE_MLAPO=1
    export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
    export VLLM_ASCEND_ENABLE_FUSED_MC2=1

    vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.2-w8a8c8 \
    --host 0.0.0.0 \
    --port 8077 \
    --headless \
    --data-parallel-size 8 \
    --data-parallel-start-rank 4 \
    --data-parallel-size-local 4 \
    --data-parallel-address $node0_ip \
    --data-parallel-rpc-port 12980 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --seed 1024 \
    --served-model-name glm-5 \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --enable-auto-tool-choice \
    --max-num-seqs 6 \
    --max-model-len 202752 \
    --max-num-batched-tokens 4096 \
    --trust-remote-code \
    --gpu-memory-utilization 0.92 \
    --quantization ascend \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --async-scheduling \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_dsa_cp": true, "enable_sparse_sfa_c8": true, "enable_sparse_li_c8": true, "enable_balance_scheduling": true, "fuse_muls_add": true}' \
    --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp","enforce_eager":true}'
```

**Notice:**

- `VLLM_ASCEND_ENABLE_FUSED_MC2=1` conflicts with `"multistream_overlap_shared_expert": true` — the runtime automatically disables `multistream_overlap_shared_expert` when fused MC2 is enabled.
- When testing with a prefix cache hit rate > 0, keep `--enable-prefix-caching` (as in the script above); when the hit rate is 0, replace it with `--no-enable-prefix-caching`.

##### 5.1.1.3 Prefill-Decode Disaggregation

We'd like to show the deployment guide of `GLM-5.2` on multi-node environment with Prefill-Decode (PD) disaggregation for better performance.

In the PD disaggregation scenario, Mooncake is used as the KV cache transfer connector between the prefill and decode nodes. Please refer to [KV Cache Pool (Ascend Store) Deployment Guide](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/user_guide/feature_guide/kv_pool.md) for the Mooncake configuration.

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

**GLM-5.2-w8a8c8 (198K context, PD disaggregation with PP2):**

`GLM-5.2-w8a8c8` PD disaggregation can be deployed on 4 Atlas 800 A3 (128GB × 8): 2 prefill nodes (`PP2 TP16`, 78 layers partitioned as `41/37`, one PP rank per node) and 2 decode nodes (`DP8 TP4`, 4 DP ranks per node).

**Prefill node 0** (`run_dp_template.sh`): `node_rank=0`, engine port `9081` (PP master node, `--master-addr` uses `$local_ip`).

```shell
        nic_name="xxxx" # change to your own nic name
        local_ip="xxxx" # change to your own ip
        # pp=2
        export VLLM_PP_LAYER_PARTITION="41,37"
        node_rank=0

        export VLLM_ASCEND_ENABLE_FUSED_MC2=1
        export HCCL_OP_EXPANSION_MODE="AIV"

        export HCCL_IF_IP=$local_ip
        export GLOO_SOCKET_IFNAME=$nic_name
        export TP_SOCKET_IFNAME=$nic_name
        export HCCL_SOCKET_IFNAME=$nic_name
        export HCCL_TRANSFER_TIMEOUT=600
        export HCCL_EXEC_TIMEOUT=3600
        export HCCL_CONNECT_TIMEOUT=3600

        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=1
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export HCCL_BUFFSIZE=400

        export ACL_OP_INIT_MODE=1
        export ASCEND_A3_ENABLE=1

        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

        export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

        vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.2-w8a8c8 \
            --host 0.0.0.0 \
            --port 9081 \
            --pipeline-parallel-size 2 \
            --distributed-executor-backend mp \
            --master-addr $local_ip \
            --master-port 7060 \
            --nnodes 2 \
            --node-rank 0 \
            --tensor-parallel-size 16 \
            --enable-expert-parallel \
            --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp","enforce_eager":true}' \
            --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' \
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

**Prefill node 1** (`run_dp_template.sh`): `node_rank=1`, non-master node running with `--headless` (no API server). Both prefill nodes form a single PP2 engine, so `--master-addr` points to prefill node 0 and the KV transfer `engine_id` stays `0`.

```shell
        nic_name="xxxx" # change to your own nic name
        local_ip="xxxx" # change to your own ip
        node_p0_ip="xxxx" # ip of prefill node 0 (PP master node)
        # pp=2
        export VLLM_PP_LAYER_PARTITION="41,37"
        node_rank=1

        export VLLM_ASCEND_ENABLE_FUSED_MC2=1
        export HCCL_OP_EXPANSION_MODE="AIV"

        export HCCL_IF_IP=$local_ip
        export GLOO_SOCKET_IFNAME=$nic_name
        export TP_SOCKET_IFNAME=$nic_name
        export HCCL_SOCKET_IFNAME=$nic_name
        export HCCL_TRANSFER_TIMEOUT=600
        export HCCL_EXEC_TIMEOUT=3600
        export HCCL_CONNECT_TIMEOUT=3600

        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=1
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export HCCL_BUFFSIZE=400

        export ACL_OP_INIT_MODE=1
        export ASCEND_A3_ENABLE=1

        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

        export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

        vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.2-w8a8c8 \
            --host 0.0.0.0 \
            --pipeline-parallel-size 2 \
            --distributed-executor-backend mp \
            --master-addr $node_p0_ip \
            --master-port 7060 \
            --nnodes 2 \
            --node-rank 1 \
            --headless \
            --tensor-parallel-size 16 \
            --enable-expert-parallel \
            --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp","enforce_eager":true}' \
            --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' \
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

**Decode node 0** (`run_dp_template.sh`): hosts DP ranks 0–3. The template launches one instance per DP rank on the node; positional parameters: `$1` = visible devices, `$2` = engine port, `$3` = data-parallel-size, `$4` = data-parallel-rank, `$5` = data-parallel-address, `$6` = data-parallel-rpc-port, `$7` = tensor-parallel-size. Prepare `run_dp_template.sh` on decode node 0 with the content below.

```shell
        nic_name="xxxx" # change to your own nic name
        local_ip="xxxx" # change to your own ip

        # 每个 DP rank 使用独立的 engine_id,避免 KV 路由混淆
        # $4 = data-parallel-rank; 节点内 rank 偏移 0-3 → engine_id 100-103
        ENGINE_ID=$((100 + $4))

        export HCCL_OP_EXPANSION_MODE="AIV"

        export HCCL_IF_IP=$local_ip
        export GLOO_SOCKET_IFNAME=$nic_name
        export TP_SOCKET_IFNAME=$nic_name
        export HCCL_SOCKET_IFNAME=$nic_name
        export HCCL_TRANSFER_TIMEOUT=600
        export HCCL_EXEC_TIMEOUT=3600
        export HCCL_CONNECT_TIMEOUT=3600

        #Mooncake
        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=1

        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export HCCL_BUFFSIZE=256

        export ACL_OP_INIT_MODE=1
        export ASCEND_A3_ENABLE=1
        export TASK_QUEUE_ENABLE=1
        export ASCEND_RT_VISIBLE_DEVICES=$1

        export VLLM_ASCEND_ENABLE_MLAPO=1
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
        export VLLM_ASCEND_ENABLE_FUSED_MC2=1

        vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.2-w8a8c8 \
            --host 0.0.0.0 \
            --port $2 \
            --data-parallel-size $3 \
            --data-parallel-rank $4 \
            --data-parallel-address $5 \
            --data-parallel-rpc-port $6 \
            --tensor-parallel-size $7 \
            --enable-expert-parallel \
            --seed 1024 \
            --served-model-name glm-5 \
            --max-model-len 202752 \
            --max-num-batched-tokens 164 \
            --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
            --speculative-config '{"num_speculative_tokens": 3,  "method":"deepseek_mtp","enforce_eager":true}' \
            --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' \
            --additional-config '{"fuse_muls_add":true, "recompute_scheduler_enable":true, "multistream_overlap_shared_expert":true, "enable_sparse_sfa_c8": true, "enable_sparse_li_c8": true}' \
            --trust-remote-code \
            --max-num-seqs 32 \
            --gpu-memory-utilization 0.92 \
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

**Decode node 1** (`run_dp_template.sh`): hosts DP ranks 4–7. Prepare `run_dp_template.sh` on decode node 1 with the content below.

```shell
        nic_name="xxxx" # change to your own nic name
        local_ip="xxxx" # change to your own ip

        # 每个 DP rank 使用独立的 engine_id,避免 KV 路由混淆
        # $4 = data-parallel-rank; 节点内 rank 偏移 0-3 → engine_id 100-103
        ENGINE_ID=$((100 + $4))

        export HCCL_OP_EXPANSION_MODE="AIV"

        export HCCL_IF_IP=$local_ip
        export GLOO_SOCKET_IFNAME=$nic_name
        export TP_SOCKET_IFNAME=$nic_name
        export HCCL_SOCKET_IFNAME=$nic_name
        export HCCL_TRANSFER_TIMEOUT=600
        export HCCL_EXEC_TIMEOUT=3600
        export HCCL_CONNECT_TIMEOUT=3600

        #Mooncake
        export OMP_PROC_BIND=false
        export OMP_NUM_THREADS=1

        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export HCCL_BUFFSIZE=256

        export ACL_OP_INIT_MODE=1
        export ASCEND_A3_ENABLE=1
        export TASK_QUEUE_ENABLE=1
        export ASCEND_RT_VISIBLE_DEVICES=$1

        export VLLM_ASCEND_ENABLE_MLAPO=1
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
        export VLLM_ASCEND_ENABLE_FUSED_MC2=1

        vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.2-w8a8c8 \
            --host 0.0.0.0 \
            --port $2 \
            --data-parallel-size $3 \
            --data-parallel-rank $4 \
            --data-parallel-address $5 \
            --data-parallel-rpc-port $6 \
            --tensor-parallel-size $7 \
            --enable-expert-parallel \
            --seed 1024 \
            --served-model-name glm-5 \
            --max-model-len 202752 \
            --max-num-batched-tokens 164 \
            --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
            --speculative-config '{"num_speculative_tokens": 3,  "method":"deepseek_mtp","enforce_eager":true}' \
            --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' \
            --additional-config '{"fuse_muls_add":true, "recompute_scheduler_enable":true, "multistream_overlap_shared_expert":true, "enable_sparse_sfa_c8": true, "enable_sparse_li_c8": true}' \
            --trust-remote-code \
            --max-num-seqs 32 \
            --gpu-memory-utilization 0.92 \
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

Once the preparation is done, start the server on each node:

1. Prefill node 0:

    ```shell
    bash run_dp_template.sh
    ```

2. Prefill node 1:

    ```shell
    bash run_dp_template.sh
    ```

3. Decode node 0:

    ```shell
    # change ip to your own
    python launch_online_dp.py --dp-size 8 --tp-size 4 --pp-size 1 --dp-size-local 4 --dp-rank-start 0 --dp-address $node_d0_ip --dp-rpc-port 12321 --vllm-start-port 8000
    ```

4. Decode node 1:

    ```shell
    # change ip to your own
    python launch_online_dp.py --dp-size 8 --tp-size 4 --pp-size 1 --dp-size-local 4 --dp-rank-start 4 --dp-address $node_d0_ip --dp-rpc-port 12321 --vllm-start-port 8000
    ```

To set up request forwarding, run the following script on any machine. You can get the proxy program in the repository's examples: [load_balance_proxy_server_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py)

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

**Notice:**

- When testing with a prefix cache hit rate > 0, add `--enable-prefix-caching` on the prefill nodes (as in the script above); when the hit rate is 0, use `--no-enable-prefix-caching` instead.
- `"recompute_scheduler_enable"` is set to `false` on prefill nodes and `true` on decode nodes in this scenario.

Key Parameter Descriptions (in addition to [Single-Node Deployment](#5111-single-node-deployment) and [Multi-Node Co-Located Deployment](#5112-multi-node-co-located-deployment)):

**`launch_online_dp.py` parameters:**

|Parameter|Type|Required|Default|Description|
|---------|----|--------|-------|-----------|
|`--dp-size`|int|Yes|-|Data parallel size (total number of DP ranks across all nodes).|
|`--tp-size`|int|No|1|Tensor parallel size within each DP rank.|
|`--pp-size`|int|No|1|Pipeline parallel size within each DP rank.|
|`--dp-size-local`|int|No|(same as `--dp-size`)|Number of DP ranks on the current node. If not set, defaults to `--dp-size`.|
|`--dp-rank-start`|int|No|0|Starting rank offset for data parallel ranks on this node.|
|`--dp-address`|str|Yes|-|IP address of the data parallel master node (node 0).|
|`--dp-rpc-port`|str|No|12321|RPC port for data parallel master communication.|
|`--vllm-start-port`|int|No|8000|Starting port for each vLLM engine instance on this node. Each DP rank's engine port = `vllm_start_port` + local rank index.|

**Prefill node-specific configurations (PP2):**

- `VLLM_PP_LAYER_PARTITION="41,37"` / `--pipeline-parallel-size 2` / `--node-rank`: The 78 layers are partitioned as `41/37` across 2 prefill nodes (PP2). Set `node_rank=0` on prefill node 0 and `node_rank=1` on prefill node 1.
- `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`: Enables FlashComm optimization to reduce communication and computation overhead on prefill nodes. With FlashComm enabled, `layer_sharding` cannot include `o_proj` as an element.
- `VLLM_ASCEND_ENABLE_FUSED_MC2=1`: Enables the `dispatch_ffn_combine`/`mega_moe` fused operators. Note: fused MC2 conflicts with `multistream_overlap_shared_expert`.
- `ACL_OP_INIT_MODE=1` / `ASCEND_A3_ENABLE=1`: A3-specific optimizations for operator initialization and communication aggregation.
- `--speculative-config '{"num_speculative_tokens": 1, ...}'`: Minimal MTP speculation during prefill (decode nodes use a higher count, see below).
- `--additional-config '{"recompute_scheduler_enable": false, ...}'`: The recompute scheduler is disabled on prefill nodes in this scenario.

**Decode node-specific configurations:**

- `--max-num-batched-tokens 164`: Small batch token limit on decode nodes — decode processes one token per sequence per step, so batch tokens should be close to `max-num-seqs`.
- `--speculative-config '{"num_speculative_tokens": 3, ...}'`: MTP speculation count on decode nodes to maximize decode throughput.
- `--additional-config '{"recompute_scheduler_enable": true}'`: Enables the recomputation scheduler. When the decode node KV cache is insufficient, requests are sent back to the prefill node to recompute the KV cache.

**Mooncake KV transfer configuration (`--kv-transfer-config`):**

- `"kv_connector": "MooncakeConnectorV1"`: Uses Mooncake as the KV cache transfer connector between prefill and decode nodes.
- `"kv_role": "kv_producer"` / `"kv_consumer"`: `kv_producer` on prefill nodes, `kv_consumer` on decode nodes.
- `"kv_port"`: Port for Mooncake KV transfer communication. Use different port ranges for prefill (`30000`) and decode (`30200`) node groups.
- `"use_ascend_direct": true`: Enables Ascend direct transfer for KV cache, reducing latency.
- `"prefill"` / `"decode"` sections: Specify the parallelism of the prefill and decode node groups respectively. These must match the actual deployment topology (`prefill: pp_size 2, tp_size 16, pp_layer_partition 41,37`, `decode: dp_size 8, tp_size 4`).

**Request forwarding (proxy):**

- The proxy command maps every prefill engine endpoint (1 prefiller host/port) and every decode engine endpoint (8 decoder hosts/ports) to a single entry point on port `9000`.
- The proxy program can be found in [load_balance_proxy_server_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py).

Please refer to [envs.py](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/envs.py) for further explanation and restrictions of the environment variables above.

#### 5.1.2 Atlas 800 A2

##### 5.1.2.1 Multi-Node Co-Located Deployment

- `GLM-5.2-w4a8c8` (experimental): can be deployed on 2 Atlas 800 A2 (64GB × 8). A single Atlas 800 A2 node (8 × 64GB) cannot fit the w4a8c8 weights, so the 2-node deployment is the minimum configuration for the A2 series.

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
    export VLLM_RPC_TIMEOUT=360000
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
    export HCCL_EXEC_TIMEOUT=200
    export HCCL_CONNECT_TIMEOUT=120
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=10
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export ACL_OP_INIT_MODE=1
    #export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
    #export USE_MULTI_GROUPS_KV_CACHE=1
    #export USE_MULTI_BLOCK_POOL=1
    export TASK_QUEUE_ENABLE=1
    export CPU_AFFINITY_CONF=1
    export VLLM_ENGINE_READY_TIMEOUT_S=1200

    vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.2-w4a8c8 \
    --max_model_len 40000 \
    --max-num-batched-tokens 4096 \
    --served-model-name glm-52 \
    --seed 1024 \
    --gpu-memory-utilization 0.95 \
    --api-server-count 1 \
    --max-num-seqs 16 \
    --data-parallel-size 2 \
    --data-parallel-size-local 1 \
    --data-parallel-address $node0_ip \
    --data-parallel-rpc-port 13389 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --quantization ascend \
    --port 7000 \
    --safetensors-load-strategy 'prefetch' \
    --block-size 128 \
    --async-scheduling \
    --additional-config '{"multistream_overlap_shared_expert": true}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
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
    export VLLM_RPC_TIMEOUT=360000
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
    export HCCL_EXEC_TIMEOUT=200
    export HCCL_CONNECT_TIMEOUT=120
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=10
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export ACL_OP_INIT_MODE=1
    #export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
    #export USE_MULTI_GROUPS_KV_CACHE=1
    #export USE_MULTI_BLOCK_POOL=1
    export TASK_QUEUE_ENABLE=1
    export CPU_AFFINITY_CONF=1
    export VLLM_ENGINE_READY_TIMEOUT_S=1200

    vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.2-w4a8c8 \
    --max_model_len 40000 \
    --max-num-batched-tokens 4096 \
    --served-model-name glm-52 \
    --seed 1024 \
    --gpu-memory-utilization 0.95 \
    --max-num-seqs 16 \
    --headless \
    --data-parallel-size 2 \
    --data-parallel-size-local 1 \
    --data-parallel-start-rank 1 \
    --data-parallel-address $node0_ip \
    --data-parallel-rpc-port 13389 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --quantization ascend \
    --port 7000 \
    --safetensors-load-strategy 'prefetch' \
    --block-size 128 \
    --async-scheduling \
    --additional-config '{"multistream_overlap_shared_expert": true}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}'
```

Key Parameter Descriptions (in addition to [Single-Node Deployment](#5111-single-node-deployment) and [Multi-Node Co-Located Deployment](#5112-multi-node-co-located-deployment)):

The A2 series uses a different optimization stack than A3: FlashComm1 and DSA-CP are not used here.

**A2-specific environment variables:**

- `CPU_AFFINITY_CONF=1`: Enables CPU core affinity binding for worker processes.
- `ACL_OP_INIT_MODE=1`: ACL operator initialization mode to speed up operator compilation.
- `VLLM_RPC_TIMEOUT=360000` / `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000` / `HCCL_EXEC_TIMEOUT=200` / `HCCL_CONNECT_TIMEOUT=120` / `VLLM_ENGINE_READY_TIMEOUT_S=1200`: Timeout settings for multi-node startup and model execution on the slower A2 platform. Increase them if the engine fails to become ready in time.

##### 5.1.2.2 Prefill-Decode Disaggregation

On Atlas 800 A2, where each node exposes 8 cards, the same global P/D topology (Prefill `DP4 TP8`, Decode `DP8 TP4`) is split across 8 nodes: 4 prefill nodes hosting 1 DP rank each (8 cards per rank), and 4 decode nodes hosting 2 DP ranks each (4 cards per rank). The `launch_online_dp.py` above is reused as-is. The prefill side enables FlashComm1 and DSA CP; the decode side enables MLAPO and `DYNAMIC_EPLB` with a `FULL_DECODE_ONLY` graph. Both sides enable prefix caching and MTP (`num_speculative_tokens=1` on prefill, `3` on decode). All IPs, NIC names, ports and weight paths below are placeholders.

**Note:** `GLM-5.2-w4a8c8` is an experimental feature with known accuracy issues in Prefill-Decode (PD) disaggregation scenarios (see [Model Weight](#31-model-weight)).

Please refer to the [KV Cache Pool (Ascend Store) Deployment Guide](https://docs.vllm.ai/projects/ascend/zh-cn/latest/user_guide/feature_guide/kv_pool.html) for the KV Cache Pool startup method and the Mooncake configuration file.

`run_dp_template.sh` for the prefill nodes:

```bash
#!/usr/bin/bash
nic_name="<NIC_NAME>"
local_ip="<CURRENT_NODE_IP>"

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_MLAPO=1
export HCCL_BUFFSIZE=256
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES=$1
export LD_LIBRARY_PATH=/usr/local/python3.11.10/lib:/usr/local/lib:$LD_LIBRARY_PATH
export ASCEND_AGGREGATE_ENABLE=1
export ASCEND_TRANSPORT_PRINT=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export PYTHONHASHSEED=0
export MOONCAKE_CONFIG_PATH=<MOONCAKE_CONFIG_PATH>
export HCCL_INTRA_ROCE_ENABLE=1
export ACL_OP_INIT_MODE=1

vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.2-w4a8c8 \
    --host 0.0.0.0 \
    --port $2 \
    --data-parallel-size $3 \
    --data-parallel-rank $4 \
    --data-parallel-address $5 \
    --data-parallel-rpc-port $6 \
    --tensor-parallel-size $7 \
    --enable-expert-parallel \
    --enable-prefix-caching \
    --seed 1024 \
    --enable-chunked-prefill \
    --served-model-name glm-5 \
    --async-scheduling \
    --max-model-len 256000 \
    --max-num-batched-tokens 8192 \
    --trust-remote-code \
    --max-num-seqs 256 \
    --gpu-memory-utilization 0.95 \
    --safetensors-load-strategy prefetch \
    --quantization ascend \
    --enforce-eager \
    --enable-auto-tool-choice \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --kv-transfer-config \
    '{
    "kv_connector": "MultiConnector",
    "kv_role": "kv_producer",
    "kv_load_failure_policy": "recompute",
    "kv_connector_extra_config": {
        "connectors": [
            {
                "kv_connector": "MooncakeConnectorV1",
                "kv_role": "kv_producer",
                "kv_port": "30000",
                "kv_connector_extra_config": {
                    "prefill": {
                        "dp_size": 4,
                        "tp_size": 8
                    },
                    "decode": {
                        "dp_size": 8,
                        "tp_size": 4
                    }
                }
            },
            {
                "kv_connector": "AscendStoreConnector",
                "kv_role": "kv_producer",
                "kv_connector_extra_config": {
                    "lookup_rpc_port":"0",
                    "backend": "mooncake"
                }
            }
        ]
    }
    }' \
    --additional-config '{"enable_flashcomm1": true, "enable_dsa_cp": true, "ascend_compilation_config": {"enable_npugraph_ex": true}, "multistream_overlap_shared_expert": true, "enable_sparse_sfa_c8": true, "enable_sparse_li_c8": true, "enable_cpu_binding": true}' \
    --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp", "enforce_eager":true}'
```

`run_dp_template.sh` for the decode nodes:

```bash
#!/usr/bin/bash

nic_name="<NIC_NAME>"
local_ip="<CURRENT_NODE_IP>"

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export VLLM_HOST_IP=$local_ip
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_MLAPO=1
export HCCL_BUFFSIZE=2560
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES=$1
export LD_LIBRARY_PATH=/usr/local/python3.11.10/lib:/usr/local/lib:$LD_LIBRARY_PATH
export PYTHONHASHSEED=0
export MOONCAKE_CONFIG_PATH=<MOONCAKE_CONFIG_PATH>
export HCCL_INTRA_ROCE_ENABLE=1
export ACL_OP_INIT_MODE=1

vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.2-w4a8c8 \
    --host 0.0.0.0 \
    --port $2 \
    --data-parallel-size $3 \
    --data-parallel-rank $4 \
    --data-parallel-address $5 \
    --data-parallel-rpc-port $6 \
    --tensor-parallel-size $7 \
    --enable-expert-parallel \
    --enable-prefix-caching \
    --seed 1024 \
    --served-model-name glm-5 \
    --async-scheduling \
    --max-model-len 256000 \
    --max-num-batched-tokens 768 \
    --trust-remote-code \
    --max-num-seqs 128 \
    --gpu-memory-utilization 0.95 \
    --safetensors-load-strategy prefetch \
    --quantization ascend \
    --enable-auto-tool-choice \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --kv-transfer-config \
    '{
    "kv_connector": "MultiConnector",
    "kv_role": "kv_consumer",
    "kv_load_failure_policy": "recompute",
    "kv_connector_extra_config": {
        "connectors": [
            {
                "kv_connector": "MooncakeConnectorV1",
                "kv_role": "kv_consumer",
                "kv_port": "30100",
                "kv_connector_extra_config": {
                    "prefill": {
                        "dp_size": 4,
                        "tp_size": 8
                    },
                    "decode": {
                        "dp_size": 8,
                        "tp_size": 4
                    }
                }
            },
            {
                "kv_connector": "AscendStoreConnector",
                "kv_role": "kv_consumer",
                "kv_connector_extra_config": {
                    "lookup_rpc_port":"0",
                    "load_async": true,
                    "backend": "mooncake"
                }
            }
        ]
    }
    }' \
     --compilation-config \
    '{
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": [4,8,16,24,32,40,48,56,64,96,128,160,192,224,256,298,320,352,384]
    }' \
    --additional-config '{"ascend_compilation_config": {"enable_npugraph_ex": true}, "multistream_overlap_shared_expert": true, "enable_sparse_sfa_c8": true, "enable_sparse_li_c8": true, "enable_cpu_binding": true, "recompute_scheduler_enable": true}' \
    --speculative-config '{"num_speculative_tokens": 5, "method":"deepseek_mtp", "enforce_eager":true}'
```

Once the preparation is done, start the server with the following commands:

1. Prefill nodes — run on `$node_p0_ip`, `$node_p1_ip`, `$node_p2_ip`, `$node_p3_ip` with `--dp-rank-start` `0/1/2/3`:

    ```shell
    python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 1 --dp-rank-start 0 --dp-address $node_p0_ip --dp-rpc-port 16591 --vllm-start-port 9081
    python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 1 --dp-rank-start 1 --dp-address $node_p0_ip --dp-rpc-port 16591 --vllm-start-port 9081
    python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 1 --dp-rank-start 2 --dp-address $node_p0_ip --dp-rpc-port 16591 --vllm-start-port 9081
    python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 1 --dp-rank-start 3 --dp-address $node_p0_ip --dp-rpc-port 16591 --vllm-start-port 9081
    ```

2. Decode nodes — run on `$node_d0_ip`, `$node_d1_ip`, `$node_d2_ip`, `$node_d3_ip` with `--dp-rank-start` `0/2/4/6`:

    ```shell
    python launch_online_dp.py --dp-size 8 --tp-size 4 --dp-size-local 2 --dp-rank-start 0 --dp-address $node_d0_ip --dp-rpc-port 16600 --vllm-start-port 9900
    python launch_online_dp.py --dp-size 8 --tp-size 4 --dp-size-local 2 --dp-rank-start 2 --dp-address $node_d0_ip --dp-rpc-port 16600 --vllm-start-port 9900
    python launch_online_dp.py --dp-size 8 --tp-size 4 --dp-size-local 2 --dp-rank-start 4 --dp-address $node_d0_ip --dp-rpc-port 16600 --vllm-start-port 9900
    python launch_online_dp.py --dp-size 8 --tp-size 4 --dp-size-local 2 --dp-rank-start 6 --dp-address $node_d0_ip --dp-rpc-port 16600 --vllm-start-port 9900
    ```

For request forwarding on this 8-node A2 layout, use 4 prefiller hosts (1 endpoint each) and 4 decoder hosts (2 endpoints each) in the Request Forwarding command below.

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
      $node_p2_ip \
      $node_p3_ip \
    --prefiller-ports \
      9081 9081 \
      9081 9081 \
    --decoder-hosts \
      $node_d0_ip \
      $node_d0_ip \
      $node_d1_ip \
      $node_d1_ip \
      $node_d2_ip \
      $node_d2_ip \
      $node_d3_ip \
      $node_d3_ip \
    --decoder-ports \
      9900 9901 9900 9901 \
      9900 9901 9900 9901
```

Key Parameter Descriptions (in addition to [Prefill-Decode Disaggregation](#5113-prefill-decode-disaggregation)):

This 8-node A2 layout splits the global P/D topology across 8 Atlas 800 A2 nodes: 4 prefill nodes hosting 1 DP rank each (8 cards per rank, `DP4 TP8`) and 4 decode nodes hosting 2 DP ranks each (4 cards per rank, `DP8 TP4`). The `launch_online_dp.py` script is the same as in [Prefill-Decode Disaggregation](#5113-prefill-decode-disaggregation).

**Prefill node-specific configurations (A2):**

- `--additional-config '{"enable_sparse_sfa_c8": true, "enable_sparse_li_c8": true, ...}'`: Both SFA C8 optimizations are enabled in this scenario.

**Decode node-specific configurations (A2):**

- `VLLM_ASCEND_ENABLE_MLAPO=1`: MLAPO fusion on decode nodes for memory-bandwidth-bound token generation.
- `--max-num-batched-tokens 256`: Small batch token limit on decode nodes.

**Multi-connector KV transfer configuration (`--kv-transfer-config`):**

- `"kv_connector": "MultiConnector"`: Combines multiple KV transfer connectors.
- `MooncakeConnectorV1`: Mooncake KV cache transfer between prefill and decode nodes (same role as in [Prefill-Decode Disaggregation](#5113-prefill-decode-disaggregation)).
- `AscendStoreConnector`: The KV Cache Pool (Ascend Store) connector, available since v0.23.0 — see the [KV Cache Pool (Ascend Store) Deployment Guide](https://docs.vllm.ai/projects/ascend/zh-cn/latest/user_guide/feature_guide/kv_pool.html).
- `"kv_load_failure_policy": "recompute"`: When a KV block fails to load from the KV pool, the request falls back to recomputation instead of failing.
- `"load_async": true` (decode side): Asynchronously loads KV cache from the pool on decode nodes.
- `"backend": "mooncake"`: The Ascend Store backend used by the connector.

**Environment variables for the A2 PD scenario:**

- `VLLM_USE_V1=1`: Forces the v1 scheduler.
- `ASCEND_AGGREGATE_ENABLE=1` / `ASCEND_TRANSPORT_PRINT=1`: Communication aggregation and transport debug printing.
- `PYTHONHASHSEED=0`: Fixed hash seed for reproducible distributed scheduling.
- `MOONCAKE_CONFIG_PATH="/mnt/share/scripts/mooncake.json"`: Path to the Mooncake configuration file (must exist on each node).
- `HCCL_INTRA_ROCE_ENABLE=1`: Enables HCCL intra-node communication over RoCE.
- `ACL_OP_INIT_MODE=1`: ACL operator initialization mode.
- `VLLM_HOST_IP=$local_ip`: Host IP advertised by the decode engine for cross-node communication.

Please refer to [envs.py](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/envs.py) for further explanation and restrictions of the environment variables above.

### 5.2 1M Context Deployment

The 1M context scenarios are validated on Atlas 800 A3 only; the A2 series is not validated for 1M context.

::::{warning}
For GLM-5.2's Sparse Flash Attention (SFA) backend, when using DCP in PD disaggregation, enable it on both the prefiller and decoder nodes, or disable it on both. Enabling DCP on only one side can cause known accuracy issues.

DCP and Sparse Flash Attention C8 (`enable_sparse_sfa_c8`, also referred to as `sfa_c8`) have different behavior depending on the weight version:

- `GLM-5.2-w8a8c8`: they can be enabled together — the reference w8a8c8 configs in this document enable both.
- `GLM-5.2-w4a8c8`: enabling them together has known issues in v0.23.0, including performance degradation, and is not recommended.
::::

#### 5.2.1 Dual-Node Co-Located 1M Deployment

**GLM-5.2-w8a8c8 (1M context, dual-node co-located):**

`GLM-5.2-w8a8c8` 1M co-located deployment on 2 Atlas 800 A3 (128GB × 8) (`DP8 TP4`, 4 DP ranks per node, decode context parallelism 4).

Run the following scripts on two nodes respectively.

**node 0**

```shell
nic_name="xxxx" # change to your own nic name
local_ip="xxxx" # change to your own ip

# The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
node0_ip="xxxx"

export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export HCCL_TRANSFER_TIMEOUT=600
export HCCL_EXEC_TIMEOUT=3600
export HCCL_CONNECT_TIMEOUT=3600
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=400
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_MLAPO=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1

vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.2-w8a8c8 \
--host 0.0.0.0 \
--port 8077 \
--api-server-count 1 \
--data-parallel-size 8 \
--data-parallel-start-rank 0 \
--data-parallel-size-local 4 \
--data-parallel-address $node0_ip \
--data-parallel-rpc-port 12980 \
--tensor-parallel-size 4 \
--enable-expert-parallel \
--prefill-context-parallel-size 1 \
--decode-context-parallel-size 4 \
--cp-kv-cache-interleave-size 128 \
--seed 1024 \
--served-model-name glm-5 \
--tool-call-parser glm47 \
--reasoning-parser glm45 \
--enable-auto-tool-choice \
--max-num-seqs 6 \
--max-model-len 1048576 \
--max-num-batched-tokens 4096 \
--trust-remote-code \
--gpu-memory-utilization 0.92 \
--quantization ascend \
--enable-chunked-prefill \
--enable-prefix-caching \
--async-scheduling \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
--additional-config '{"enable_dsa_cp": true, "enable_sparse_sfa_c8": true, "enable_sparse_li_c8": true, "enable_balance_scheduling": true, "fuse_muls_add": true}' \
--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp","enforce_eager":true}'
```

**node 1**

```shell
nic_name="xxxx" # change to your own nic name
local_ip="xxxx" # change to your own ip

# The value of node0_ip must be consistent with the value of local_ip set in node0 (master node)
node0_ip="xxxx"

export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export HCCL_TRANSFER_TIMEOUT=600
export HCCL_EXEC_TIMEOUT=3600
export HCCL_CONNECT_TIMEOUT=3600
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=400
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_MLAPO=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1

vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.2-w8a8c8 \
--host 0.0.0.0 \
--port 8077 \
--headless \
--data-parallel-size 8 \
--data-parallel-start-rank 4 \
--data-parallel-size-local 4 \
--data-parallel-address $node0_ip \
--data-parallel-rpc-port 12980 \
--tensor-parallel-size 4 \
--enable-expert-parallel \
--prefill-context-parallel-size 1 \
--decode-context-parallel-size 4 \
--cp-kv-cache-interleave-size 128 \
--seed 1024 \
--served-model-name glm-5 \
--tool-call-parser glm47 \
--reasoning-parser glm45 \
--enable-auto-tool-choice \
--max-num-seqs 6 \
--max-model-len 1048576 \
--max-num-batched-tokens 4096 \
--trust-remote-code \
--gpu-memory-utilization 0.92 \
--quantization ascend \
--enable-chunked-prefill \
--enable-prefix-caching \
--async-scheduling \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
--additional-config '{"enable_dsa_cp": true, "enable_sparse_sfa_c8": true, "enable_sparse_li_c8": true, "enable_balance_scheduling": true, "fuse_muls_add": true}' \
--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp","enforce_eager":true}'
```

**Notice:**

- `VLLM_ASCEND_ENABLE_FUSED_MC2=1` conflicts with `"multistream_overlap_shared_expert": true` — the runtime automatically disables `multistream_overlap_shared_expert` when fused MC2 is enabled.
- When testing with a prefix cache hit rate > 0, keep `--enable-prefix-caching` (as in the script above); when the hit rate is 0, replace it with `--no-enable-prefix-caching`.

#### 5.2.2 PD Disaggregation 1M Deployment

**GLM-5.2-w8a8c8 (1M context, PD disaggregation):**

`GLM-5.2-w8a8c8` 1M PD disaggregation can be deployed on 4 Atlas 800 A3 (128GB × 8): 2 prefill nodes (`DP2 TP16`, 1 DP rank per node, decode context parallelism 16) and 2 decode nodes (`DP8 TP4`, 4 DP ranks per node, decode context parallelism 4).

**Prefill node 0** (`run_dp_template.sh`): DP rank 0, engine port `9081`. The value of `node_p0_ip` must be consistent with the `local_ip` set on prefill node 0 (DP master node).

```shell
nic_name="xxxx" # change to your own nic name
local_ip="xxxx" # change to your own ip
node_p0_ip="xxxx" # ip of prefill node 0 (DP master node)

export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export HCCL_OP_EXPANSION_MODE="AIV"

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export HCCL_TRANSFER_TIMEOUT=600
export HCCL_EXEC_TIMEOUT=3600
export HCCL_CONNECT_TIMEOUT=3600

export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=400

export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.2-w8a8c8 \
    --host 0.0.0.0 \
    --port 9081 \
    --data-parallel-size 2 \
    --data-parallel-rank 0 \
    --data-parallel-address $node_p0_ip \
    --data-parallel-rpc-port 16600 \
    --tensor-parallel-size 16 \
    --enable-expert-parallel \
    --prefill-context-parallel-size 1 \
    --decode-context-parallel-size 16 \
    --cp-kv-cache-interleave-size 128 \
    --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp","enforce_eager":true}' \
    --seed 1024 \
    --served-model-name glm-5 \
    --max-model-len 1048576 \
    --additional-config '{"fuse_muls_add":true, "recompute_scheduler_enable" : false, "multistream_overlap_shared_expert": true, "enable_dsa_cp":true, "enable_sparse_sfa_c8": true, "enable_sparse_li_c8": true, "c8_enable_reshape_optim":true}' \
    --max-num-batched-tokens 8192 \
    --trust-remote-code \
    --enable-prefix-caching \
    --max-num-seqs 64 \
    --quantization ascend \
    --gpu-memory-utilization 0.9 \
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
        "prefill": {"dp_size": 2, "tp_size": 16},
        "decode": {"dp_size": 8, "tp_size": 4}
    }
}'
```

**Prefill node 1** (`run_dp_template.sh`): DP rank 1, engine port `9082`. `--data-parallel-address` still points to prefill node 0 (DP master node).

```shell
nic_name="xxxx" # change to your own nic name
local_ip="xxxx" # change to your own ip
node_p0_ip="xxxx" # ip of prefill node 0 (DP master node)

export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export HCCL_OP_EXPANSION_MODE="AIV"

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export HCCL_TRANSFER_TIMEOUT=600
export HCCL_EXEC_TIMEOUT=3600
export HCCL_CONNECT_TIMEOUT=3600

export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=400

export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.2-w8a8c8 \
    --host 0.0.0.0 \
    --port 9082 \
    --data-parallel-size 2 \
    --data-parallel-rank 1 \
    --data-parallel-address $node_p0_ip \
    --data-parallel-rpc-port 16600 \
    --tensor-parallel-size 16 \
    --enable-expert-parallel \
    --prefill-context-parallel-size 1 \
    --decode-context-parallel-size 16 \
    --cp-kv-cache-interleave-size 128 \
    --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp","enforce_eager":true}' \
    --seed 1024 \
    --served-model-name glm-5 \
    --max-model-len 1048576 \
    --additional-config '{"fuse_muls_add":true, "recompute_scheduler_enable" : false, "multistream_overlap_shared_expert": true, "enable_dsa_cp":true, "enable_sparse_sfa_c8": true, "enable_sparse_li_c8": true, "c8_enable_reshape_optim":true}' \
    --max-num-batched-tokens 8192 \
    --trust-remote-code \
    --enable-prefix-caching \
    --max-num-seqs 64 \
    --quantization ascend \
    --gpu-memory-utilization 0.9 \
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
        "prefill": {"dp_size": 2, "tp_size": 16},
        "decode": {"dp_size": 8, "tp_size": 4}
    }
}'
```

**Decode node 0** (`run_dp_template.sh`): hosts DP ranks 0–3. Prepare `run_dp_template.sh` and `launch_online_dp.py` (see [Prefill-Decode Disaggregation](#5113-prefill-decode-disaggregation)) on decode node 0 with the content below.

```shell
nic_name="xxxx" # change to your own nic name
local_ip="xxxx" # change to your own ip

export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export HCCL_OP_EXPANSION_MODE="AIV"

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export HCCL_TRANSFER_TIMEOUT=600
export HCCL_EXEC_TIMEOUT=3600
export HCCL_CONNECT_TIMEOUT=3600

#Mooncake
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=400

export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export TASK_QUEUE_ENABLE=1
export ASCEND_RT_VISIBLE_DEVICES=$1

export VLLM_ASCEND_ENABLE_MLAPO=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.2-w8a8c8 \
    --host 0.0.0.0 \
    --port $2 \
    --data-parallel-size $3 \
    --data-parallel-rank $4 \
    --data-parallel-address $5 \
    --data-parallel-rpc-port $6 \
    --tensor-parallel-size $7 \
    --enable-expert-parallel \
    --prefill-context-parallel-size 1 \
    --decode-context-parallel-size 4 \
    --cp-kv-cache-interleave-size 128 \
    --speculative-config '{"num_speculative_tokens": 5,  "method":"deepseek_mtp","enforce_eager":true}' \
    --seed 1024 \
    --served-model-name glm-5 \
    --max-model-len 1048576 \
    --additional-config '{"fuse_muls_add":true, "recompute_scheduler_enable":true, "multistream_overlap_shared_expert":true, "enable_sparse_sfa_c8": true, "enable_sparse_li_c8": true}' \
    --max-num-batched-tokens 192 \
    --trust-remote-code \
    --max-num-seqs 32 \
    --quantization ascend \
    --gpu-memory-utilization 0.92 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --async-scheduling \
    --enable-auto-tool-choice \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --kv-transfer-config \
    '{"kv_connector": "MooncakeConnectorV1",
    "kv_role": "kv_consumer",
    "kv_port": "30200",
    "engine_id": "2",
    "kv_connector_extra_config": {
        "use_ascend_direct": true,
        "prefill": {"dp_size": 2, "tp_size": 16},
        "decode": {"dp_size": 8, "tp_size": 4}
    }
}'
```

**Decode node 1** (`run_dp_template.sh`): hosts DP ranks 4–7. Prepare `run_dp_template.sh` and `launch_online_dp.py` (see [Prefill-Decode Disaggregation](#5113-prefill-decode-disaggregation)) on decode node 1 with the content below.

```shell
nic_name="xxxx" # change to your own nic name
local_ip="xxxx" # change to your own ip

export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export HCCL_OP_EXPANSION_MODE="AIV"

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export HCCL_TRANSFER_TIMEOUT=600
export HCCL_EXEC_TIMEOUT=3600
export HCCL_CONNECT_TIMEOUT=3600

#Mooncake
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=400

export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export TASK_QUEUE_ENABLE=1
export ASCEND_RT_VISIBLE_DEVICES=$1

export VLLM_ASCEND_ENABLE_MLAPO=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.2-w8a8c8 \
    --host 0.0.0.0 \
    --port $2 \
    --data-parallel-size $3 \
    --data-parallel-rank $4 \
    --data-parallel-address $5 \
    --data-parallel-rpc-port $6 \
    --tensor-parallel-size $7 \
    --enable-expert-parallel \
    --prefill-context-parallel-size 1 \
    --decode-context-parallel-size 4 \
    --cp-kv-cache-interleave-size 128 \
    --speculative-config '{"num_speculative_tokens": 5,  "method":"deepseek_mtp","enforce_eager":true}' \
    --seed 1024 \
    --served-model-name glm-5 \
    --max-model-len 1048576 \
    --additional-config '{"fuse_muls_add":true, "recompute_scheduler_enable":true, "multistream_overlap_shared_expert":true, "enable_sparse_sfa_c8": true, "enable_sparse_li_c8": true}' \
    --max-num-batched-tokens 192 \
    --trust-remote-code \
    --max-num-seqs 32 \
    --quantization ascend \
    --gpu-memory-utilization 0.92 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --async-scheduling \
    --enable-auto-tool-choice \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --kv-transfer-config \
    '{"kv_connector": "MooncakeConnectorV1",
    "kv_role": "kv_consumer",
    "kv_port": "30200",
    "engine_id": "2",
    "kv_connector_extra_config": {
        "use_ascend_direct": true,
        "prefill": {"dp_size": 2, "tp_size": 16},
        "decode": {"dp_size": 8, "tp_size": 4}
    }
}'
```

Once the preparation is done, start the server on each node:

1. Prefill node 0: run the script above (DP rank 0, port `9081`).

2. Prefill node 1: run the script above (DP rank 1, port `9082`).

3. Decode node 0:

    ```shell
    # change ip to your own
    python launch_online_dp.py --dp-size 8 --tp-size 4 --dp-size-local 4 --dp-rank-start 0 --dp-address $node_d0_ip --dp-rpc-port 16600 --vllm-start-port 9900
    ```

4. Decode node 1:

    ```shell
    # change ip to your own
    python launch_online_dp.py --dp-size 8 --tp-size 4 --dp-size-local 4 --dp-rank-start 4 --dp-address $node_d0_ip --dp-rpc-port 16600 --vllm-start-port 9900
    ```

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
    9081 \
    9082 \
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
    9900 9901 9902 9903 \
    9900 9901 9902 9903
```

**Notice:**

- When testing with a prefix cache hit rate > 0, add `--enable-prefix-caching` on the prefill nodes (as in the script above); when the hit rate is 0, use `--no-enable-prefix-caching` instead.
- `"recompute_scheduler_enable"` is set to `false` on prefill nodes and `true` on decode nodes in this scenario.

Key Parameter Descriptions (in addition to [Prefill-Decode Disaggregation](#5113-prefill-decode-disaggregation) and [Dual-Node Co-Located 1M Deployment](#521-dual-node-co-located-1m-deployment)):

**Prefill nodes (1M):**

- `--prefill-context-parallel-size 1` / `--decode-context-parallel-size 16` / `--cp-kv-cache-interleave-size 128`: Decode context parallelism of 16 for the decode phase of the 1M scenario.
- `--additional-config '{"recompute_scheduler_enable": false, ...}'`: The recompute scheduler is disabled on prefill nodes in this scenario.
- `--kv-transfer-config`: Mooncake connector as `kv_producer` with `prefill: dp2 tp16` / `decode: dp8 tp4`.

**Decode nodes (1M):**

- `--decode-context-parallel-size 4` / `--cp-kv-cache-interleave-size 128`: Decode context parallelism of 4 per decode node.
- `--max-num-batched-tokens 192`: Decode nodes store the large 1M KV cache received from prefill nodes.
- `--kv-transfer-config`: Mooncake connector as `kv_consumer` (`kv_port: 30200`).

## 6 Functional Verification

Once your server is started, you can query the model with input prompts:

```shell
curl http://<node0_ip>:<port>/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "glm-52",
        "prompt": "The future of AI is",
        "max_completion_tokens": 50,
        "temperature": 0
    }'
```

## 7 Accuracy Evaluation

Here are two accuracy evaluation methods.

### 7.1 Using AISBench

1. Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details.

2. After execution, you can get the result.

### 7.2 Using Language Model Evaluation Harness

Not tested yet.

## 8 Performance Evaluation

### 8.1 Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### 8.2 Using vLLM Benchmark

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

## 9 Performance Tuning

### 9.1 Recommended Configurations

> **Note**: The following configurations are validated in specific test environments and are for reference only. The optimal configuration depends on factors such as maximum input/output length, prefix cache hit rate, precision requirements, and deployment machine ratios. It is recommended to refer to [Tuning Guidelines](#92-tuning-guidelines) for tuning based on actual conditions.

The tables below provide recommended parameter configurations for different deployment scenarios. All scenarios are categorized by use case (High Throughput, Low Latency, Long Context) and correspond to the deployment modes documented in [Deployment](#5-deployment).

#### 9.1.1 Table 1: Detailed Node Configuration

> The TP/DP columns show the values **per node** as configured in the Deployment scripts (a prefill node hosting 1 DP rank of TP16 uses 16 NPUs; a decode node hosting 4 DP ranks of TP4 uses 16 NPUs). The 198K PD scenario prefill side uses `PP2 TP16` with the layer partition `41,37`. All scenarios use the `GLM-5.2-w8a8c8` weights.
>
> When testing with a prefix cache hit rate > 0, keep `--enable-prefix-caching` (as in the deployment scripts); when the hit rate is 0, replace it with `--no-enable-prefix-caching`.

|Scenario|Weight Version|Configuration|NPUs|TP|DP|Max Num Seqs|Max Num Batched Tokens|Max Model Len|MTP Spec Num|
|--------|--------------|-------------|-----|--|--|------------|----------------------|--------------|-------------|
|Dual-Node Co-Located 198K High Throughput (A3)|w8a8c8|Dual-Node Co-Located Node (0/1)|16|4|4|6|4096|202752|3|
|PD 198K High Throughput (A3)|w8a8c8|PD — Server-P Node (PP2)|16|16|1|64|16384|202752|1|
|PD 198K High Throughput (A3)|w8a8c8|PD — Server-D Node|16|4|4|32|164|202752|3|
|Dual-Node Co-Located 1M Long Context (A3)|w8a8c8|Dual-Node Co-Located Node (0/1)|16|4|4|6|4096|1048576|3|
|PD 1M Long Context (A3)|w8a8c8|PD — Server-P Node|16|16|1|64|8192|1048576|1|
|PD 1M Long Context (A3)|w8a8c8|PD — Server-D Node|16|4|4|32|192|1048576|5|

#### 9.1.2 Table 2: Optimizations Requiring Explicit Enablement

The following optimizations must be explicitly enabled to take effect. They apply to the A3 series (w8a8c8) or the A2 series (w4a8c8) as indicated:

|Optimization|Scenario|Enablement|Principle (Benefits)|Notes|
|------------|--------|----------|---------------------|-----|
|FlashComm_v1|A3 prefill nodes / co-located nodes|`export VLLM_ASCEND_ENABLE_FLASHCOMM1=1`|Splits AllReduce into Reduce-Scatter and All-Gather, improving prefill throughput and reducing communication latency|Not available when `layer_sharding` includes `o_proj`|
|Fused MC2|A3 prefill nodes|`export VLLM_ASCEND_ENABLE_FUSED_MC2=1`|Replaces ALLTOALL+MC2 with the `dispatch_ffn_combine`/`dispatch_gmm_combine_decode` operators, reducing MoE communication overhead and improving MoE inference performance|`dispatch_ffn_combine` only for w8a8, EP≤32, non-MTP, non-dynamic-EPLB; conflicts with `multistream_overlap_shared_expert` (the latter is auto-disabled)|
|MLAPO|A3 co-located / PD decode nodes; A2 P/D nodes|`export VLLM_ASCEND_ENABLE_MLAPO=1`|Fuses the MLA preprocess operations, significantly improving decode performance|Consumes more NPU memory; in PD scenarios enable on decode nodes only|
|DSA CP|A3 prefill nodes; long context (≥128K)|`--additional-config '{"enable_dsa_cp": true}'`|DSA context parallelism accelerates long-context prefill, reducing TTFT for long prompts|In the reference configs, enabled on co-located nodes and PD prefill nodes; PD decode nodes use decode context parallelism instead|
|Balance Scheduling|A3 single-node / co-located / non-PD scenarios|`--additional-config '{"enable_balance_scheduling": true}'`|Improves output throughput and reduces TPOT in the v1 scheduler|TTFT may degrade; not recommended when Prefill-Decode is separated|
|Sparse SFA C8|A3 (w8a8c8); long-context prefill|`--additional-config '{"enable_sparse_sfa_c8": true}'`|Sparse Flash Attention skips unnecessary attention computation of the C8 quantized model, accelerating long-context prefill|Experimental in v0.23.0. On the w8a8c8 weights it can be combined with DCP/context parallelism (as in the reference configs in this document); on the w4a8c8 weights enabling both together has known issues and is not recommended|
|Sparse LI C8|A3 (w8a8c8)|`--additional-config '{"enable_sparse_li_c8": true}'`|Sparse attention optimization reduces computation of the C8 quantized model, improving throughput|Independent of `enable_sparse_sfa_c8`|
|Recompute Scheduler|A3 decode nodes|`--additional-config '{"recompute_scheduler_enable": true}'`|Recomputes KV cache on prefill nodes when decode KV cache is insufficient, avoiding decode-side OOM and improving throughput|Set to `false` on prefill nodes|
|Multistream Overlap Shared Expert|A3|`--additional-config '{"multistream_overlap_shared_expert": true}'`|Overlaps shared-expert computation on an additional stream, hiding its latency and improving decode performance|Auto-disabled when `VLLM_ASCEND_ENABLE_FUSED_MC2=1`|

> For complete startup commands and detailed parameter descriptions, please refer to the deployment examples and Key Parameter Descriptions in [Deployment](#5-deployment).

#### 9.1.3 Table 3: Performance-Related Parameter Tuning Guide

|Parameter|Low Latency|High Throughput|Long Context|Description|
|---------|-----------|---------------|-------------|-----------|
|`--max-num-seqs`|Lower (4–12)|Higher (6–64)|Controlled (6–64)|Limits concurrent sequences. Lower values reduce scheduling latency; higher values increase throughput.|
|`--max-model-len`|Shorter (40K–135K)|Longer (128K–200K)|Maximum (1M)|Maximum context length. Must accommodate your longest input+output. Larger values consume more KV cache memory.|
|`--max-num-batched-tokens`|Lower (4096–8192)|Higher for prefill (8192–16384)|Higher for prefill (8192)|Controls batch size per step. Lower values reduce per-step latency; higher values improve prefill throughput.|
|`--gpu-memory-utilization`|0.92–0.95|0.90–0.95|0.90–0.92|NPU memory fraction. The reference w8a8c8 1M configs in this document use 0.90 (PD prefill) to 0.92 (co-located / PD decode). Reduce if OOM.|
|`--enable-chunked-prefill`|Enable|Enable|Enable|Splits long prompts into chunks to prevent prefill from blocking decode. Recommended in all scenarios.|
|`num_speculative_tokens` (MTP)|3–5|3–5|1 (prefill) / 3–5 (decode)|MTP speculation count. Higher values improve decode throughput at the cost of memory for the draft model KV cache. Use `1` on prefill nodes in PD mode.|
|`cudagraph_mode`|FULL_DECODE_ONLY|FULL_DECODE_ONLY (co-located / decode nodes)|FULL_DECODE_ONLY (co-located / decode nodes)|Graph capture for the decode phase only. Prefill nodes in PD mode use `--enforce-eager` instead.|

### 9.2 Tuning Guidelines

For general performance tuning methods, refer to the [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md).

For detailed feature descriptions and configuration options, refer to the [Feature Guide](../../user_guide/support_matrix/feature_matrix.md).

For environment variable descriptions and constraints, refer to [envs.py](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/envs.py).

## 10 FAQ

- **Q: How to enable function calling for GLM-5.2?**

  A: Please add following configurations in vLLM startup command

  ```shell
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --enable-auto-tool-choice \
  ```
