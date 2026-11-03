# Qwen3.8-Flash-Next (Experimental)

## 1 Introduction

Qwen3.8-Flash-Next is a multimodal Mixture-of-Experts (MoE) model and an experimental preview of the architecture that will underpin Qwen4. Its language model combines Gated DeltaNet (GDN) and Qwen Sparse Attention (QSA), gated residual connections, Position Learning Enhancement (PLE), and a native Multi-Token Prediction (MTP) head.

This tutorial describes model preparation, installation, online serving, functional verification, and evaluation. It covers colocated deployment and prefill/decode (PD) separation on A2 series, A3 series, and Ascend 950 DT. A2 series and A3 series share the service configuration; PD separation requires 16 devices in total for the shared TP2 × DP4 configuration.

Qwen3.8-Flash-Next was first supported in `vllm-ascend:v0.26.0rc`. The original A3 series W8A8 configuration was validated with text and multimodal input. The deployment examples below use W4A8 checkpoints (W4A8-MXFP on Ascend 950 DT) and additional performance settings; they require separate runtime and accuracy validation.

!!! warning

    Automatic Prefix Caching is experimental. For A2 series / A3 series colocated deployment, enable it with `--enable-prefix-caching --mamba-cache-mode align` and revalidate NPU memory capacity for the target context length and concurrency.

## 2 Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_features.md) to get the model's supported feature matrix.

Refer to [Feature Guide](../../user_guide/feature_guide/index.md) to get feature configuration details.

## 3 Prerequisites

### 3.1 Model Weight

| Weight Version | Hardware Requirements | Download Links |
| --- | --- | --- |
| `Qwen3.8-Flash-Next-w8a8-mtp` (quantized version) | 1 A3 series node (16 devices) | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/Qwen3.8-Flash-Next-w8a8-mtp) |
| `Qwen3.8-Flash-Next-w4a8` (quantized version) | 1 A3 series node (16 devices) or 1 A2 series node (8 devices) | |
| `Qwen3.8-Flash-Next-w4a8-mxfp` (quantized version) | 1 Ascend 950 DT node (8 devices) | |

The commands use fixed model directories under `/models`. Mount the corresponding checkpoint at the shown directory or edit that path directly. For PD separation, both roles must load the same checkpoint.

## 4 Installation

### 4.1 Docker Image Installation

The following images were provided for the original A3 series baseline. Verify that the installed runtime supports the selected W4A8 checkpoint. For A2 series, use a compatible environment as described in Section 4.2. For 950 DT, prepare an environment with compatible vLLM and vLLM-Ascend versions; do not use an A3 series image for that deployment.

A pre-built Qwen3.8 A3 series image is available in the [vllm-ascend repository](https://quay.io/repository/ascend/vllm-ascend?tab=tags&tag=latest). Select the image that matches the host operating system:

- openEuler: `quay.io/ascend/vllm-ascend:qwen3.8-next-a3-openeuler`
- Ubuntu: `quay.io/ascend/vllm-ascend:qwen3.8-next-a3`

For PD deployment, verify host NPU connectivity and `/etc/hccn.conf` using the [Mooncake communication checks](../features/pd_disaggregation_mooncake_single_node.md#verify-communication-environment).

The following example uses the AArch64 image and exposes 16 devices on an A3 series node. The service command in [Section 5.1](#51-single-node-online-deployment) uses the first eight devices; the A3 series PD example in Section 5.2 uses all 16.

=== "A3 series"

    ```bash
    docker pull quay.io/ascend/vllm-ascend:qwen3.8-next-a3

    docker run --rm \
        --name vllm-ascend-qwen38-flash-next \
        --shm-size=16g \
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
        -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
        -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        -v /etc/hccn.conf:/etc/hccn.conf:ro \
        -v /path/to/models:/models \
        -it quay.io/ascend/vllm-ascend:qwen3.8-next-a3 bash
    ```

After entering the container, verify that vLLM and vLLM-Ascend can be imported:

```bash
python -c "import vllm, vllm_ascend; print('vllm and vllm-ascend are ready')"
```

### 4.2 Source Code Installation

For A2 series / A3 series and Ascend 950 DT, install `vllm-ascend` from source by following [installation](../../getting_started/installation.md#installation-existing-cann-install).

If you want to deploy a multi-node environment, you need to set up the environment on each node.

## 5 Online Service Deployment {: #5-online-service-deployment }

### 5.1 Single-Node Online Deployment

Prefill and decode run in the same service. Select the hardware tab below.

Before starting:

- Mount the matching checkpoint at the model directory shown in the command.
- Network examples use `192.168.0.1` and `eth0`. Replace these literals with the serving host address and its network interface.
- Verify available NPU memory before changing context length, concurrency, or memory utilization.

=== "A2 series / A3 series"

    This W4A8 deployment uses TP8 × DP1, a maximum context length of 131072 tokens, and up to 8 active sequences.

    ```bash
    unset CPLUS_INCLUDE_PATH CPATH C_INCLUDE_PATH
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh

    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export VLLM_USE_V2_MODEL_RUNNER=1
    export VLLM_ENGINE_READY_TIMEOUT_S=4800
    export VLLM_ASCEND_PLE_BACKEND=torch
    export VLLM_ASCEND_ENABLE_QSA_LIGHTNING_INDEXER=1
    export VLLM_ASCEND_ENABLE_QSA_E3V=1
    unset VLLM_ASCEND_FORCE_QSA_REFERENCE

    vllm serve /models/Qwen3.8-Flash-Next-w4a8 \
        --host 0.0.0.0 \
        --port 8088 \
        --served-model-name qwen3.8-flash-next \
        --trust-remote-code \
        --quantization ascend \
        --tensor-parallel-size 8 \
        --data-parallel-size 1 \
        --data-parallel-size-local 1 \
        --data-parallel-start-rank 0 \
        --enable-expert-parallel \
        --max-model-len 131072 \
        --max-num-seqs 8 \
        --max-num-batched-tokens 4096 \
        --gpu-memory-utilization 0.93 \
        --enable-prefix-caching \
        --mamba-cache-mode align \
        --async-scheduling \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_xml \
        --reasoning-parser qwen3 \
        --engram-config '{"cpu_offload":true,"dp_shared_memory":false}' \
        --compilation-config '{"cudagraph_capture_sizes":[4,8,12,16,20,24,28,32],"cudagraph_mode":"FULL_DECODE_ONLY"}' \
        --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":false}' \
        --additional-config '{"enable_cpu_binding":true,"enable_shared_expert_dp":true,"enable_fused_mc2":0,"enable_flashcomm1":true,"multistream_overlap_shared_expert":true,"ascend_compilation_config":{"fuse_norm_quant":false,"enable_npugraph_ex":true,"enable_static_kernel":false,"enable_super_kernel":false}}'
    ```

    **Key parameters:**

    - `VLLM_ASCEND_ENABLE_QSA_LIGHTNING_INDEXER=1` enables the fused QSA Lightning Indexer path.
    - `VLLM_ASCEND_ENABLE_QSA_E3V=1` enables the fused QSA Expand E3 path.
    - `--quantization ascend` loads the Ascend-compatible W4A8 checkpoint.
    - `--tensor-parallel-size 8` and `--data-parallel-size 1` configure the A2 series / A3 series single-DP TP8 topology.
    - `--max-model-len 131072` caps the maximum context length. Common values: `65536` (64K), `131072` (128K).
    - `--enable-expert-parallel` enables expert parallelism for the MoE layers.
    - `--enable-prefix-caching --mamba-cache-mode align` enables Prefix Caching with aligned GDN and PLE state checkpoints.
    - The command does not set `--language-model-only`. Multimodal requests require a checkpoint that includes the vision encoder; verify them with the selected W4A8 checkpoint. For a text-only deployment, you may add `--language-model-only` to skip loading the vision encoder.
    - `--enable-auto-tool-choice --tool-call-parser qwen3_xml` enables automatic Function Calling with the Qwen3 XML parser.
    - `--reasoning-parser qwen3` separates reasoning from the final answer in the OpenAI-compatible response.
    - `--speculative-config` uses the model's MTP head to draft three tokens. `enforce_eager=false` allows the MTP proposer to use graph execution, matching the PD decode setting.
    - `--compilation-config` enables `FULL_DECODE_ONLY` ACLGraph for decode.
    - `--engram-config` enables Engram embedding-table CPU offload with `cpu_offload=true`. This DP1 configuration sets `dp_shared_memory=false`; provide sufficient host memory and use a checkpoint with supported n-gram layers.
    - `fuse_norm_quant=false` disables norm/quantization fusion for this W4A8 deployment.
    - `VLLM_USE_V2_MODEL_RUNNER=1` selects Model Runner V2; `VLLM_ASCEND_PLE_BACKEND=torch` selects the PLE backend.
    - `--async-scheduling` enables asynchronous scheduling.
    - `--additional-config` enables CPU binding, FlashComm1, shared expert DP, and shared expert multistream overlap. Shared expert DP requires expert parallelism and TP > 1. `enable_fused_mc2=0` disables fused MC2 and is compatible with shared expert overlap.
    - `enable_npugraph_ex=true` enables NPU graph extension; static kernels and super kernels remain disabled.

=== "Ascend 950 DT"

    This W4A8-MXFP deployment uses TP4 × DP1, a maximum context length of 141312 tokens, and up to 32 active sequences.

    ```bash
    unset CPLUS_INCLUDE_PATH CPATH C_INCLUDE_PATH
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh

    export VLLM_HOST_IP="192.168.0.1"
    export HCCL_IF_IP="192.168.0.1"
    export GLOO_SOCKET_IFNAME="eth0"
    export TP_SOCKET_IFNAME="eth0"
    export HCCL_SOCKET_IFNAME="eth0"
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
    export VLLM_USE_V2_MODEL_RUNNER=1
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
    export VLLM_ENGINE_READY_TIMEOUT_S=4800
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export TASK_QUEUE_ENABLE=1
    export HCCL_BUFFSIZE=1024
    export HCCL_NPU_SOCKET_PORT_RANGE=auto

    vllm serve /models/Qwen3.8-Flash-Next-w4a8-mxfp \
        --host 0.0.0.0 \
        --port 8087 \
        --served-model-name qwen3.8-flash-next-950-dt \
        --trust-remote-code \
        --quantization ascend \
        --tensor-parallel-size 4 \
        --data-parallel-size 1 \
        --max-model-len 141312 \
        --max-num-seqs 32 \
        --max-num-batched-tokens 4096 \
        --gpu-memory-utilization 0.90 \
        --enable-prefix-caching \
        --async-scheduling \
        --enable-expert-parallel \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_xml \
        --reasoning-parser qwen3 \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
        --additional-config '{"enable_cpu_binding":true,"enable_shared_expert_dp":true,"enable_fused_mc2":0,"enable_flashcomm1":true,"multistream_overlap_shared_expert":true,"ascend_compilation_config":{"fuse_norm_quant":false,"enable_npugraph_ex":true,"enable_static_kernel":false,"enable_super_kernel":false}}' \
        --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}' \
        --profiler-config '{"profiler":"torch","torch_profiler_dir":"/tmp/qwen38-next-950-dt-profiling","torch_profiler_with_stack":false}'
    ```

    **Key parameters:**

    - `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3` selects the four devices for TP4.
    - `VLLM_USE_V2_MODEL_RUNNER=1` selects Model Runner V2.
    - `--quantization ascend` loads the Ascend-compatible W4A8-MXFP checkpoint.
    - `--max-model-len 141312` and `--max-num-seqs 32` set context and concurrency limits; verify memory capacity before increasing them.
    - `--async-scheduling` enables asynchronous scheduling; `--enable-expert-parallel` enables expert parallelism for MoE layers.
    - `--additional-config` enables CPU binding, FlashComm1, shared expert DP, and shared expert multistream overlap. Shared expert DP requires expert parallelism and TP > 1. `enable_fused_mc2=0` disables fused MC2 and is compatible with shared expert overlap.
    - NPU graph extension is enabled; static kernels, super kernels, and norm/quantization fusion are disabled.
    - `FULL_DECODE_ONLY` enables decode graph capture. The MTP configuration uses three draft tokens without explicitly forcing eager execution.
    - Prefix Caching is enabled. The current runtime automatically selects `mamba_cache_mode=align` when Prefix Caching is enabled and the mode is left at its default `none`; the A2 series / A3 series example sets the same mode explicitly. This is not a hardware-specific difference.
    - `--enable-auto-tool-choice --tool-call-parser qwen3_xml` enables Function Calling; `--reasoning-parser qwen3` separates reasoning from the final answer.
    - `--profiler-config` sets the trace output directory; it does not start profiling by itself.

**Service Verification:**

Wait for model loading and graph capture to complete, then check the service from the serving host. Use port `8088` for A2 series / A3 series or `8087` for 950 DT:

```bash
curl --fail http://127.0.0.1:8088/v1/models
```

The response should list the configured model name. After startup, use [Section 6](#6-functional-verification) to verify generation.

### 5.2 PD Separation Deployment

Select the hardware tab below.

=== "A2 series / A3 series"

    Prefill and Decode each use eight devices with TP2 × DP4 and the same W4A8 checkpoint. The commands below use one 16-device A3 series host; for two eight-device A2 series hosts, apply the changes in the table below.

    Before starting:

    - Install compatible vLLM, vLLM-Ascend, and [Mooncake](../features/pd_disaggregation_mooncake_single_node.md#install-mooncake). Adjust the Mooncake library path if it differs from the Python 3.12.16 path shown below.
    - Expose the devices used by each host and mount the same checkpoint and `/etc/hccn.conf`. Replace the example IP addresses, `eth0`, and model path with your environment settings; provide sufficient host memory for Engram CPU offload.
    - Run P and D in separate terminals. After both APIs are ready, start the proxy from the vLLM-Ascend repository root.

    For A2 series, run P on `192.168.0.1` and D on `192.168.0.2`, keeping each role's four DP replicas on its own host:

    | Setting | Prefill host | Decode host |
    | --- | --- | --- |
    | `ASCEND_RT_VISIBLE_DEVICES` | `0,1,2,3,4,5,6,7` | `0,1,2,3,4,5,6,7` |
    | `VLLM_HOST_IP` / `HCCL_IF_IP` | `192.168.0.1` | `192.168.0.2` |
    | Proxy backend | `--prefiller-hosts 192.168.0.1` | `--decoder-hosts 192.168.0.2` |
    | API verification from the proxy host | `http://192.168.0.1:25568/v1/models` | `http://192.168.0.2:25569/v1/models` |

    Run the A2 series proxy on the prefill host. Set `no_proxy` and `NO_PROXY` in all three terminals to `localhost,127.0.0.1,192.168.0.1,192.168.0.2`, and update each host's socket-interface variables.

    **Prefill:**

    ```bash
    #!/usr/bin/env bash
    set -eo pipefail
    unset CPLUS_INCLUDE_PATH CPATH C_INCLUDE_PATH
    unset ftp_proxy FTP_PROXY https_proxy HTTPS_PROXY http_proxy HTTP_PROXY
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh

    export VLLM_SERVER_DEV_MODE=1
    export VLLM_USE_V2_MODEL_RUNNER=1
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
    export VLLM_ENGINE_READY_TIMEOUT_S=4800
    export VLLM_ASCEND_PLE_BACKEND=torch
    export VLLM_HOST_IP="192.168.0.1"
    export HCCL_IF_IP="192.168.0.1"
    export GLOO_SOCKET_IFNAME="eth0"
    export TP_SOCKET_IFNAME="eth0"
    export HCCL_SOCKET_IFNAME="eth0"
    export no_proxy="localhost,127.0.0.1,192.168.0.1"
    export NO_PROXY="localhost,127.0.0.1,192.168.0.1"
    export LD_LIBRARY_PATH="/usr/local/python3.12.16/lib/python3.12/site-packages/mooncake:${LD_LIBRARY_PATH:-}"
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export HCCL_NPU_SOCKET_PORT_RANGE=61400-61499
    export HCCL_BUFFSIZE=4096

    exec vllm serve /models/Qwen3.8-Flash-Next-w4a8 \
        --host 0.0.0.0 \
        --port 25568 \
        --served-model-name qwen38-flash-next-a2-a3-pd \
        --trust-remote-code \
        --quantization ascend \
        --tensor-parallel-size 2 \
        --data-parallel-size 4 \
        --max-model-len 141312 \
        --max-num-seqs 32 \
        --max-num-batched-tokens 4096 \
        --gpu-memory-utilization 0.90 \
        --enable-prefix-caching \
        --async-scheduling \
        --enable-expert-parallel \
        --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
        --compilation-config '{"cudagraph_mode":"NONE"}' \
        --additional-config '{"enable_shared_expert_dp":true,"enable_flashcomm1":true,"enable_prefill_mc2":true,"enable_fused_mc2":0,"enable_cpu_binding":true,"multistream_overlap_shared_expert":true,"ascend_compilation_config":{"fuse_norm_quant":false}}' \
        --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":true}' \
        --kv-transfer-config '{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_producer","kv_port":"32600","kv_connector_extra_config":{"prefill":{"dp_size":4,"tp_size":2},"decode":{"dp_size":4,"tp_size":2}}}'
    ```

    **Decode:**

    ```bash
    #!/usr/bin/env bash
    set -eo pipefail
    unset CPLUS_INCLUDE_PATH CPATH C_INCLUDE_PATH
    unset ftp_proxy FTP_PROXY https_proxy HTTPS_PROXY http_proxy HTTP_PROXY
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh

    export VLLM_SERVER_DEV_MODE=1
    export VLLM_USE_V2_MODEL_RUNNER=1
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
    export VLLM_ENGINE_READY_TIMEOUT_S=4800
    export VLLM_ASCEND_PLE_BACKEND=torch
    export VLLM_HOST_IP="192.168.0.1"
    export HCCL_IF_IP="192.168.0.1"
    export GLOO_SOCKET_IFNAME="eth0"
    export TP_SOCKET_IFNAME="eth0"
    export HCCL_SOCKET_IFNAME="eth0"
    export no_proxy="localhost,127.0.0.1,192.168.0.1"
    export NO_PROXY="localhost,127.0.0.1,192.168.0.1"
    export LD_LIBRARY_PATH="/usr/local/python3.12.16/lib/python3.12/site-packages/mooncake:${LD_LIBRARY_PATH:-}"
    export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
    export HCCL_NPU_SOCKET_PORT_RANGE=61500-61599
    export HCCL_BUFFSIZE=512

    exec vllm serve /models/Qwen3.8-Flash-Next-w4a8 \
        --host 0.0.0.0 \
        --port 25569 \
        --served-model-name qwen38-flash-next-a2-a3-pd \
        --trust-remote-code \
        --quantization ascend \
        --tensor-parallel-size 2 \
        --data-parallel-size 4 \
        --max-model-len 141312 \
        --max-num-seqs 32 \
        --max-num-batched-tokens 4096 \
        --gpu-memory-utilization 0.90 \
        --enable-prefix-caching \
        --async-scheduling \
        --enable-expert-parallel \
        --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
        --additional-config '{"enable_flashcomm1":true,"enable_cpu_binding":true,"enable_shared_expert_dp":true,"multistream_overlap_shared_expert":true,"scheduler_config":{"recompute_scheduler_enable":true},"ascend_compilation_config":{"fuse_norm_quant":false,"enable_npugraph_ex":true,"enable_static_kernel":true,"enable_super_kernel":false}}' \
        --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":false}' \
        --kv-transfer-config '{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_consumer","kv_port":"32700","kv_connector_extra_config":{"prefill":{"dp_size":4,"tp_size":2},"decode":{"dp_size":4,"tp_size":2}}}'
    ```

    **Proxy:**

    ```bash
    #!/usr/bin/env bash
    set -eo pipefail
    unset ftp_proxy FTP_PROXY https_proxy HTTPS_PROXY http_proxy HTTP_PROXY
    export no_proxy="localhost,127.0.0.1,192.168.0.1"
    export NO_PROXY="localhost,127.0.0.1,192.168.0.1"

    exec python examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
        --host 0.0.0.0 --port 25570 \
        --prefiller-hosts "192.168.0.1" --prefiller-ports 25568 \
        --decoder-hosts "192.168.0.1" --decoder-ports 25569
    ```

    **Service Verification:**

    Wait for both services to finish loading. On the serving host, check that both APIs respond successfully and list `qwen38-flash-next-a2-a3-pd`:

    ```bash
    curl --fail http://127.0.0.1:25568/v1/models
    curl --fail http://127.0.0.1:25569/v1/models
    ```

    After both checks succeed, run the Proxy command in a separate terminal.

    Once the proxy is listening, verify the complete PD path:

    ```bash
    curl --fail http://127.0.0.1:25570/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "qwen38-flash-next-a2-a3-pd",
            "messages": [{"role": "user", "content": "Who are you?"}],
            "max_tokens": 1024,
            "temperature": 0
        }'
    ```

    Inspect the prefill, decode, and proxy terminal output if the request fails.

    **Key parameters:**

    - `VLLM_ASCEND_PLE_BACKEND=torch` selects the PLE backend for both roles.
    - Both roles enable Prefix Caching; the current runtime automatically changes the default Mamba cache mode from `none` to `align`.
    - Engram requires a checkpoint with supported n-gram layers. `cpu_offload=true` keeps Engram embedding tables in host memory. `dp_shared_memory=true` shares the corresponding table shards among node-local DP replicas; replicas must share the same host and shared-memory namespace. Both options are booleans. This DP configuration satisfies the DP > 1 requirement; elastic EP is not supported with shared memory.
    - Prefill sets `cudagraph_mode=NONE`, enables `enable_prefill_mc2=true`, and disables fused MC2. Verify MC2 token limits for TP2 × DP4 before increasing the batch token budget.
    - Decode enables `FULL_DECODE_ONLY`, NPU graph extension, static kernels, and recompute scheduling. Super kernels remain disabled.
    - Both roles use three MTP draft tokens. Prefill sets `enforce_eager=true`; decode sets `enforce_eager=false`.
    - `enable_fused_mc2=0` disables fused MC2, so it does not conflict with `multistream_overlap_shared_expert=true`. The decode command relies on the runtime's default value of zero.
    - Keep the Mooncake DP4 / TP2 declarations synchronized with both services' parallel settings.
    - For Function Calling and reasoning parsing, add `--enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3` to both roles and verify proxy forwarding. Multimodal input and thinking control also require validation through the proxy with the W4A8 checkpoint.

=== "Ascend 950 DT"

    This W4A8-MXFP deployment uses eight devices on one Ascend 950 DT host. Prefill and decode each use TP2 × DP2, with a separate API process for each DP rank. MooncakeConnectorV2 transfers KV cache between the two roles.

    | Setting | Prefill | Decode |
    | --- | --- | --- |
    | Rank 0 / rank 1 devices | 0,1 / 2,3 | 4,5 / 6,7 |
    | TP × DP | 2 × 2 | 2 × 2 |
    | Rank 0 / rank 1 API ports | 8088 / 8089 | 8092 / 8093 |
    | DP RPC port | 13388 | 13389 |
    | Mooncake role / KV base port | `kv_producer` / 30000 | `kv_consumer` / 30100 |
    | Maximum context / sequences / batched tokens | 141312 / 64 / 4096 | 141312 / 64 / 4096 |

    Mount the same compatible W4A8-MXFP checkpoint at `/models/Qwen3.8-Flash-Next-w4a8-mxfp` for all four processes. Install compatible vLLM, vLLM-Ascend, and Mooncake in the active Python environment. Follow the [Mooncake installation guide](../features/pd_disaggregation_mooncake_single_node.md#install-mooncake), using a Mooncake version compatible with the installed vLLM-Ascend release. Replace `192.168.0.1` and `eth0` with the serving host address and network interface. Provide sufficient host memory for Engram CPU offload and shared memory.

    Run each rank command in a separate terminal. Start the proxy after all four APIs are ready. Port 8087 is also used by the colocated example, so stop that service before starting this proxy.

    **Prefill:**

    Rank 0:

    ```bash
    unset CPLUS_INCLUDE_PATH CPATH C_INCLUDE_PATH
    unset ftp_proxy FTP_PROXY https_proxy HTTPS_PROXY http_proxy HTTP_PROXY
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh

    export VLLM_HOST_IP="192.168.0.1"
    export HCCL_IF_IP="192.168.0.1"
    export GLOO_SOCKET_IFNAME="eth0"
    export TP_SOCKET_IFNAME="eth0"
    export HCCL_SOCKET_IFNAME="eth0"
    export ASCEND_RT_VISIBLE_DEVICES=0,1
    export VLLM_SERVER_DEV_MODE=1
    export VLLM_USE_V2_MODEL_RUNNER=1
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
    export VLLM_ENGINE_READY_TIMEOUT_S=4800
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export TASK_QUEUE_ENABLE=1
    export HCCL_BUFFSIZE=1024
    export HCCL_NPU_SOCKET_PORT_RANGE=auto
    export LD_LIBRARY_PATH="$(python -c 'import mooncake; print(next(iter(mooncake.__path__)))'):${LD_LIBRARY_PATH:-}"

    vllm serve /models/Qwen3.8-Flash-Next-w4a8-mxfp \
        --host 0.0.0.0 \
        --port 8088 \
        --served-model-name qwen38-flash-next-950-dt-pd \
        --trust-remote-code \
        --quantization ascend \
        --tensor-parallel-size 2 \
        --data-parallel-size 2 \
        --data-parallel-size-local 1 \
        --data-parallel-rank 0 \
        --data-parallel-address 192.168.0.1 \
        --data-parallel-rpc-port 13388 \
        --max-model-len 141312 \
        --max-num-seqs 64 \
        --max-num-batched-tokens 4096 \
        --gpu-memory-utilization 0.90 \
        --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
        --enable-prefix-caching \
        --async-scheduling \
        --enable-expert-parallel \
        --compilation-config '{"cudagraph_mode":"NONE"}' \
        --additional-config '{"enable_cpu_binding":true,"enable_shared_expert_dp":true,"enable_fused_mc2":0,"enable_flashcomm1":true,"multistream_overlap_shared_expert":true,"ascend_compilation_config":{"fuse_norm_quant":false,"enable_npugraph_ex":true,"enable_static_kernel":false,"enable_super_kernel":false}}' \
        --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}' \
        --profiler-config '{"profiler":"torch","torch_profiler_dir":"/tmp/qwen38-950-dt-pd/prefill/rank0","torch_profiler_with_stack":false}' \
        --kv-transfer-config '{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_producer","kv_port":30000,"kv_connector_extra_config":{"prefill":{"dp_size":2,"tp_size":2},"decode":{"dp_size":2,"tp_size":2}}}'
    ```

    Rank 1:

    ```bash
    unset CPLUS_INCLUDE_PATH CPATH C_INCLUDE_PATH
    unset ftp_proxy FTP_PROXY https_proxy HTTPS_PROXY http_proxy HTTP_PROXY
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh

    export VLLM_HOST_IP="192.168.0.1"
    export HCCL_IF_IP="192.168.0.1"
    export GLOO_SOCKET_IFNAME="eth0"
    export TP_SOCKET_IFNAME="eth0"
    export HCCL_SOCKET_IFNAME="eth0"
    export ASCEND_RT_VISIBLE_DEVICES=2,3
    export VLLM_SERVER_DEV_MODE=1
    export VLLM_USE_V2_MODEL_RUNNER=1
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
    export VLLM_ENGINE_READY_TIMEOUT_S=4800
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export TASK_QUEUE_ENABLE=1
    export HCCL_BUFFSIZE=1024
    export HCCL_NPU_SOCKET_PORT_RANGE=auto
    export LD_LIBRARY_PATH="$(python -c 'import mooncake; print(next(iter(mooncake.__path__)))'):${LD_LIBRARY_PATH:-}"

    vllm serve /models/Qwen3.8-Flash-Next-w4a8-mxfp \
        --host 0.0.0.0 \
        --port 8089 \
        --served-model-name qwen38-flash-next-950-dt-pd \
        --trust-remote-code \
        --quantization ascend \
        --tensor-parallel-size 2 \
        --data-parallel-size 2 \
        --data-parallel-size-local 1 \
        --data-parallel-rank 1 \
        --data-parallel-address 192.168.0.1 \
        --data-parallel-rpc-port 13388 \
        --max-model-len 141312 \
        --max-num-seqs 64 \
        --max-num-batched-tokens 4096 \
        --gpu-memory-utilization 0.90 \
        --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
        --enable-prefix-caching \
        --async-scheduling \
        --enable-expert-parallel \
        --compilation-config '{"cudagraph_mode":"NONE"}' \
        --additional-config '{"enable_cpu_binding":true,"enable_shared_expert_dp":true,"enable_fused_mc2":0,"enable_flashcomm1":true,"multistream_overlap_shared_expert":true,"ascend_compilation_config":{"fuse_norm_quant":false,"enable_npugraph_ex":true,"enable_static_kernel":false,"enable_super_kernel":false}}' \
        --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}' \
        --profiler-config '{"profiler":"torch","torch_profiler_dir":"/tmp/qwen38-950-dt-pd/prefill/rank1","torch_profiler_with_stack":false}' \
        --kv-transfer-config '{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_producer","kv_port":30000,"kv_connector_extra_config":{"prefill":{"dp_size":2,"tp_size":2},"decode":{"dp_size":2,"tp_size":2}}}'
    ```

    **Decode:**

    Rank 0:

    ```bash
    unset CPLUS_INCLUDE_PATH CPATH C_INCLUDE_PATH
    unset ftp_proxy FTP_PROXY https_proxy HTTPS_PROXY http_proxy HTTP_PROXY
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh

    export VLLM_HOST_IP="192.168.0.1"
    export HCCL_IF_IP="192.168.0.1"
    export GLOO_SOCKET_IFNAME="eth0"
    export TP_SOCKET_IFNAME="eth0"
    export HCCL_SOCKET_IFNAME="eth0"
    export ASCEND_RT_VISIBLE_DEVICES=4,5
    export VLLM_SERVER_DEV_MODE=1
    export VLLM_USE_V2_MODEL_RUNNER=1
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
    export VLLM_ENGINE_READY_TIMEOUT_S=4800
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export TASK_QUEUE_ENABLE=1
    export HCCL_BUFFSIZE=1024
    export HCCL_NPU_SOCKET_PORT_RANGE=auto
    export LD_LIBRARY_PATH="$(python -c 'import mooncake; print(next(iter(mooncake.__path__)))'):${LD_LIBRARY_PATH:-}"

    vllm serve /models/Qwen3.8-Flash-Next-w4a8-mxfp \
        --host 0.0.0.0 \
        --port 8092 \
        --served-model-name qwen38-flash-next-950-dt-pd \
        --trust-remote-code \
        --quantization ascend \
        --tensor-parallel-size 2 \
        --data-parallel-size 2 \
        --data-parallel-size-local 1 \
        --data-parallel-rank 0 \
        --data-parallel-address 192.168.0.1 \
        --data-parallel-rpc-port 13389 \
        --max-model-len 141312 \
        --max-num-seqs 64 \
        --max-num-batched-tokens 4096 \
        --gpu-memory-utilization 0.90 \
        --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
        --no-enable-prefix-caching \
        --async-scheduling \
        --enable-expert-parallel \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
        --additional-config '{"enable_cpu_binding":true,"enable_shared_expert_dp":true,"enable_fused_mc2":0,"enable_flashcomm1":true,"multistream_overlap_shared_expert":true,"ascend_compilation_config":{"fuse_norm_quant":false,"enable_npugraph_ex":true,"enable_static_kernel":true,"enable_super_kernel":false},"scheduler_config":{"recompute_scheduler_enable":true}}' \
        --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}' \
        --profiler-config '{"profiler":"torch","torch_profiler_dir":"/tmp/qwen38-950-dt-pd/decode/rank0","torch_profiler_with_stack":false}' \
        --kv-transfer-config '{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_consumer","kv_port":30100,"kv_connector_extra_config":{"prefill":{"dp_size":2,"tp_size":2},"decode":{"dp_size":2,"tp_size":2}}}'
    ```

    Rank 1:

    ```bash
    unset CPLUS_INCLUDE_PATH CPATH C_INCLUDE_PATH
    unset ftp_proxy FTP_PROXY https_proxy HTTPS_PROXY http_proxy HTTP_PROXY
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh

    export VLLM_HOST_IP="192.168.0.1"
    export HCCL_IF_IP="192.168.0.1"
    export GLOO_SOCKET_IFNAME="eth0"
    export TP_SOCKET_IFNAME="eth0"
    export HCCL_SOCKET_IFNAME="eth0"
    export ASCEND_RT_VISIBLE_DEVICES=6,7
    export VLLM_SERVER_DEV_MODE=1
    export VLLM_USE_V2_MODEL_RUNNER=1
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
    export VLLM_ENGINE_READY_TIMEOUT_S=4800
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export TASK_QUEUE_ENABLE=1
    export HCCL_BUFFSIZE=1024
    export HCCL_NPU_SOCKET_PORT_RANGE=auto
    export LD_LIBRARY_PATH="$(python -c 'import mooncake; print(next(iter(mooncake.__path__)))'):${LD_LIBRARY_PATH:-}"

    vllm serve /models/Qwen3.8-Flash-Next-w4a8-mxfp \
        --host 0.0.0.0 \
        --port 8093 \
        --served-model-name qwen38-flash-next-950-dt-pd \
        --trust-remote-code \
        --quantization ascend \
        --tensor-parallel-size 2 \
        --data-parallel-size 2 \
        --data-parallel-size-local 1 \
        --data-parallel-rank 1 \
        --data-parallel-address 192.168.0.1 \
        --data-parallel-rpc-port 13389 \
        --max-model-len 141312 \
        --max-num-seqs 64 \
        --max-num-batched-tokens 4096 \
        --gpu-memory-utilization 0.90 \
        --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
        --no-enable-prefix-caching \
        --async-scheduling \
        --enable-expert-parallel \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
        --additional-config '{"enable_cpu_binding":true,"enable_shared_expert_dp":true,"enable_fused_mc2":0,"enable_flashcomm1":true,"multistream_overlap_shared_expert":true,"ascend_compilation_config":{"fuse_norm_quant":false,"enable_npugraph_ex":true,"enable_static_kernel":true,"enable_super_kernel":false},"scheduler_config":{"recompute_scheduler_enable":true}}' \
        --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}' \
        --profiler-config '{"profiler":"torch","torch_profiler_dir":"/tmp/qwen38-950-dt-pd/decode/rank1","torch_profiler_with_stack":false}' \
        --kv-transfer-config '{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_consumer","kv_port":30100,"kv_connector_extra_config":{"prefill":{"dp_size":2,"tp_size":2},"decode":{"dp_size":2,"tp_size":2}}}'
    ```

    **Proxy:**

    Run the following command from the vLLM-Ascend repository root after all four model APIs are ready:

    ```bash
    unset ftp_proxy FTP_PROXY https_proxy HTTPS_PROXY http_proxy HTTP_PROXY

    python examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
        --host 192.168.0.1 \
        --port 8087 \
        --prefiller-hosts 192.168.0.1 192.168.0.1 \
        --prefiller-ports 8088 8089 \
        --decoder-hosts 192.168.0.1 192.168.0.1 \
        --decoder-ports 8092 8093
    ```

    **Service Verification:**

    Check that all four APIs list `qwen38-flash-next-950-dt-pd` before starting the proxy:

    ```bash
    curl --fail http://192.168.0.1:8088/v1/models
    curl --fail http://192.168.0.1:8089/v1/models
    curl --fail http://192.168.0.1:8092/v1/models
    curl --fail http://192.168.0.1:8093/v1/models
    ```

    Once the proxy is listening, verify the complete PD path:

    ```bash
    curl --fail http://192.168.0.1:8087/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model":"qwen38-flash-next-950-dt-pd","messages":[{"role":"user","content":"Who are you?"}],"max_tokens":1024,"temperature":0}'
    ```

    **Key parameters:**

    - Each command starts one DP rank with TP2. Keep each role's DP address, RPC port, and model settings consistent across its two ranks.
    - Both roles enable Model Runner V2, async scheduling, and expert parallelism (EP). EP is distinct from expert parallel load balancing (EPLB); these commands do not enable EPLB.
    - `cpu_offload=true` keeps Engram embedding tables in host memory; `dp_shared_memory=true` shares the corresponding table shards among node-local DP replicas. Both options are booleans. DP replicas must share the same host and shared-memory namespace; this DP2 configuration satisfies the DP > 1 requirement and does not enable elastic EP.
    - Prefill enables Prefix Caching, which automatically selects Mamba cache mode `align` from the default `none`, and uses `cudagraph_mode=NONE`; static kernels are disabled.
    - Decode disables Prefix Caching and enables `FULL_DECODE_ONLY`, static kernels, and recompute scheduling.
    - Both roles enable CPU binding, shared expert DP, FlashComm1, shared expert multistream overlap, and NPU graph extension. Fused MC2, norm/quantization fusion, and super kernels are disabled.
    - Both roles use three MTP draft tokens and omit `enforce_eager`, leaving the default override unset. This does not require identical graph execution: prefill uses `cudagraph_mode=NONE`, while decode uses `FULL_DECODE_ONLY`. Keep these role-specific settings.
    - Keep the Mooncake DP2 / TP2 declarations synchronized with the service topology and route both ranks of each role through the proxy. `kv_port` is a base port shared by ranks of the same role: the connector adds the DP serving index × TP × PP × PCP, then the local device index. With TP2 and PP=PCP=1, prefill rank 0 uses ports 30000–30001 and rank 1 uses 30002–30003; decode uses 30100–30103. Do not increment the configured base port for rank 1.
    - The profiler configuration sets trace output paths without starting profiling.
    - For Function Calling and reasoning parsing, add `--enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3` to all four model commands and verify proxy forwarding. Validate multimodal input and thinking control through the proxy with the selected checkpoint.

## 6 Functional Verification

After the service is started, the model can be invoked by sending a prompt. The `chat/completions` API is supported. Use the `--served-model-name` you configured (`qwen3.8-flash-next` for `Qwen3.8-Flash-Next`).

Use the API address and served model name for your deployment:

| Deployment | API address | Model name |
| --- | --- | --- |
| A2 series colocated | `http://127.0.0.1:8088` | `qwen3.8-flash-next` |
| A3 series colocated | `http://127.0.0.1:8088` | `qwen3.8-flash-next` |
| Ascend 950 DT colocated | `http://127.0.0.1:8087` | `qwen3.8-flash-next-950-dt` |
| A2 series PD proxy | `http://127.0.0.1:25570` | `qwen38-flash-next-a2-a3-pd` |
| A3 series PD proxy | `http://127.0.0.1:25570` | `qwen38-flash-next-a2-a3-pd` |
| Ascend 950 DT PD proxy | `http://192.168.0.1:8087` | `qwen38-flash-next-950-dt-pd` |

For PD, first complete the proxy check and review the key parameters in [Section 5.2](#52-pd-separation-deployment).

### 6.1 Basic Chat Completion

```bash
curl http://127.0.0.1:8088/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3.8-flash-next",
        "messages": [
            {"role": "user", "content": "Who are you?"}
        ],
        "max_tokens": 1024,
        "temperature": 0
    }'
```

Expected result: the service returns HTTP 200 OK.

### 6.2 Multimodal Chat Completion

Use a W4A8 checkpoint that includes the vision encoder to verify image-and-text input. Replace `<IMAGE_URL>` with an image URL accessible from the serving environment:

```bash
curl http://127.0.0.1:8088/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3.8-flash-next",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "<IMAGE_URL>"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Describe this image."
                    }
                ]
            }
        ],
        "max_tokens": 1024,
        "temperature": 0
    }'
```

Expected result: the service returns HTTP 200 OK and describes the supplied image in `choices[0].message.content`.

### 6.3 Function Calling

Function Calling requires the following startup options, which are included in the colocated examples in Section 5; for PD separation, see [Section 5.2](#52-pd-separation-deployment):

```text
--enable-auto-tool-choice --tool-call-parser qwen3_xml
```

Send a request containing the available tools and set `tool_choice` to `auto`:

```bash
curl http://127.0.0.1:8088/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3.8-flash-next",
        "messages": [
            {
                "role": "user",
                "content": "What is the weather in Beijing?"
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "City name"
                            }
                        },
                        "required": ["city"]
                    }
                }
            }
        ],
        "tool_choice": "auto",
        "max_tokens": 1024,
        "temperature": 0
    }'
```

When the model chooses the function, the parsed result appears in `choices[0].message.tool_calls`. The caller must execute the function and send its result back to the model in a follow-up message.

### 6.4 Reasoning Parser and Thinking Control

Reasoning parsing requires the following startup option, which is included in the colocated examples in Section 5; for PD separation, see [Section 5.2](#52-pd-separation-deployment):

```text
--reasoning-parser qwen3
```

Qwen3.8-Flash-Next uses thinking mode by default. To request reasoning explicitly, set `enable_thinking` to `true` (or omit it). The parser returns the reasoning separately from the final answer:

```bash
curl http://127.0.0.1:8088/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3.8-flash-next",
        "messages": [
            {
                "role": "user",
                "content": "Who are you?"
            }
        ],
        "max_tokens": 1024,
        "temperature": 0,
        "top_p": 1,
        "chat_template_kwargs": {
            "enable_thinking": true
        }
    }'
```

The final answer is returned in `choices[0].message.content`, while the parsed reasoning is returned separately in the `reasoning_content` field of the response.

To disable thinking and request a direct answer, use the following request:

```bash
curl http://127.0.0.1:8088/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3.8-flash-next",
        "messages": [
            {
                "role": "user",
                "content": "Who are you?"
            }
        ],
        "max_tokens": 1024,
        "temperature": 0,
        "top_p": 1,
        "chat_template_kwargs": {
            "enable_thinking": false
        }
    }'
```

## 7 Accuracy Evaluation

### 7.1 Using AISBench

Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for evaluation setup and usage. The result below is for reference only.

The following result was recorded for the original A3 series W8A8 TP8 × DP1 configuration. It is historical reference data and does not establish the accuracy of any current W4A8 or W4A8-MXFP deployment example in this guide.

| Hardware | Dataset | Metric | Score |
| --- | --- | --- | --- |
| A3 series | GPQA Diamond | Accuracy | 90.4 |

## 8 Performance Evaluation

### 8.1 Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### 8.2 Using vLLM Benchmark

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

## 9 Performance Tuning

### 9.1 Recommended Configurations

Use the configuration for your hardware and deployment mode. The table summarizes the commands in [Section 5](#5-online-service-deployment); tune context length, concurrency, and memory utilization for the target workload.

| Hardware | Deployment / role | Devices | TP × DP | Max model length | Max sequences | Batch tokens |
| --- | --- | --- | --- | --- | --- | --- |
| A2 series | Colocated | 0–7 | 8 × 1 | 131072 | 8 | 4096 |
| A3 series | Colocated | 0–7 | 8 × 1 | 131072 | 8 | 4096 |
| Ascend 950 DT | Colocated | 0–3 | 4 × 1 | 141312 | 32 | 4096 |
| A2 series | Prefill | 0–7 on prefill host | 2 × 4 | 141312 | 32 | 4096 |
| A3 series | Prefill | 0–7 | 2 × 4 | 141312 | 32 | 4096 |
| A2 series | Decode | 0–7 on decode host | 2 × 4 | 141312 | 32 | 4096 |
| A3 series | Decode | 8–15 | 2 × 4 | 141312 | 32 | 4096 |
| Ascend 950 DT | Prefill | 0–3 | 2 × 2 | 141312 | 64 | 4096 |
| Ascend 950 DT | Decode | 4–7 | 2 × 2 | 141312 | 64 | 4096 |

### 9.2 Tuning Guidelines

Please refer to the [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md) for tuning methods.

Please refer to the [Feature Matrix](../../user_guide/support_matrix/feature_matrix.md) for detailed feature descriptions.

### 9.3 Model Runner V2 Decode Submission

Audited Qwen3.8 Flash Next Model Runner V2 consumers automatically use optimistic CPU length bounds while exact execution lengths remain on device. This capability is enabled by default for supported consumers and requires no configuration switch. The resolved target architecture must be `Qwen4ExpForCausalLM` or `Qwen4ExpForConditionalGeneration`. Other models retain their existing behavior.

The model checks the initialized target and MTP draft attention consumers once after KV initialization. A supported model skips both the corrected-count D2H copy and its CPU synchronization, using the runner's existing paired exemption. Exact positions, accepted counts, KV slots and recurrent state still use device state. Existing connector loads, waits, acknowledgements and errors are unchanged.

The same capability covers text and ordinary non-realtime multimodal requests. Completed prompts use the existing encoder gather fast path; there are no extra media certificates or PD observers.

Unknown target/draft consumers, context parallelism, sparse KV offload and realtime media retain exact CPU synchronization. The runner's existing PP exact-count protection still applies. There is no extra TP8, DP1, PP1, SP-off, prefix-cache-off, PLE weight-offload, FULL-graph or fixed-K requirement. Underlying model/runtime restrictions still apply.

The decision is fixed for the model state's initialized lifetime so production and consumption cannot switch independently between batches. Changes to runtime configuration or attention consumers require model-state reinitialization. Initialization drains the old copy stream once before enabling the exemption.

## 10 FAQ

For common environment, installation, and parameter issues, refer to the [Public FAQs](../../faqs.md).
