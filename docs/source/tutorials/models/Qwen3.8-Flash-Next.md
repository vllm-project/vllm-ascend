# Qwen3.8-Flash-Next (Experimental)

## 1 Introduction

Qwen3.8-Flash-Next is a multimodal Mixture-of-Experts (MoE) model and an experimental preview of the architecture that will underpin Qwen4. Its language model combines Gated DeltaNet (GDN) and Qwen Sparse Attention (QSA), gated residual connections, Position Learning Enhancement (PLE), and a native Multi-Token Prediction (MTP) head.

This tutorial describes model preparation, installation, online serving, functional verification, and evaluation. It covers single-node colocated deployment on Atlas 800 A3 and Ascend 950 DT, and prefill/decode (PD) separation on Atlas 800 A3.

Qwen3.8-Flash-Next was first supported in `vllm-ascend:v0.26.0rc`. The original A3 W8A8 configuration was validated with text and multimodal input. The deployment examples below include additional performance settings and require separate runtime and accuracy validation.

!!! warning

    Automatic Prefix Caching is experimental. For A3 colocated deployment, enable it with `--enable-prefix-caching --mamba-cache-mode align` and revalidate NPU memory capacity for the target context length and concurrency.

## 2 Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_features.md) to get the model's supported feature matrix.

Refer to [Feature Guide](../../user_guide/feature_guide/index.md) to get feature configuration details.

## 3 Prerequisites

### 3.1 Model Weight

Select the checkpoint that matches the deployment configuration:

| Hardware | Deployment | Checkpoint | Device allocation | Download Links |
| --- | --- | --- | --- | --- |
| Atlas 800 A3 | Colocated | `Qwen3.8-Flash-Next-w8a8-mtp` | 8 devices, TP8 × DP1 | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/Qwen3.8-Flash-Next-w8a8-mtp) |
| Atlas 800 A3 | PD separation | Ascend-compatible W4A8 | 16 devices, TP2 × DP4 per role | |
| Ascend 950 DT | Colocated | `Qwen3.8-Flash-Next-w8a8-mxfp8` | 4 devices, TP4 × DP1 | |

The commands use fixed model directories under `/models`. Mount the corresponding checkpoint at the shown directory or edit that path directly. For PD separation, both roles must load the same checkpoint.

## 4 Installation

### 4.1 Docker Image Installation

The following images apply to the A3 W8A8 baseline. For 950 DT, prepare an environment with compatible vLLM and vLLM-Ascend versions; do not use an A3 image for that deployment.

A pre-built Qwen3.8 A3 image is available in the [vllm-ascend repository](https://quay.io/repository/ascend/vllm-ascend?tab=tags&tag=latest). Select the image that matches the host operating system:

- openEuler: `quay.io/ascend/vllm-ascend:qwen3.8-next-a3-openeuler`
- Ubuntu: `quay.io/ascend/vllm-ascend:qwen3.8-next-a3`

The following example uses the AArch64 image and exposes 16 devices on an Atlas 800 A3 node. The service command in [Section 5.1](#51-single-node-online-deployment) uses the first eight devices; the A3 PD example in Section 5.2 uses all 16.

=== "A3 Series"

    Start the docker image on each node.

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
        -v /path/to/models:/models \
        -it quay.io/ascend/vllm-ascend:qwen3.8-next-a3 bash
    ```

After entering the container, verify that vLLM and vLLM-Ascend can be imported:

```shell
python -c "import vllm, vllm_ascend; print('vllm and vllm-ascend are ready')"
```

### 4.2 Source Code Installation

If you don't want to use the docker image as above, you can also build all from source:

- Install `vllm-ascend` from source, refer to [installation](../../getting_started/installation.md#installation-existing-cann-install).

If you want to deploy a multi-node environment, you need to set up the environment on each node.

## 5 Online Service Deployment {: #5-online-service-deployment }

### 5.1 Single-Node Online Deployment

Prefill and decode run in the same service. Select the hardware tab below.

Before starting:

- Mount the matching checkpoint at the model directory shown in the command.
- Network examples use `192.168.0.1` and `eth0`. Replace these literals with the serving host address and its network interface.
- Verify available NPU memory before changing context length, concurrency, or memory utilization.

=== "A3 Series"

    Run the following command to serve the W8A8 model with TP8 × DP1 and a maximum context length of 131072 tokens.

    ```bash
    unset CPLUS_INCLUDE_PATH CPATH C_INCLUDE_PATH
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh

    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export VLLM_USE_V2_MODEL_RUNNER=1
    export VLLM_ASCEND_PLE_BACKEND=torch
    export VLLM_ASCEND_ENABLE_QSA_LIGHTNING_INDEXER=1
    export VLLM_ASCEND_ENABLE_QSA_E3V=1
    unset VLLM_ASCEND_FORCE_QSA_REFERENCE

    vllm serve /models/Qwen3.8-Flash-Next-w8a8-mtp \
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
        --compilation-config '{"cudagraph_capture_sizes":[4,8,12,16,20,24,28,32],"cudagraph_mode":"FULL_DECODE_ONLY"}' \
        --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":true}' \
        --additional-config '{"enable_cpu_binding":true,"enable_shared_expert_dp":true,"enable_fused_mc2":0,"enable_flashcomm1":true,"multistream_overlap_shared_expert":true,"ascend_compilation_config":{"fuse_norm_quant":false,"enable_npugraph_ex":true,"enable_static_kernel":false,"enable_super_kernel":false}}'
    ```

    **Key parameters:**

    - `VLLM_ASCEND_ENABLE_QSA_LIGHTNING_INDEXER=1` enables the fused QSA Lightning Indexer path.
    - `VLLM_ASCEND_ENABLE_QSA_E3V=1` enables the fused QSA Expand E3 path.
    - `--quantization ascend` loads the Ascend-compatible W8A8 checkpoint.
    - `--tensor-parallel-size 8` and `--data-parallel-size 1` configure the A3 single-DP TP8 topology.
    - `--max-model-len 131072` caps the maximum context length. Common values: `65536` (64K), `131072` (128K).
    - `--enable-expert-parallel` enables expert parallelism for the MoE layers.
    - `--enable-prefix-caching --mamba-cache-mode align` enables Prefix Caching with aligned GDN and PLE state checkpoints.
    - The command does not set `--language-model-only`, so the vision encoder remains enabled and the service accepts both text and multimodal requests. For a text-only deployment, you may add `--language-model-only` to skip loading the vision encoder.
    - `--enable-auto-tool-choice --tool-call-parser qwen3_xml` enables automatic Function Calling with the Qwen3 XML parser.
    - `--reasoning-parser qwen3` separates reasoning from the final answer in the OpenAI-compatible response.
    - `--speculative-config` uses the model's MTP head to draft three tokens. `enforce_eager=true` keeps the MTP proposer in eager mode.
    - `--compilation-config` enables `FULL_DECODE_ONLY` ACLGraph for decode.
    - `fuse_norm_quant=false` disables norm/quantization fusion for this W8A8 deployment.
    - `VLLM_USE_V2_MODEL_RUNNER=1` selects Model Runner V2; `VLLM_ASCEND_PLE_BACKEND=torch` selects the PLE backend.
    - `--async-scheduling` enables asynchronous scheduling.
    - `--additional-config` enables CPU binding, FlashComm1, shared expert DP, and shared expert multistream overlap. Shared expert DP requires expert parallelism and TP > 1. Keep `enable_fused_mc2=0` when using shared expert overlap.
    - `enable_npugraph_ex=true` enables NPU graph extension; static kernels and super kernels remain disabled.

=== "950 DT"

    The W8A8-MXFP8 example uses TP4 × DP1, Model Runner V2, async scheduling, FlashComm1, shared expert DP, shared expert multistream overlap, and MTP. Static kernels and super kernels are disabled.

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
    export VLLM_ENGINE_READY_TIMEOUT_S=2400
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export TASK_QUEUE_ENABLE=1
    export HCCL_BUFFSIZE=1024
    export HCCL_NPU_SOCKET_PORT_RANGE=auto

    mkdir -p "/tmp/qwen38-next-950-dt-profiling"

    vllm serve /models/Qwen3.8-Flash-Next-w8a8-mxfp8 \
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
    - `--max-model-len 141312` and `--max-num-seqs 32` set context and concurrency limits; verify memory capacity before increasing them.
    - `enable_fused_mc2=0` preserves shared expert multistream overlap.
    - `FULL_DECODE_ONLY` enables decode graph capture. The MTP configuration uses three draft tokens without explicitly forcing eager execution.
    - Prefix caching is enabled without an explicit Mamba cache mode. Confirm cache behavior with the installed runtime.
    - `--profiler-config` sets the trace output directory; it does not start profiling by itself.

**Service Verification:**

Wait for model loading and graph capture to complete, then check the service from the serving host. Use port `8088` for A3 or `8087` for 950 DT:

```bash
curl --fail http://127.0.0.1:8088/v1/models
```

The response should list the configured model name. After startup, use [Section 6](#6-functional-verification) to verify generation.

### 5.2 PD Separation Deployment

On a single Atlas 800 A3 host, prefill and decode each use eight devices with TP2 × DP4. Both roles load the same W4A8 checkpoint and enable expert parallelism, async scheduling, and Model Runner V2. MooncakeConnectorV2 transfers KV cache between them; requests enter the proxy on port 25570.

| Setting | Prefill | Decode |
| --- | --- | --- |
| Visible devices | 0–7 | 8–15 |
| TP × DP | 2 × 4 | 2 × 4 |
| API port | 25568 | 25569 |
| Mooncake role / KV port | `kv_producer` / 32600 | `kv_consumer` / 32700 |
| HCCL socket port range | 61400–61499 | 61500–61599 |
| HCCL buffer size | 4096 | 512 |
| Maximum context / sequences / batched tokens | 141312 / 32 / 4096 | 141312 / 32 / 4096 |

Before starting:

- Expose devices 0–15 and mount the same W4A8 checkpoint for both roles.
- Install compatible vLLM, vLLM-Ascend, and Mooncake in the active Python environment.
- The scripts use `192.168.0.1`, `eth0`, and `/models/Qwen3.8-Flash-Next-w4a8`. Replace these literals consistently for the target host. Logs and profiling traces are written to `/tmp/qwen38-a3-pd`. Provide sufficient host memory for Engram CPU offload.
- Place the compatible `PD_proxy_server.py` at `/opt/qwen38/PD_proxy_server.py`. The proxy must support Mooncake PD request routing and the options shown below; its implementation is not included in this guide.
- The examples use the Mooncake library directory from the Python 3.12.16 environment. Edit the literal library path if Mooncake is installed elsewhere.

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
export VLLM_ENGINE_READY_TIMEOUT_S=2400
export VLLM_ASCEND_PLE_BACKEND=torch
export VLLM_HOST_IP="192.168.0.1"
export HCCL_IF_IP="192.168.0.1"
export GLOO_SOCKET_IFNAME="eth0"
export TP_SOCKET_IFNAME="eth0"
export HCCL_SOCKET_IFNAME="eth0"
export no_proxy="localhost,127.0.0.1,192.168.0.1"
export NO_PROXY="localhost,127.0.0.1,192.168.0.1"
export LD_LIBRARY_PATH="/usr/local/python3.12.16/lib/python3.12/site-packages/mooncake:${LD_LIBRARY_PATH:-}"
mkdir -p "/tmp/qwen38-a3-pd" "/tmp/qwen38-a3-pd/profiling/prefill" "/tmp/qwen38-a3-pd/profiling/decode"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_NPU_SOCKET_PORT_RANGE=61400-61499
export HCCL_BUFFSIZE=4096

exec vllm serve /models/Qwen3.8-Flash-Next-w4a8 \
    --host 0.0.0.0 \
    --port 25568 \
    --served-model-name qwen38-flash-next-a3-pd \
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
    --engram-config '{"cpu_offload":true,"dp_shared_memory":false}' \
    --additional-config '{"enable_shared_expert_dp":true,"enable_flashcomm1":true,"enable_prefill_mc2":true,"enable_fused_mc2":0,"enable_cpu_binding":true,"multistream_overlap_shared_expert":true,"ascend_compilation_config":{"fuse_norm_quant":false}}' \
    --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":true}' \
    --profiler-config '{"profiler":"torch","torch_profiler_dir":"/tmp/qwen38-a3-pd/profiling/prefill"}' \
    --kv-transfer-config '{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_producer","kv_port":"32600","kv_connector_extra_config":{"prefill":{"dp_size":4,"tp_size":2},"decode":{"dp_size":4,"tp_size":2}}}' \
    > "/tmp/qwen38-a3-pd/prefill.log" 2>&1
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
export VLLM_ENGINE_READY_TIMEOUT_S=2400
export VLLM_ASCEND_PLE_BACKEND=torch
export VLLM_HOST_IP="192.168.0.1"
export HCCL_IF_IP="192.168.0.1"
export GLOO_SOCKET_IFNAME="eth0"
export TP_SOCKET_IFNAME="eth0"
export HCCL_SOCKET_IFNAME="eth0"
export no_proxy="localhost,127.0.0.1,192.168.0.1"
export NO_PROXY="localhost,127.0.0.1,192.168.0.1"
export LD_LIBRARY_PATH="/usr/local/python3.12.16/lib/python3.12/site-packages/mooncake:${LD_LIBRARY_PATH:-}"
mkdir -p "/tmp/qwen38-a3-pd" "/tmp/qwen38-a3-pd/profiling/prefill" "/tmp/qwen38-a3-pd/profiling/decode"
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export HCCL_NPU_SOCKET_PORT_RANGE=61500-61599
export HCCL_BUFFSIZE=512

exec vllm serve /models/Qwen3.8-Flash-Next-w4a8 \
    --host 0.0.0.0 \
    --port 25569 \
    --served-model-name qwen38-flash-next-a3-pd \
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
    --engram-config '{"cpu_offload":true,"dp_shared_memory":false}' \
    --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,12,16,20,24,28,32],"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_flashcomm1":true,"enable_cpu_binding":true,"enable_shared_expert_dp":true,"multistream_overlap_shared_expert":true,"scheduler_config":{"recompute_scheduler_enable":true},"ascend_compilation_config":{"fuse_norm_quant":false,"enable_npugraph_ex":true,"enable_static_kernel":true,"enable_super_kernel":false}}' \
    --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":false}' \
    --profiler-config '{"profiler":"torch","torch_profiler_dir":"/tmp/qwen38-a3-pd/profiling/decode"}' \
    --kv-transfer-config '{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_consumer","kv_port":"32700","kv_connector_extra_config":{"prefill":{"dp_size":4,"tp_size":2},"decode":{"dp_size":4,"tp_size":2}}}' \
    > "/tmp/qwen38-a3-pd/decode.log" 2>&1
```

**Proxy:**

```bash
#!/usr/bin/env bash
set -eo pipefail
mkdir -p "/tmp/qwen38-a3-pd"
unset ftp_proxy FTP_PROXY https_proxy HTTPS_PROXY http_proxy HTTP_PROXY
export no_proxy="localhost,127.0.0.1,192.168.0.1"
export NO_PROXY="localhost,127.0.0.1,192.168.0.1"

exec python /opt/qwen38/PD_proxy_server.py \
    --host 0.0.0.0 --port 25570 \
    --prefiller-host "192.168.0.1" --prefiller-port 25568 \
    --decoder-host "192.168.0.1" --decoder-port 25569 \
    > "/tmp/qwen38-a3-pd/proxy.log" 2>&1
```

Start prefill and decode:

```bash
nohup bash P.sh > p.launcher.log 2>&1 &
nohup bash D.sh > d.launcher.log 2>&1 &
```

Wait for both services to finish loading. On the serving host, check that both APIs respond successfully and list `qwen38-flash-next-a3-pd`:

```bash
curl --fail http://127.0.0.1:25568/v1/models
curl --fail http://127.0.0.1:25569/v1/models
```

After both checks succeed, start the proxy:

```bash
nohup bash proxy.sh > proxy.launcher.log 2>&1 &
```

Once the proxy is listening, verify the complete PD path:

```bash
curl --fail http://127.0.0.1:25570/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen38-flash-next-a3-pd",
        "messages": [{"role": "user", "content": "Who are you?"}],
        "max_tokens": 1024,
        "temperature": 0
    }'
```

Inspect the prefill, decode, and proxy logs under `/tmp/qwen38-a3-pd` if the request fails.

**Key parameters:**

- `VLLM_ASCEND_PLE_BACKEND=torch` selects the PLE backend for both roles.
- Both roles enable Prefix Caching without an explicit Mamba cache mode.
- Engram requires a checkpoint with supported n-gram layers. Both roles request `cpu_offload=true` and `dp_shared_memory=false`; Model Runner V2 with DP > 1 may enable sharing automatically, depending on the runtime version. Check the resolved configuration in the startup logs.
- Prefill enables `enable_prefill_mc2=true` and disables fused MC2. Verify MC2 token limits for TP2 × DP4 before increasing the batch token budget.
- Decode enables `FULL_DECODE_ONLY`, NPU graph extension, static kernels, and recompute scheduling. Super kernels remain disabled.
- Both roles use three MTP draft tokens. Prefill sets `enforce_eager=true`; decode sets `enforce_eager=false`.
- Keep fused MC2 disabled when using shared expert multistream overlap. The decode command relies on the runtime's default value of zero.
- Keep the Mooncake DP4 / TP2 declarations synchronized with both services' parallel settings.
- For Function Calling and reasoning parsing, add `--enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3` to both roles and verify proxy forwarding. Multimodal input and thinking control also require validation through the proxy with the W4A8 checkpoint.

## 6 Functional Verification

After the service is started, the model can be invoked by sending a prompt. The `chat/completions` API is supported. Use the `--served-model-name` you configured (`qwen3.8-flash-next` for `Qwen3.8-Flash-Next`).

The requests below use the A3 colocated endpoint. Select the endpoint and model name for the deployment being tested:

| Deployment | Port | Model name |
| --- | --- | --- |
| A3 colocated | 8088 | `qwen3.8-flash-next` |
| 950 DT colocated | 8087 | `qwen3.8-flash-next-950-dt` |
| A3 PD proxy | 25570 | `qwen38-flash-next-a3-pd` |

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

The A3 W8A8 model supports image-and-text input. Replace `<IMAGE_URL>` with an image URL accessible from the serving environment:

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

Function Calling requires the following startup options, which are included in the colocated examples in Section 5; for A3 PD, see [Section 5.2](#52-pd-separation-deployment):

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

Reasoning parsing requires the following startup option, which is included in the colocated examples in Section 5; for A3 PD, see [Section 5.2](#52-pd-separation-deployment):

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

1. Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for evaluation setup and usage.

2. After execution, you can get the result. Here are the results of `Qwen3.8-Flash-Next` in `vllm-ascend:v0.26.0rc` for reference only.

The following result was recorded for the original A3 W8A8 TP8 × DP1 configuration, before adding Model Runner V2, the explicit torch PLE backend, async scheduling, and MoE communication optimizations to the command in [Section 5.1](#51-single-node-online-deployment). It does not establish the accuracy of the updated command or the 950 DT and W4A8 PD configurations.

| Hardware | Dataset | Metric | Score |
| --- | --- | --- | --- |
| Atlas 800 A3 | GPQA Diamond | Accuracy | 90.4 |

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
| A3 | Colocated | 0–7 | 8 × 1 | 131072 | 8 | 4096 |
| 950 DT | Colocated | 0–3 | 4 × 1 | 141312 | 32 | 4096 |
| A3 | Prefill | 0–7 | 2 × 4 | 141312 | 32 | 4096 |
| A3 | Decode | 8–15 | 2 × 4 | 141312 | 32 | 4096 |

### 9.2 Tuning Guidelines

#### 9.2.1 General Tuning Reference

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
