# GLM-5.3

## 1 Introduction

[GLM-5.3](https://www.modelscope.cn/models/ZhipuAI/GLM-5.3) is a Mixture-of-Experts model with DeepSeek-style sparse attention. Unlike GLM-5.1 and GLM-5.2, which are published in BF16 and quantized with ModelSlim before serving, GLM-5.3 ships **natively FP8-quantized**: weights are stored as `float8_e4m3fn` with a `weight_scale_inv` holding one scale per 128x128 block. vLLM Ascend serves those weights directly, so no `--quantization` flag and no offline quantization step are needed.

This document covers the deployment scenario validated for this model on Ascend 950 products.

## 2 Supported Features

Refer to [Supported Features List](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [Feature Guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## 3 Prerequisites

### 3.1 Model Weight

`GLM-5.3` requires 1 Atlas 950 node (8 NPUs). [Download model weight](https://www.modelscope.cn/models/ZhipuAI/GLM-5.3).

The checkpoint is already FP8-quantized. Do not pass `--quantization`: the `quantization_config` block in `config.json` selects the native FP8 path automatically.

## 4 Installation

### 4.1 Docker Image Installation

```shell
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
    -v /usr/lib64:/usr/lib64 \
    -it $IMAGE bash
```

!!! warning
    Mount the whole `/usr/local/Ascend/driver` directory as shown above. Mounting only `lib64` and `version.info` leaves out `driver/topo`, and HCCL then fails to initialize with `Config_Error_Ranktable(EI0014): ... topo_file_path is invalid`.

### 4.2 Source Code Installation

If you don't want to use the docker image as above, you can also build all from source:

- Install `vllm-ascend` from source, refer to [installation](../../installation.md).

## 5 Deployment

### 5.1 Single-Node Deployment on Ascend 950

Run the following script to execute online inference.

```shell
export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE="AIV"
export TASK_QUEUE_ENABLE=1
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

vllm serve /root/.cache/modelscope/hub/models/ZhipuAI/GLM-5.3 \
--served-model-name glm-53 \
--tensor-parallel-size 8 \
--trust-remote-code \
--max-model-len 131072 \
--max-num-seqs 16 \
--block-size 512 \
--gpu-memory-utilization 0.9 \
--speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

Key Parameter Descriptions:

- `--tensor-parallel-size 8`: TP8 over the 8 NPUs of one node. Expert parallel is deliberately not enabled, see [Known Limitations](#7-known-limitations).
- `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'`: Enables Multi-Token Prediction. GLM-5.3 declares `num_nextn_predict_layers: 1`, so one speculative token per step is the ceiling for this checkpoint.
- No `--quantization` flag: the checkpoint carries its own block-wise FP8 `quantization_config`. On Ascend 950 the resolved weights are re-quantized to MXFP8 at load time; on other hardware they are served in the model dtype.

To enable function calling, add:

```shell
--tool-call-parser glm47 \
--reasoning-parser glm45 \
--enable-auto-tool-choice
```

## 6 Functional Verification

Once your server is started, you can query the model with input prompts:

```shell
curl http://<node_ip>:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "glm-53",
        "messages": [{"role": "user", "content": "The future of AI is"}],
        "max_tokens": 50,
        "temperature": 0
    }'
```

To confirm that MTP is actually contributing, watch the `SpecDecoding metrics` line in the server log. A healthy run reports an acceptance rate well above zero; a rate of exactly `0.0%` means the draft head is producing unusable logits rather than merely disagreeing with the target.

## 7 Known Limitations

- **Expert parallel is not supported on this platform.** `--enable-expert-parallel` selects the MC2 fused dispatch path, which needs an inter-card link protocol that Atlas 350 topologies do not provide; the operator fails while acquiring HCCL communication resources (`GetHcclCommLink: No matching communication protocol found`). Serve GLM-5.3 with tensor parallelism only.
- **Speculative decoding runs the draft model eagerly.** As with the rest of the GLM series, the draft model is switched to eager mode automatically; the target model still uses graph mode.

## 8 Accuracy and Performance Evaluation

Not evaluated yet for this model. Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) and [vLLM benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for the general procedures.
