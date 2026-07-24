# Mistral Large 3

## 1 Introduction

Mistral Large 3 is a granular Mixture-of-Experts model with 675B total
parameters and about 41B active parameters. Its text decoder contains 61
layers, 128 routed experts, four selected experts per token, one shared
expert, and MLA attention. The model configuration exposes a theoretical
maximum sequence length of 294,912 tokens; this guide uses 131,072 tokens as
the initial Ascend validation target.

This adapter covers the text `MistralLarge3ForCausalLM` architecture. The
vision encoder in the full checkpoint requires a separate multimodal model
integration and is not enabled by this class.

## 2 Supported Features

| Feature | Status | Notes |
| --- | --- | --- |
| Text generation | Supported by the adapter | Requires real-weight Ascend validation |
| MLA attention | Inherited from the vLLM DeepSeek-V3 model | Uses the Ascend attention backend |
| Expert parallelism | Supported by the model path | Enable with `--enable-expert-parallel` |
| FlashComm1 | Available for MoE validation | Enable with `VLLM_ASCEND_ENABLE_FLASHCOMM1=1` |
| ACLGraph | Available for validation | Eager mode is an isolation fallback |
| NVFP4 loading | Weight-name interface reserved | Ascend NVFP4 kernels are not implemented here |
| MTP | Checkpoint missing | The target checkpoint does not contain MTP layers |
| Multimodal input | Not supported by this class | Text decoder only |

## 3 Environment Preparation

The FP8 checkpoint is approximately 682 GB. An Atlas 800 A3 server with 16
NPUs is the recommended starting point. Confirm available memory before
raising context length or concurrency.

=== "A3 series"

    ```shell
    export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|-a3
    docker run --rm --name vllm-ascend --net=host --shm-size=1g \
        --privileged=true --device /dev/davinci_manager \
        --device /dev/devmm_svm --device /dev/hisi_hdc \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
        -v /root/.cache:/root/.cache -it ${IMAGE} bash
    ```

=== "A2 series"

    The full checkpoint normally needs multiple A2 nodes. Start the standard
    A2 image on every node and configure multi-node TP/EP before serving.

    ```shell
    export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
    docker run --rm --name vllm-ascend --net=host --shm-size=1g \
        --privileged=true --device /dev/davinci_manager \
        --device /dev/devmm_svm --device /dev/hisi_hdc \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
        -v /root/.cache:/root/.cache -it ${IMAGE} bash
    ```

## 4 Deployment

Run the service directly from `/workspace`. The command keeps the native
Mistral configuration and checkpoint loaders and enables the MoE feature
path first.

```shell
cd /workspace
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

vllm serve mistralai/Mistral-Large-3-675B-Instruct-2512 \
    --served-model-name mistral-large-3 \
    --tokenizer-mode mistral \
    --config-format mistral \
    --load-format mistral \
    --dtype bfloat16 \
    --tensor-parallel-size 16 \
    --enable-expert-parallel \
    --max-model-len 131072 \
    --max-num-seqs 16 \
    --port 8000
```

If graph capture is suspected, add `--enforce-eager` for isolation. Dummy
weights can check model construction with `--load-format dummy`, but they do
not validate checkpoint mapping or quantization and cannot replace a
real-weight run.

## 5 Functional Verification

Check readiness, then require a non-empty response from the first request:

```shell
curl -f http://127.0.0.1:8000/v1/models

curl http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "mistral-large-3",
      "messages": [{"role": "user", "content": "Say hello in Chinese."}],
      "temperature": 0,
      "max_tokens": 32
    }'
```

## 6 Accuracy Evaluation

The e2e configuration uses GPQA Diamond as a regression target. The value
below is the published checkpoint reference, not an Ascend measurement; it
must be replaced or confirmed after real-weight evaluation on the target
hardware.

| Dataset | Metric | Few-shot | Reference |
| --- | --- | ---: | ---: |
| GPQA Diamond | exact match | 0 | 0.6717 |

Run the repository model-evaluation workflow with
`tests/e2e/models/configs/Mistral-Large-3-675B-Instruct-2512.yaml` and record
the Ascend result before declaring the adaptation validated.

## 7 Performance

Start with `max-model-len=131072` and `max-num-seqs=16`. Record TTFT, TPOT,
throughput, peak NPU memory, ACLGraph replay evidence, and the result of both
EP and FlashComm1 runs. Increase concurrency only after the baseline passes.
