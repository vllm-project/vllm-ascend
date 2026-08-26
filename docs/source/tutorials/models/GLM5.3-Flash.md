# GLM-5.3-Flash

## 1 Introduction

[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) is a multimodal Mixture-of-Experts model with about 321B total parameters and 18B active parameters per token. The 45-layer language model interleaves Kimi Delta Attention (KDA) linear-attention layers with NoPE sparse MLA layers (34 KDA + 11 sparse MLA), routes each token through 8 of 288 experts, and includes a vision encoder plus one MTP draft layer. The published checkpoint is native block-wise FP8 (`quant_method: fp8`, `weight_block_size: [128, 128]`) with a 1,048,576-token context window.

This document focuses on serving the official FP8 weights on Ascend NPUs, following the same load-time path as other native FP8 checkpoints: **Ascend 950 re-quantizes the tiles to MXFP8; other generations resolve them to BF16**. Do not pass `--quantization ascend`.

vLLM must include GLM-5.3-Flash model support ([vLLM#53906](https://github.com/vllm-project/vllm/pull/53906), v0.27.0+). Use a vLLM-Ascend image that vendors that vLLM revision.

## 2 Supported Features

Refer to [Supported Features List](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [Feature Guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## 3 Prerequisites

### 3.1 Model Weight

- `GLM-5.3-Flash` (native FP8, ~306 GiB): recommended on **1 Ascend950PR series (128GB × 8)** node with TP=8. On Ascend 950 the weights stay one byte per element (MXFP8). [Download](https://huggingface.co/zai-org/GLM-5.3-Flash) or ModelScope `ZhipuAI/GLM-5.3-Flash`.
- `GLM-5.3-Flash-BF16` (~772 GiB): only when you explicitly need the BF16 variant.

On non-950 hardware native FP8 is dequantized to BF16 at load, so plan for roughly twice the weight memory. An Atlas 800 A2 (64GB × 8) single node does not have enough HBM for that path.

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`.

### 3.2 Verify Multi-node Communication (Optional)

If you want to deploy the model in a multi-node environment, verify the communication environment according to [verify multi-node communication environment](../../installation.md#verify-multi-node-communication).

## 4 Installation

### 4.1 Docker Image Installation

Select an image based on your machine type and start the docker image on your node, refer to [using docker](../../installation.md#set-up-using-docker).

=== "Ascend950DT/PR series"

    Start the docker image on each node.

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
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

=== "A3 series"

    Start the docker image on each node.

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a3
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

    Native FP8 is resolved to BF16 on A2, so a single 8-card A2 node is not enough. Use at least two Atlas 800 A2 nodes, or prefer Ascend 950PR.

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
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

After entering the container, verify that vLLM and vLLM-Ascend can be imported:

```shell
python -c "import vllm, vllm_ascend; print('vllm and vllm_ascend are ready')"
```

### 4.2 Source Code Installation

You can also build and install `vllm-ascend` from source. Refer to [set up using Python](../../installation.md#set-up-using-python).

**vLLM model class is not on vLLM main yet.** `Glm5NextForConditionalGeneration` lives in [vLLM#53906](https://github.com/vllm-project/vllm/pull/53906) (`ZJY0516/vllm` branch `glm-release`). On the 950PR node, overlay that tree onto the vLLM that ships with the Ascend image, then install this `vllm-ascend` branch:

```bash
# 1) Overlay GLM-5.3-Flash model support into the image's vLLM
cd /vllm-workspace/vllm
git remote add glm53 https://github.com/ZJY0516/vllm.git || true
git fetch glm53 glm-release
git checkout glm53/glm-release -- \
    vllm/models/glm5next \
    vllm/transformers_utils/configs/glm5_next.py
# If `vllm serve` still reports an unknown architecture, check out the
# full glm-release branch (or cherry-pick the registry / FLA kda commits)
# instead of copying only the model package.

# 2) Install the vllm-ascend NPU path from this GLM-5.3-Flash branch
cd /vllm-workspace/vllm-ascend
pip install -e .

python -c "import vllm, vllm_ascend; print('vllm and vllm_ascend are ready')"
```

## 5 Online Service Deployment

### 5.1 Single-Node Online Deployment (Ascend 950PR, 8 cards)

Replace `MODEL_PATH` with the local checkpoint or `zai-org/GLM-5.3-Flash`. Do **not** add `--quantization ascend`: the checkpoint is already `quant_method: fp8`.

```bash
#!/bin/sh
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE=512
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export MODEL_PATH=/root/.cache/modelscope/hub/models/ZhipuAI/GLM-5.3-Flash

vllm serve $MODEL_PATH \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 8 \
    --trust-remote-code \
    --served-model-name glm-5.3-flash \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --enable-auto-tool-choice \
    --speculative-config '{"method":"mtp","num_speculative_tokens":5,"enforce_eager":true}' \
    --max-model-len 131072 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 8192 \
    --block-size 512 \
    --gpu-memory-utilization 0.90
```

Bring-up order on 950PR: first add `--load-format dummy` to prove the architecture / KDA / SFA path, then drop it and load the real FP8 weights. Dummy success is not a real-weight sign-off.

Key parameter descriptions:

- `--tensor-parallel-size 8`: splits the 321B MoE across the 8 NPUs on one 950PR node.
- Do not pass `--quantization ascend`. Native FP8 is detected from `config.json` (`quant_method: fp8` + `weight_block_size: [128, 128]`). On 950 the tiles are re-grouped to MXFP8 at load.
- `--block-size 512`: required by the kpool indexer (`block_size` must be a multiple of `index_kpool * 32`; 16 × 32 = 512). The default 64 silently under-sizes the pool pages.
- `--max-model-len 131072`: practical single-node baseline. The config theoretical max is 1,048,576 tokens; raise this only after KV-cache headroom is confirmed.
- `--max-num-seqs 16`: default capacity baseline. Increase to 32/64 if HBM allows.
- `--speculative-config`: uses the checkpoint's single MTP layer. `enforce_eager: true` is required because GLM series models do not yet support graph-mode speculative decoding.
- `--tool-call-parser glm47` / `--reasoning-parser glm45` / `--enable-auto-tool-choice`: same GLM tool/reasoning parsers as GLM-5.x.
- `--enable-expert-parallel`: optional MoE path. Try it after the TP=8 serve is stable.

Eager isolation (only if graph capture fails):

```bash
# add --enforce-eager to the serve command above
```

## 6 Functional Verification

After the service is started, send `GET /v1/models` first, then a chat request. Use the `--served-model-name` you configured (`glm-5.3-flash`).

**Readiness:**

```bash
curl -sf http://127.0.0.1:8000/v1/models
```

**Chat Completions API:**

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "glm-5.3-flash",
        "messages": [
            {"role": "user", "content": "Summarize sparse attention in one sentence."}
        ],
        "max_tokens": 256,
        "temperature": 1.0
    }'
```

Expected Result: HTTP 200 with a non-empty `choices[0].message.content`. `Application startup complete` alone is not success.

**Vision (optional):** send one text+image `chat/completions` request. The architecture is `Glm5NextForConditionalGeneration` and the checkpoint includes a 24-layer ViT.

## 7 Accuracy Evaluation

Evaluate with AISBench after the 950PR serve is up. Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md).

Numbers below are placeholders until the real-weight gate on Ascend 950PR is recorded.

| dataset | model | metric | mode | vllm-api-general-chat |
| ----- | ----- | ----- | ----- | ----- |
| GSM8K | GLM-5.3-Flash | exact_match | gen | TBD |

## 8 Performance

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/benchmarking/) and the upstream recipe:

```bash
vllm bench serve \
    --backend openai-chat \
    --model $MODEL_PATH \
    --served-model-name glm-5.3-flash \
    --dataset-name random \
    --random-input-len 8192 \
    --random-output-len 1024 \
    --max-concurrency 16 \
    --num-prompts 64
```

Fill in 950PR tok/s, TTFT, and TPOT after the first bench run.
