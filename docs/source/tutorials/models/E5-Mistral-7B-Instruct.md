# E5-Mistral-7B-Instruct

## 1 Introduction

[E5-Mistral-7B-Instruct](https://huggingface.co/intfloat/e5-mistral-7b-instruct) is a 7B-parameter text embedding model from the intfloat (unilm/e5) family. It is initialized from Mistral-7B-v0.1 and fine-tuned on a mixture of multilingual datasets with instruction tuning: the architecture is a plain `MistralModel` (no LM head) with a 4096-dimensional hidden size, 32 layers, and a sliding window of 4096. This guide describes how to run the model with vLLM Ascend.

The model uses **last-token pooling with L2 normalization** and accepts inputs of **at most 4096 tokens** (the official recommendation; the config allows up to 32768). Queries must carry a one-sentence instruction prefix, while passages stay bare text. See the [model card](https://huggingface.co/intfloat/e5-mistral-7b-instruct) for details.

## 2 Supported Features

Refer to [Supported Models List](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

## 3 Prerequisites

### 3.1 Model Weight

- `intfloat/e5-mistral-7b-instruct` [Download model weight](https://www.modelscope.cn/models/intfloat/e5-mistral-7b-instruct) (also available on [Hugging Face](https://huggingface.co/intfloat/e5-mistral-7b-instruct), ~15GB, MIT license)

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`

### 3.2 Key Model Facts

- Pooling: **last-token pooling + L2 normalization**. Request last-token pooling via `--pooler-config '{"pooling_type":"LAST"}'`; on vLLM 0.26+ the embedding pooler head applies L2 normalization by default (the `normalize` field was removed from `PoolerConfig`), so the returned vectors are already normalized.
- Tokenizer: sentencepiece `LlamaTokenizer` (Mistral-7B-v0.1 family). **Do not** pass `--tokenizer-mode mistral`; that flag only applies to v3+ tekken tokenizers.
- Input template (asymmetric): the query side needs an instruction prefix, the passage side stays bare.

## 4 Installation

### 4.1 Docker Image Installation

You can use our official docker image to run `E5-Mistral-7B-Instruct` model directly.

Select an image based on your machine type and start the docker image on your node, refer to [using docker](../../getting_started/installation.md#installation-prebuilt-image).

=== "A2 series"

    ```shell
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
    docker run --rm \
        --name vllm-ascend \
        --shm-size=1g \
        --net=host \
        --privileged=true \
        --device /dev/davinci0 \
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

=== "A3 series"

    ```shell
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a3
    docker run --rm \
        --name vllm-ascend \
        --shm-size=1g \
        --net=host \
        --privileged=true \
        --device /dev/davinci0 \
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

### 4.2 Source Code Installation

If you don't want to use the docker image as above, you can also build all from source:

- Install `vllm-ascend` from source, refer to [installation](../../getting_started/installation.md).

## 5 Online Service Deployment

=== "A2 series / A3 series"

    ```shell
    #!/bin/sh
    vllm serve intfloat/e5-mistral-7b-instruct  \
      --served-model-name e5-mistral-7b-instruct  \
      --runner pooling \
      --dtype bfloat16 \
      --max-model-len 4096 \
      --enforce-eager \
      --pooler-config '{"pooling_type":"LAST"}' \
      --port 8000
    ```

    The command is identical on A2 and A3 series products; only the docker image
    differs (the A3 image tag carries an `-a3` suffix, see 4.1).

Key parameter descriptions:

- `--runner pooling` selects the pooling runner; the embedding task of this MistralModel variant needs it. The task itself can also be spelled `--convert embed`, but `auto` (the default) already resolves to `embed` for this model, so passing it is optional.
- `--pooler-config '{"pooling_type":"LAST"}'` forces last-token pooling, matching the model card recipe; the embedding pooler head applies L2 normalization by default on vLLM 0.26+.
- `--dtype bfloat16` matches the model card usage; `--max-model-len 4096` is the official recommended upper limit (the config allows 32768).
- `--enforce-eager` is recommended for the first bring-up; remove it after the graph path is verified.
- `--tokenizer-mode` is intentionally omitted: the sentencepiece LlamaTokenizer is picked up automatically.

Common Issues Tip: If you encounter issues, please refer to the [Public FAQs](../../faqs.md) for troubleshooting.

## 6 Functional Verification

Once your server is started, you can verify it with the following command. Note the asymmetric input template: **the query side carries the `Instruct: ... \nQuery: ...` prefix, the passage side stays bare**.

Query side (with instruction prefix):

```bash
curl -X POST http://localhost:8000/v1/embeddings -H "Content-Type: application/json" -d '{
  "model": "e5-mistral-7b-instruct",
  "input": [
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: what is the capital of France"
    ]
}'
```

Passage side (bare text):

```bash
curl -X POST http://localhost:8000/v1/embeddings -H "Content-Type: application/json" -d '{
  "model": "e5-mistral-7b-instruct",
  "input": [
        "Paris is the capital and most populous city of France."
    ]
}'
```

Expected Result:

The service returns HTTP 200 OK with a JSON response containing the `embedding` field. The vector dimension is 4096 and its L2 norm is 1 (the pooler normalizes). Example output:

```json
{
  "id": "embd-8136155c01e8411d",
  "object": "list",
  "created": 1784538286,
  "model": "e5-mistral-7b-instruct",
  "data": [
    {
      "index": 0,
      "object": "embedding",
      "embedding": [
      -0.04725276678800583,-0.021066857501864433
      ]
    }
  ],
  "usage": {
    "prompt_tokens": 39,
    "total_tokens": 39,
    "completion_tokens": 0,
    "prompt_tokens_details": null
  }
}
```

The `embedding` array is abbreviated above for readability: a real response
returns 4096 dims in total (the two values shown are its first entries).

For more usage examples, please reference the [examples](https://github.com/vllm-project/vllm/tree/main/examples/pooling/embed).

## 7 Accuracy Evaluation

Here are two accuracy evaluation methods.

### 7.1 Using MTEB

1. Refer to [MTEB](https://docs.mteb.org/) for details.

2. Run the following code to execute the accuracy evaluation.

    ```python

    import os
    import mteb

    from mteb.models.vllm_wrapper import VllmEncoderWrapper

    # ModelScope cache layout: ~/.cache/modelscope/hub/models/<namespace>/<model-name>
    # Download the weight first, for example:
    #   modelscope download --model intfloat/e5-mistral-7b-instruct --local_dir "$MODEL_PATH"
    MODEL_PATH = os.path.expanduser("~/.cache/modelscope/hub/models/intfloat/e5-mistral-7b-instruct")

    if __name__ == "__main__":

        data_path = "/home/data/mteb_data"
        os.environ["HF_DATASETS_CACHE"] = data_path
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

        model = VllmEncoderWrapper(MODEL_PATH,
                                    dtype="bfloat16",
                                    max_model_len=4096,
                                    pooler_config={"pooling_type": "LAST"},
                                   )

        cache = mteb.ResultCache("/home/data/mteb_data")
        tasks = mteb.get_tasks(tasks=["LeCaRDv2"])
        results = mteb.evaluate(model, tasks=tasks, cache=cache, encode_kwargs={"batch_size": 2}, overwrite_strategy="always")
        df = results.to_dataframe()
        print(df)
    ```

    Note this snippet does **not** add the asymmetric instruction prefix: the
    wrapper above encodes the raw texts, so the printed scores only show that the
    model loads and encodes, and are meant as a smoke-run reference rather than an
    accuracy baseline. To follow the model card recipe, prepend the
    `Instruct: {task instruction}` line, a newline and `Query:` (followed by a
    space) to every query, while passages stay bare, e.g. through mteb's
    `prompt_dict` / `use_instructions` options.

3. After execution, you can get the result.

### 7.2 Using the offline correctness test

The repository ships an offline correctness test that compares vLLM Ascend against a Hugging Face reference (`AutoModel` + official last-token pooling + L2 normalization) on a self-constructed bilingual corpus:

```bash
pytest -sv tests/e2e/pull_request/one_card/pooling/test_e5_mistral_7b_instruct.py
```

## 8 Performance Evaluation

### 8.1 Using vLLM Benchmark

Run performance of `intfloat/e5-mistral-7b-instruct` as an example. Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/benchmarking/cli/) for more details.

Take the `serve` as an example. Run the code as follows.

```bash
vllm bench serve --model intfloat/e5-mistral-7b-instruct --backend openai-embeddings --port 8000 --dataset-name random --endpoint /v1/embeddings --random-input 200 --save-result --result-dir ./
```

After about several minutes, you can get the performance evaluation result.

## 9 FAQ

For common environment, installation, and general parameter issues, please refer to the [Public FAQs](../../faqs.md).

**1. Do I need to add the instruction prefix to passages?**

No. The prefix is only added on the query side. The model card states: "On the other hand, there is no need to add instructions to the document side." Adding it to passages deviates from the training template and hurts accuracy.

**2. What happens if I forget `--pooler-config`?**

Probably nothing, for this model: the repo ships a sentence-transformers `1_Pooling/config.json` with `pooling_mode_lasttoken: true`, so vLLM 0.26 already defaults to last-token pooling here (measured). Explicitly passing `--pooler-config '{"pooling_type":"LAST"}'` is still the recommended practice as a defense against upstream default changes, or self-hosted weights that lack the pooling config (L2 normalization is applied by the embedding pooler head by default on vLLM 0.26+).

**3. Can I use `--tokenizer-mode mistral`?**

No. This model uses the sentencepiece `LlamaTokenizer`; `mistral` tokenizer mode only applies to v3+ tekken tokenizers and breaks loading.

**4. How long can my input be?**

The official recommendation is at most 4096 tokens per input (the config allows up to 32768). With `--max-model-len 4096`, longer inputs are rejected with an error instead of being silently truncated.

**5. Is this model recommended for Chinese?**

The model has some multilingual capability, but the model card recommends English for best accuracy and points to `multilingual-e5-large` for multilingual use cases.
