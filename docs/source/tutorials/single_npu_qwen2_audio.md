# Single NPU (Qwen2-Audio-7B)

## Run vllm-ascend on Single NPU

### Offline Inference on Single NPU

Run docker container:

```{code-block} bash
   :substitutions:
# Update the vllm-ascend image
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
docker run --rm \
--name vllm-ascend \
--shm-size=1g \
--device /dev/davinci0 \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /root/.cache:/root/.cache \
-p 8000:8000 \
-it $IMAGE bash
```

Set up environment variables:

```bash
# Load model from ModelScope to speed up download
export VLLM_USE_MODELSCOPE=True

# Set `max_split_size_mb` to reduce memory fragmentation and avoid out of memory
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256
```

:::{note}
`max_split_size_mb` prevents the native allocator from splitting blocks larger than this size (in MB). This can reduce fragmentation and may allow some borderline workloads to complete without running out of memory. You can find more details [<u>here</u>](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/envref/envref_07_0061.html).
:::

Install packages required for audio processing:

```bash
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install librosa soundfile
```

Run the following script to execute offline inference on a single NPU:

```python
from vllm import LLM, SamplingParams
from vllm.assets.audio import AudioAsset
from vllm.utils import FlexibleArgumentParser

# If network issues prevent AudioAsset from fetching remote audio files, retry or check your network.
audio_assets = [AudioAsset("mary_had_lamb"), AudioAsset("winning_call")]
question_per_audio_count = {
    1: "What is recited in the audio?",
    2: "What sport and what nursery rhyme are referenced?"
}


def prepare_inputs(audio_count: int):
    audio_in_prompt = "".join([
        f"Audio {idx+1}: <|audio_bos|><|AUDIO|><|audio_eos|>\n"
        for idx in range(audio_count)
    ])
    question = question_per_audio_count[audio_count]
    prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
              "<|im_start|>user\n"
              f"{audio_in_prompt}{question}<|im_end|>\n"
              "<|im_start|>assistant\n")

    mm_data = {
        "audio":
        [asset.audio_and_sample_rate for asset in audio_assets[:audio_count]]
    }

    # Merge text prompt and audio data into inputs
    inputs = {"prompt": prompt, "multi_modal_data": mm_data}
    return inputs


def main(audio_count: int):
    # NOTE: The default `max_num_seqs` and `max_model_len` may result in OOM on
    # lower-end GPUs.
    # Unless specified, these settings have been tested to work on a single L4.
    # `limit_mm_per_prompt`: the max num items for each modality per prompt.
    llm = LLM(model="Qwen/Qwen2-Audio-7B-Instruct",
              max_model_len=4096,
              max_num_seqs=5,
              limit_mm_per_prompt={"audio": audio_count})

    inputs = prepare_inputs(audio_count)

    sampling_params = SamplingParams(temperature=0.2,
                                     max_tokens=64,
                                     stop_token_ids=None)

    outputs = llm.generate(inputs, sampling_params=sampling_params)

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)


if __name__ == "__main__":
    audio_count = 2
    main(audio_count)
```

If you run this script successfully, you can see the info shown below:

```bash
The sport referenced is baseball, and the nursery rhyme is 'Mary Had a Little Lamb'.
```

### Online Serving on Single NPU

Currently, the `chat_template` for `Qwen2-Audio` has some issues which caused audio placeholder failed to be inserted, find more details [<u>here</u>](https://github.com/vllm-project/vllm/issues/19977).

Nevertheless, we could use a custom template for online serving, which is shown below:

```jinja
{% set audio_count = namespace(value=0) %}
{% for message in messages %}
    {% if loop.first and message['role'] != 'system' %}
        <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
    {% endif %}
    <|im_start|>{{ message['role'] }}\n
    {% if message['content'] is string %}
        {{ message['content'] }}<|im_end|>\n
    {% else %}
        {% for content in message['content'] %}
            {% if 'audio' in content or 'audio_url' in content or message['type'] == 'audio' or content['type'] == 'audio' %}
                {% set audio_count.value = audio_count.value + 1 %}
                Audio {{ audio_count.value }}: <|audio_bos|><|AUDIO|><|audio_eos|>\n
            {% elif 'text' in content %}
                {{ content['text'] }}
            {% endif %}
        {% endfor %}
        <|im_end|>\n
    {% endif %}
{% endfor %}
{% if add_generation_prompt %}
    <|im_start|>assistant\n
{% endif %}
```

:::{note}
You can find this template at `vllm-ascend/examples/chat_templates/template_qwen2_audio.jinja`.
:::

Run docker container to start the vLLM server on a single NPU:

```{code-block} bash
# Update the vllm-ascend image
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
docker run --rm \
--name vllm-ascend \
--shm-size=1g \
--device /dev/davinci0 \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /root/.cache:/root/.cache \
-p 8000:8000 \
-e VLLM_USE_MODELSCOPE=True \
-e PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256 \
-it $IMAGE \
vllm serve Qwen/Qwen2-Audio-7B-Instruct \
--max_model_len 16384 \
--max-num-batched-tokens 16384 \
--limit-mm-per-prompt '{"audio":2}' \
--chat-template /path/to/your/vllm-ascend/examples/chat_templates/template_qwen2_audio.jinja
```

:::{note}
Replace `/path/to/your/vllm-ascend` with your own path.
:::

If your service start successfully, you can see the info shown below:

```bash
INFO:     Started server process [2736]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

Once your server is started, you can query the model with input prompts:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/root/.cache/modelscope/models/Qwen/Qwen2-Audio-7B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "audio_url", "audio_url": {"url": "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/winning_call.ogg"}},
                {"type": "text", "text": "What is in this audio? How does it sound?"}
            ]}
        ],
        "max_tokens": 100
    }'
```

If you query the server successfully, you can see the info shown below (client):

```bash
{"id":"chatcmpl-31f5f698f6734a4297f6492a830edb3f","object":"chat.completion","created":1761097383,"model":"/root/.cache/modelscope/models/Qwen/Qwen2-Audio-7B-Instruct","choices":[{"index":0,"message":{"role":"assistant","content":"The audio contains a background of a crowd cheering, a ball bouncing, and an object being hit. A man speaks in English saying 'and the o one pitch on the way to edgar martinez swung on and lined out.' The speech has a happy mood.","refusal":null,"annotations":null,"audio":null,"function_call":null,"tool_calls":[],"reasoning_content":null},"logprobs":null,"finish_reason":"stop","stop_reason":null,"token_ids":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":689,"total_tokens":743,"completion_tokens":54,"prompt_tokens_details":null},"prompt_logprobs":null,"prompt_token_ids":null,"kv_transfer_params":null}
```
