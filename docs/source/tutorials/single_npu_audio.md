# Single NPU (Qwen2-Audio 7B)

## Run vllm-ascend on Single NPU

### Offline Inference on Single NPU

Run docker container:

```{code-block} bash
   :substitutions:
# Update the vllm-ascend image
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
docker run --rm \
--name vllm-ascend \
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

Setup environment variables:

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
              limit_mm_per_prompt={"audio": audio_count},
              enforce_eager=True)

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

Currently, vllm's OpenAI-compatible server doesn't support audio inputs, find more details [<u>here</u>](https://github.com/vllm-project/vllm/issues/19977).
