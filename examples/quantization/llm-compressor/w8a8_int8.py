import os
import torch

from datasets import load_dataset
from transformers import AutoModelForCausalLM, Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, \
    AutoTokenizer, AutoProcessor, AutoConfig, AutoImageProcessor

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme, QuantizationType, QuantizationStrategy

W8A8_W_cha_A_ten_static_symmetric = {
    "group_0": QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=8,
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.CHANNEL,
            symmetric=True,
            dynamic=False
        ),
        input_activations=QuantizationArgs(
            num_bits=8,
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.TENSOR,
            symmetric=True,
            dynamic=False
        ),
    ),
}

# supported modifiers
MODIFIER_DICT = {
    "PTQ": QuantizationModifier,
    "AWQ": AWQModifier,
    "GPTQ": GPTQModifier,
}

# supported schemes
SCHEMES_DICT = {
    "W8A8_W_cha_A_ten_static_symmetric": W8A8_W_cha_A_ten_static_symmetric,
}

MODEL_DICT = {
    "qwen3": AutoModelForCausalLM,
}

TOKENIZER_DICT = {
    "qwen3": AutoTokenizer,
}


def load_environment_variables():
    env_vars = {
        'model_path': "Qwen/Qwen3-32B",
        'export_path': "/llm-compressor/export/GPTQ/W8A8_W_cha_A_ten_static_symmetric",
        'modifier': "GPTQ",
        'schemes': "W8A8_W_cha_A_ten_static_symmetric",
        'calib_prompt_path': "HuggingFaceH4/ultrachat_200k"
    }

    # verify export model path
    if env_vars['export_path'] is None:
        env_vars['export_path'] = env_vars['model_path'].rstrip("/") + "-" + env_vars['modifier']
        if env_vars['schemes'] is not None:
            env_vars['export_path'] += "-" + env_vars['schemes']
    os.makedirs(env_vars['export_path'], exist_ok=True)

    return env_vars


def load_calibration_text_dataset(calib_prompt_path, tokenizer):
    # Load dataset
    for f in os.listdir(calib_prompt_path):
        print(f)
    if any(f.lower().endswith('.jsonl') for f in os.listdir(calib_prompt_path)):
        ds = load_dataset('json', data_dir=calib_prompt_path, split='validation')
    elif any(f.lower().endswith('.parquet') for f in os.listdir(calib_prompt_path)):
        ds = load_dataset("parquet", data_dir=calib_prompt_path, split="train[:512]")
    else:
        raise ValueError("Unsupported calibration file format: {}".format(
            calib_prompt_path.split('.')[-1]))

    # Preprocess dataset
    def preprocess(example):
        if tokenizer.chat_template is not None:
            return {"text": tokenizer.apply_chat_template(
                example["messages"], tokenize=False)}
        else:
            return {"text": example["messages"]}

    # Tokenize inputs
    def tokenize(sample):
        return tokenizer(
            sample["text"],
            add_special_tokens=False,
        )

    ds = ds.map(preprocess)
    ds = ds.map(tokenize, remove_columns=ds.column_names)
    return ds


# Define a oneshot data collator for multimodal inputs.
def data_collator(batch):
    assert len(batch) == 1
    return {
        key: torch.tensor(value, dtype=torch.bfloat16 if key == "pixel_values" else torch.long)
        for key, value in batch[0].items()
    }


def quantize_model(model, env_vars, dataset_dict=None):
    # since the MoE gate layers are sensitive to quantization, we add them to the ignore
    # list so they remain at full precision
    ignore = ["lm_head", "re:.*mlp.down_proj"]

    # define a llmcompressor recipe
    recipe = [
        MODIFIER_DICT[env_vars['modifier']](
            config_groups=SCHEMES_DICT[env_vars['schemes']],
            ignore=ignore,
        ),
    ]

    # quantize the model
    oneshot(
        model=model,
        dataset=dataset_dict,
        recipe=recipe,
        trust_remote_code_model=True,
    )


def save_quantized_model(model, tokenizer, save_path, save_compressed=False):
    model.save_pretrained(save_path, save_compressed=save_compressed)
    tokenizer.save_pretrained(save_path)


if __name__ == '__main__':
    # get environment variables
    env_vars = load_environment_variables()

    # support model type list
    config = AutoConfig.from_pretrained(env_vars['model_path'], trust_remote_code=True)
    model_type = config.model_type

    model = MODEL_DICT[model_type].from_pretrained(
        env_vars['model_path'], torch_dtype="auto", trust_remote_code=True
    )
    tokenizer = TOKENIZER_DICT[model_type].from_pretrained(env_vars['model_path'], trust_remote_code=True)

    ds = load_calibration_text_dataset(env_vars["calib_prompt_path"], tokenizer)

    # Quantize the model
    quantize_model(model, env_vars, ds)

    # save the quantized model
    save_quantized_model(model, tokenizer, env_vars['export_path'], True)