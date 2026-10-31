from compressed_tensors.quantization.quant_scheme import (
    MXFP4,
    MXFP8,
    QuantizationScheme,
)
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4PreTrainedModel,
)


def main():
    model_id = "RedHatAI/DeepSeek-V4-Flash-BF16"

    # DeepSeek-V4 expects these modules to remain in the model dtype during
    # loading instead of being forced to FP32.
    DeepseekV4PreTrainedModel._keep_in_fp32_modules_strict = set()

    # Load the model with llm-compressor offloading support.
    with load_context():
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto_offload",
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Mixed MXFP quantization:
    #
    #   Attention       -> MXFP8 weights + dynamic MXFP8 activations
    #   Shared experts  -> MXFP8 weights + dynamic MXFP8 activations
    #   Routed experts  -> MXFP4 weights + dynamic MXFP4 activations
    #
    # Shared experts intentionally remain in MXFP8. Only routed experts
    # are quantized to MXFP4.
    recipe = QuantizationModifier(
        config_groups={
            "attention_shared_experts": QuantizationScheme(
                targets=[
                    r"re:.*attn\.(q_a_proj|q_b_proj|kv_proj|o_a_proj|o_b_proj)$",
                    r"re:.*attn\.compressor\.indexer\.q_b_proj$",
                    r"re:.*mlp\.shared_experts\.(gate|up|down)_proj$",
                ],
                **MXFP8,
            ),
            "routed_experts": QuantizationScheme(
                targets=[
                    r"re:.*mlp\.experts\..*(gate|up|down)_proj$",
                ],
                **MXFP4,
            ),
        },
        ignore=[],
    )

    # MXFP4/MXFP8 quantization is data-free, so no calibration dataset is
    # required.
    oneshot(
        model=model,
        recipe=recipe,
        pipeline="datafree",
    )

    # Save the model in compressed-tensors format.
    save_dir = model_id.rstrip("/").split("/")[-1] + "-Mixed-MXFP4-MXFP8"
    model.save_pretrained(
        save_dir,
        save_compressed=True,
        max_shard_size="5GB",
    )
    tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":
    main()
