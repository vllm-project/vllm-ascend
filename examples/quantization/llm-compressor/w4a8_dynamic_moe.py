from llmcompressor import oneshot
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"

# Load model.
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

recipe = """
quant_stage:
    quant_modifiers:
        QuantizationModifier:
            ignore: ["lm_head", "re:.*mlp.gate$"]
            config_groups:
                group_0:
                    weights:
                        num_bits: 8
                        type: int
                        strategy: channel
                        dynamic: false
                        symmetric: true
                    input_activations:
                        num_bits: 8
                        type: int
                        strategy: token
                        dynamic: true
                        symmetric: true
                    targets: ["re:.*self_attn.k_proj.*", "re:.*self_attn.o_proj.*",
                        "re:.*self_attn.q_proj.*", "re:.*self_attn.v_proj.*"]
                group_1:
                    weights:
                        num_bits: 4
                        type: int
                        strategy: group
                        group_size: 128
                        dynamic: false
                        symmetric: true
                    input_activations:
                        num_bits: 8
                        type: int
                        strategy: token
                        dynamic: true
                        symmetric: true
                    targets: ["re:.*down_proj.*", "re:.*gate_proj.*", "re:.*up_proj.*"]
"""

# Apply quantization.
oneshot(
    model=model,
    recipe=recipe,
    trust_remote_code_model=True,
)

# Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-W4A8"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
