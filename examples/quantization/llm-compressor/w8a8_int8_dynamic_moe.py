import torch
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

recipe = QuantizationModifier(
    targets="Linear",
    scheme="INT8",
    ignore=["lm_head", "re:.*mlp.gate$"],
)

oneshot(
    model=model,
    recipe=recipe,
    trust_remote_code_model=True,
)

# Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-INT8_W8A8"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
