import transformers
import logging

def patch_transformers_for_LlamaFlashAttention2():
    """
    Patch for DeepSeek-OCR-2 / DeepSeek-V2 and other models.
    Higher versions of transformers (>=4.43.0) removed `LlamaFlashAttention2`.
    We inject a dummy alias to bypass the ImportError during AutoConfig loading.
    """
    try:
        import transformers.models.llama.modeling_llama as modeling_llama
        if not hasattr(modeling_llama, "LlamaFlashAttention2"):
            modeling_llama.LlamaFlashAttention2 = modeling_llama.LlamaAttention
            logging.getLogger(__name__).debug("Successfully patched LlamaFlashAttention2")
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to patch transformers: {e}")

patch_transformers_for_LlamaFlashAttention2()