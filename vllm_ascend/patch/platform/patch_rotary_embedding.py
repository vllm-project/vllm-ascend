_original_init = None


def _patched_init(self, enforce_enable=False, is_neox_style=True, enable_fp32_compute=False):
    _original_init(self, enforce_enable=enforce_enable, is_neox_style=is_neox_style, enable_fp32_compute=enable_fp32_compute)
    self.apply_rotary_emb_flash_attn = None


def patch_apply_rotary_emb():
    global _original_init
    from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

    _original_init = ApplyRotaryEmb.__init__
    ApplyRotaryEmb.__init__ = _patched_init


patch_apply_rotary_emb()
