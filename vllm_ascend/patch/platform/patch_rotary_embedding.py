def _patched_init(self, enforce_enable=False, is_neox_style=True, enable_fp32_compute=False):
    from vllm.model_executor.custom_op import CustomOp

    CustomOp.__init__(self, enforce_enable=enforce_enable)
    self.is_neox_style = is_neox_style
    self.enable_fp32_compute = enable_fp32_compute
    self.apply_rotary_emb_flash_attn = None


def patch_apply_rotary_emb():
    from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

    ApplyRotaryEmb.__init__ = _patched_init


patch_apply_rotary_emb()
