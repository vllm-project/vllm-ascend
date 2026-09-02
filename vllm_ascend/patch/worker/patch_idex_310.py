import vllm
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model

from vllm_ascend._310p.ops.fla.gdn_310 import AscendGatedDeltaNetAttention310
from vllm_ascend._310p.ops.fla.idex import (
    prepare_chunk_indices_310,
    prepare_chunk_offsets_310,
)
from vllm_ascend._310p.spec_decode.dflash_model_310 import (
    patch_dflash_read_mask_embedding_310,
    precompute_and_store_context_kv_310,
)
from vllm_ascend._310p.spec_decode.dflash_proposer_310 import (
    AscendDflashProposer310,
    AscendDsparkProposer310,
    wrap_dummy_run_with_draft_flag,
)
from vllm_ascend._310p.spec_decode.llm_base_proposer_310 import AscendSpecDecodeBaseProposer310
from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer
from vllm_ascend.spec_decode.dspark_proposer import AscendDsparkProposer
from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer
from vllm_ascend.utils import is_rc_device

vllm.model_executor.layers.fla.ops.index.prepare_chunk_indices = prepare_chunk_indices_310

vllm.model_executor.layers.fla.ops.index.prepare_chunk_offsets = prepare_chunk_offsets_310


_original_spec_decode_base_proposer_init_310 = AscendSpecDecodeBaseProposer.__init__


def _init_spec_decode_base_proposer_310(self, *args, **kwargs):
    """Preserve the public constructor and initialize guarded 310P buffers."""
    _original_spec_decode_base_proposer_init_310(self, *args, **kwargs)
    AscendSpecDecodeBaseProposer310._initialize_hybrid_draft_slot_mapping_310(self)


AscendSpecDecodeBaseProposer.__init__ = (  # type: ignore[method-assign]
    _init_spec_decode_base_proposer_310
)
AscendSpecDecodeBaseProposer.load_model = (  # type: ignore[method-assign]
    AscendSpecDecodeBaseProposer310.load_model
)

# 310P: protect tail slot during MTP input_ids shift to avoid GatherV2 corruption
# caused by the NPU slice-assign writing one element past the intended range
# on the persistent drafter input_ids buffer.
AscendSpecDecodeBaseProposer.set_inputs_first_pass = (  # type: ignore[method-assign]
    AscendSpecDecodeBaseProposer310.set_inputs_first_pass
)
AscendSpecDecodeBaseProposer._run_merged_draft = (  # type: ignore[method-assign]
    AscendSpecDecodeBaseProposer310._run_merged_draft
)
AscendSpecDecodeBaseProposer._compute_draft_step_slot_mapping = (  # type: ignore[method-assign]
    AscendSpecDecodeBaseProposer310._compute_draft_step_slot_mapping
)
AscendSpecDecodeBaseProposer._prepare_full_decode_draft_rope = (  # type: ignore[method-assign]
    AscendSpecDecodeBaseProposer310._prepare_full_decode_draft_rope
)
AscendSpecDecodeBaseProposer._finish_full_decode_draft_rope = (  # type: ignore[method-assign]
    AscendSpecDecodeBaseProposer310._finish_full_decode_draft_rope
)

# 310P has no Triton, so dflash/dspark build their draft-model inputs via the
# AscendC npu_copy_and_expand_dflash_inputs custom op instead of the Triton
# kernel used on other platforms.
AscendDflashProposer.set_inputs_first_pass = (  # type: ignore[method-assign]
    AscendDflashProposer310.set_inputs_first_pass
)
AscendDflashProposer.prepare_next_token_ids_padded = (  # type: ignore[method-assign]
    AscendDflashProposer310.prepare_next_token_ids_padded
)
AscendDflashProposer.build_model_inputs_first_pass = (  # type: ignore[method-assign]
    AscendDflashProposer310.build_model_inputs_first_pass
)
AscendDflashProposer._build_first_pass_per_layer_attn_metadata = (  # type: ignore[method-assign]
    AscendDflashProposer310._build_first_pass_per_layer_attn_metadata
)
AscendDsparkProposer.set_inputs_first_pass = (  # type: ignore[method-assign]
    AscendDsparkProposer310.set_inputs_first_pass
)

# 310P: the profile branch of dummy_run calls the draft model directly (not via
# _run_merged_draft), so enable the drafting RoPE flag around dummy_run so the
# draft builds cos/sin from its own cache (required for VL main + text draft,
# where the global cos/sin slice is never populated). dspark has an independent
# dummy_run, so wrap both.
AscendDflashProposer.dummy_run = wrap_dummy_run_with_draft_flag(  # type: ignore[method-assign]
    AscendDflashProposer.dummy_run
)
AscendDsparkProposer.dummy_run = wrap_dummy_run_with_draft_flag(  # type: ignore[method-assign]
    AscendDsparkProposer.dummy_run
)

# 310P has no functional Triton: kernels cannot be compiled (bishengir-compile
# fails on Ascend310P3) and the 310P worker never calls
# init_device_properties_triton(), so get_vectorcore_num() asserts. Yet
# vllm.triton_utils.HAS_TRITON is True because the triton package imports, so
# every ``if HAS_TRITON:`` runtime gate takes the broken Triton path (e.g.
# rejection_sample() and prepare_inputs_padded() both crash on the uninitialized
# vector-core count). Force the flag off on the vllm_ascend modules that gate
# inference behavior on it so their PyTorch fallbacks are used consistently.
import vllm_ascend.sample.rejection_sampler as _rejection_sampler_mod  # noqa: E402
import vllm_ascend.sample.sampler as _sampler_mod  # noqa: E402
import vllm_ascend.spec_decode.llm_base_proposer as _llm_base_proposer_mod  # noqa: E402

for _mod in (_rejection_sampler_mod, _sampler_mod, _llm_base_proposer_mod):
    if getattr(_mod, "HAS_TRITON", False):
        _mod.HAS_TRITON = False

# 310P: patch_qwen3_dflash is excluded on 310P, so wire the dflash model
# precompute (context KV + RoPE) and mask-embedding fallback here.
DFlashQwen3Model.precompute_and_store_context_kv = precompute_and_store_context_kv_310  # type: ignore[method-assign]
patch_dflash_read_mask_embedding_310()

# Patch _warmup_prefill_kernels to no-op on 310P: triton.next_power_of_2 does
# not exist in the triton version used on 310P CI, and NPU does not use these
# CUDA warmup kernel anyway.
QwenGatedDeltaNetAttention._warmup_prefill_kernels = lambda self, qkv_or_qkvz, v_dim: None  # type: ignore[method-assign]
QwenGatedDeltaNetAttention._split_ba_for_tp = AscendGatedDeltaNetAttention._split_ba_for_tp
QwenGatedDeltaNetAttention.get_state_shape = AscendGatedDeltaNetAttention.get_state_shape
QwenGatedDeltaNetAttention._forward_core = AscendGatedDeltaNetAttention310._forward_core
QwenGatedDeltaNetAttention.get_state_dtype = AscendGatedDeltaNetAttention310.get_state_dtype

# 310P: make Qwen GDN use the 310P attention backend, including the
# MTP ACL graph padding replay fixes provided by gdn_attn_builder_310.py.
QwenGatedDeltaNetAttention.get_attn_backend = AscendGatedDeltaNetAttention310.get_attn_backend

# 310P: `patch_qwen3vl` is excluded on 310P because it also patches the Qwen3
# attention forward onto a Triton kernel (triton_split_qkv_rmsnorm_mrope), which
# 310P cannot compile.  The vision pos-embed interpolation, however, has a Triton
# kernel too (bishengir-compile fails on Ascend310P3), so wire only the native
# (non-Triton) pos-embed replacement here.  Guarded because older vLLM releases
# may not expose Qwen3-VL / pos_embed_interpolate_native.
try:
    from vllm.model_executor.models.qwen3_vl import Qwen3_VisionTransformer as _Qwen3VisionTransformer310

    from vllm_ascend._310p.ops.qwen3vl_310 import fast_pos_embed_interpolate_310

    _Qwen3VisionTransformer310.fast_pos_embed_interpolate = fast_pos_embed_interpolate_310  # type: ignore[method-assign]
except Exception:
    pass

if is_rc_device():
    from vllm.model_executor.models.qwen3_vl import Qwen3_VisionTransformer
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionBackend

    from vllm_ascend._310p.ops.gdn_attn_builder_310 import GDNAttentionMetadataBuilder310
    from vllm_ascend._310p.ops.qwen3vl_310 import rot_pos_emb_310

    # 310P RC: use blocking H2D in rot_pos_emb to avoid race with subsequent indexing.
    Qwen3_VisionTransformer.rot_pos_emb = rot_pos_emb_310  # type: ignore[method-assign]

    # Qwen3.5 on 310P RC uses upstream GDNAttentionBackend via MambaBase.get_attn_backend().
    GDNAttentionBackend.get_builder_cls = staticmethod(  # type: ignore[method-assign]
        lambda: GDNAttentionMetadataBuilder310
    )
