# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from vllm.config import CUDAGraphMode, KVTransferConfig

from vllm_ascend.models.hy_v4_flash import model


@pytest.mark.parametrize("tp,dp", [(4, 1), (1, 4)])
@pytest.mark.parametrize(
    "connector,layerwise", [("MooncakeConnectorV2", False), ("MooncakeConnectorV2", True), ("Other", False)]
)
def test_flash_pd_accepts_only_whole_request_pull(monkeypatch, tp, dp, connector, layerwise):
    text = SimpleNamespace(
        validate_flash_config=lambda: None,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        engram_layer_ids=[],
        enable_ihc=True,
        learnable_sink=True,
        gated_mla=True,
        gating_type="elementwise",
        swa_qk_rope_head_dim=64,
        num_attention_heads=32,
        swa_num_attention_heads=64,
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=text, dtype=torch.bfloat16, quantization=None),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            enable_eplb=False,
            enable_dbo=False,
            tensor_parallel_size=tp,
            data_parallel_size=dp,
        ),
        use_v2_model_runner=True,
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE),
        cache_config=SimpleNamespace(cache_dtype="fp8"),
        speculative_config=None,
        kv_transfer_config=KVTransferConfig(
            kv_connector=connector,
            kv_role="kv_producer" if dp == 1 else "kv_consumer",
            kv_connector_extra_config={"use_layerwise": layerwise},
        ),
        lora_config=None,
    )
    monkeypatch.setattr(model, "get_ascend_config", lambda: SimpleNamespace(enable_fused_mc2=0))
    if connector == "MooncakeConnectorV2" and not layerwise:
        model.validate_runtime(config)
    else:
        with pytest.raises(ValueError, match="pull mode"):
            model.validate_runtime(config)
