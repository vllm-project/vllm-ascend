# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.models.deepseek_v4 import dspark as dspark_module
from vllm_ascend.models.deepseek_v4.dspark import DeepseekV4DSparkModel


def test_draft_constructor_shares_and_reserves_oproj_buffers(monkeypatch):
    config = SimpleNamespace(
        hc_mult=4,
        hidden_size=8,
        vocab_size=32,
        draft_vocab_size=None,
        dspark_block_size=6,
        dspark_target_layer_ids=[37, 38, 39],
        dspark_markov_rank=4,
        num_hidden_layers=40,
        n_mtp_layers=3,
        head_dim=4,
        rms_norm_eps=1e-6,
        hc_eps=1e-5,
        quantization_config={"quant_method": "fp8"},
    )
    quant_config = object()
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=config),
        ),
        model_config=SimpleNamespace(dtype=torch.float16),
        quant_config=quant_config,
    )
    draft_impls = []

    def layer(*args, **kwargs):
        result = torch.nn.Module()
        result.use_sequence_parallel_moe = False
        impl = MagicMock(spec=dspark_module.AscendDSAImpl)
        draft_impls.append(impl)
        result.self_attn = SimpleNamespace(dsa_attn=SimpleNamespace(dsa_attn=SimpleNamespace(impl=impl)))
        return result

    monkeypatch.setattr(dspark_module, "DeepseekV4DecoderLayer", layer)
    monkeypatch.setattr(dspark_module, "VocabParallelEmbedding", lambda *a, **kw: torch.nn.Identity())
    monkeypatch.setattr(dspark_module, "MergedColumnParallelLinear", lambda *a, **kw: torch.nn.Identity())
    monkeypatch.setattr(dspark_module, "ColumnParallelLinear", lambda *a, **kw: torch.nn.Identity())
    monkeypatch.setattr(dspark_module, "RMSNorm", lambda *a, **kw: torch.nn.Identity())
    with (
        patch.object(dspark_module, "DSparkMarkovHead", autospec=True, return_value=torch.nn.Identity()),
        patch.object(dspark_module, "DSparkConfidenceHead", autospec=True, return_value=torch.nn.Identity()),
    ):
        model = DeepseekV4DSparkModel(vllm_config=vllm_config, prefix="model")

    assert len(draft_impls) == 3
    for impl in draft_impls:
        assert impl._oproj_buffer_pool is model._oproj_buffer_pool
        impl.reserve_oproj_buffers.assert_called_once_with(
            torch.device(dspark_module.current_platform.device_type), torch.float16
        )


def test_context_wkv_weights_are_duplicated_by_layer():
    weights = [
        ("mtp.0.attn.wkv.weight", torch.arange(4)),
        ("mtp.1.attn.wq_a.weight", torch.arange(3)),
        ("mtp.2.attn.wkv.scale", torch.tensor(0.5)),
        ("mtp.3.attn.wkv.weight", torch.arange(2)),
    ]

    duplicated = list(
        dspark_module._duplicate_context_wkv_weights(
            weights,
            num_layers=3,
        )
    )

    assert [name for name, _ in duplicated] == [
        "mtp.0.attn.wkv.weight",
        "context_wkv_proj.weight",
        "mtp.1.attn.wq_a.weight",
        "mtp.2.attn.wkv.scale",
        "context_wkv_proj.scale",
        "mtp.3.attn.wkv.weight",
    ]

    assert duplicated[1][1].shard_id == 0
    assert duplicated[4][1].shard_id == 2

    # detach() does not copy storage.
    assert duplicated[0][1].data_ptr() == duplicated[1][1].data_ptr()
    assert duplicated[3][1].data_ptr() == duplicated[4][1].data_ptr()


@pytest.mark.parametrize("push,sharing", [(False, False), (True, False), (True, True)])
def test_context_kv_uses_one_stacked_wkv_projection(monkeypatch, push, sharing):
    transfer = push and sharing
    projection_calls = []
    rope = object()

    stacked_output = torch.arange(
        24,
        dtype=torch.float32,
    ).view(2, 12)

    class StackedProjection:
        def __call__(self, context_states):
            projection_calls.append(context_states)
            return stacked_output

    attns = [
        SimpleNamespace(
            name=f"attn-{i}",
            dsa_attn=SimpleNamespace(swa_cache_layer=SimpleNamespace(prefix=f"draft-{i}", kv_cache=object())),
            rotary_emb=SimpleNamespace(layername=f"layer-{i}"),
        )
        for i in range(3)
    ]

    monkeypatch.setattr(
        dspark_module,
        "get_cos_and_sin_dsa",
        lambda positions, layer_names: rope,
    )

    layers = {
        "40": SimpleNamespace(self_attn=attns[0]),
        "41": SimpleNamespace(self_attn=attns[1]),
        "42": SimpleNamespace(self_attn=attns[2]),
    }

    projected = []
    stored = []

    model = SimpleNamespace(
        publish_prompt_context=False,
        context_wkv_proj=StackedProjection(),
        num_dspark_layers=3,
        config=SimpleNamespace(head_dim=4),
        layers=layers,
    )

    def project_shared_kv(kv, positions, attn, rope=None):
        projected.append((kv.clone(), positions, attn, rope))
        return kv.unsqueeze(1)

    def store_standard_swa_kv(shared_kv, slot_mapping, attn):
        events.append(("write", attn.name))
        stored.append(
            (
                shared_kv.clone(),
                slot_mapping,
                attn,
            )
        )

    model.publish_prompt_context = push
    events = []
    store = MagicMock()
    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.connector import MooncakePushConnector

    push_child = MagicMock(spec=MooncakePushConnector)
    push_child.supports_layerwise_buffer_reuse = sharing
    store.sub_connectors = [SimpleNamespace(supports_layerwise_buffer_reuse=True), push_child]
    store.has_connector_metadata.return_value = True
    store.wait_for_layer_load.side_effect = lambda name: events.append(("wait", name))
    store.save_kv_layer.side_effect = lambda name, *_: events.append(("save", name))
    monkeypatch.setattr(dspark_module, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(dspark_module, "get_kv_transfer_group", lambda: store)
    monkeypatch.setattr(dspark_module, "record_attention_compute_start", lambda: events.append(("gate", None)))
    model._project_shared_kv = project_shared_kv
    model._store_standard_swa_kv = store_standard_swa_kv

    context_states = torch.zeros(2, 5)
    positions = torch.tensor([7, 8])

    slot_mappings = [
        torch.tensor([0, 1]),
        None,
        torch.tensor([4, 5]),
    ]

    if transfer:
        with pytest.raises(ValueError, match="slot mapping for every context row"):
            DeepseekV4DSparkModel.precompute_and_store_context_kv(model, context_states, positions, slot_mappings)
        slot_mappings[1] = torch.empty(0, dtype=torch.long)
        with pytest.raises(ValueError, match="slot mapping for every context row"):
            DeepseekV4DSparkModel.precompute_and_store_context_kv(model, context_states, positions, slot_mappings)
        slot_mappings[1] = torch.tensor([2, 3])
        cache = attns[-1].dsa_attn.swa_cache_layer
        bound_cache, cache.kv_cache = cache.kv_cache, None
        with pytest.raises(RuntimeError, match="bound draft SWA caches"):
            DeepseekV4DSparkModel.precompute_and_store_context_kv(model, context_states, positions, slot_mappings)
        cache.kv_cache = bound_cache
        assert not events
        projection_calls.clear()
    DeepseekV4DSparkModel.precompute_and_store_context_kv(
        model,
        context_states,
        positions,
        slot_mappings,
    )

    assert len(projection_calls) == 1
    assert projection_calls[0] is context_states

    if transfer:
        assert events == [
            event
            for i in range(3)
            for event in (("wait", f"draft-{i}"), ("gate", None), ("write", f"attn-{i}"), ("save", f"draft-{i}"))
        ]
        assert len(stored) == 3
        return
    assert len(projected) == 2
    assert len(stored) == 2

    expected = stacked_output.view(2, 3, 4)

    torch.testing.assert_close(
        projected[0][0],
        expected[:, 0],
    )
    torch.testing.assert_close(
        projected[1][0],
        expected[:, 2],
    )
    assert projected[0][3] is rope
    assert projected[1][3] is rope

    assert stored[0][1] is slot_mappings[0]
    assert stored[1][1] is slot_mappings[2]


def test_context_kv_profiles_stacked_projection_without_slot_mappings():
    projection_calls = []

    class StackedProjection:
        def __call__(self, context_states):
            projection_calls.append(context_states)
            return torch.zeros(2, 12)

    model = SimpleNamespace(
        publish_prompt_context=False,
        context_wkv_proj=StackedProjection(),
        num_dspark_layers=3,
        config=SimpleNamespace(head_dim=4),
        layers={str(layer_idx): SimpleNamespace(self_attn=SimpleNamespace()) for layer_idx in range(3)},
    )

    def fail_project_shared_kv(*_):
        raise AssertionError("KV preparation must be skipped without slot mappings")

    def fail_store_standard_swa_kv(*_):
        raise AssertionError("KV cache writes must be skipped without slot mappings")

    model._project_shared_kv = fail_project_shared_kv
    model._store_standard_swa_kv = fail_store_standard_swa_kv

    context_states = torch.zeros(2, 5)
    DeepseekV4DSparkModel.precompute_and_store_context_kv(
        model,
        context_states,
        torch.tensor([7, 8]),
    )

    assert len(projection_calls) == 1
    assert projection_calls[0] is context_states
