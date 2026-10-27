# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline-parallel contract of the GLM-5.3-Flash text model.

An mHC decoder layer defers its ``hc_post`` state (``post`` / ``comb``) to the
next layer. Across a PP boundary that state must travel with the intermediate
tensors: dropping it would make multi-stage execution differ from the
single-stage math. These tests pin the payload key set (which must match
``make_empty_intermediate_tensors``) and the hand-off between two stages.
"""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.sequence import IntermediateTensors

from vllm_ascend.models.glm5next import model as glm5next_model
from vllm_ascend.models.glm5next.model import Glm5NextModel

HIDDEN_SIZE = 8
N_STREAMS = 4


def _stub_model(*, mhc: bool) -> Glm5NextModel:
    # The PP plumbing only reads ``config`` plus the active layer slice, so a
    # stub avoids building the NPU-bound attention / MoE stack.
    model = Glm5NextModel.__new__(Glm5NextModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        hidden_size=HIDDEN_SIZE,
        mhc=mhc,
        hc_mult=N_STREAMS,
    )
    model.is_sequence_parallel = False
    model._sp_pp_enabled = False
    return model


def _pp_group(*, is_first_rank: bool, is_last_rank: bool):
    return lambda: SimpleNamespace(is_first_rank=is_first_rank, is_last_rank=is_last_rank, world_size=2)


class _DeferredStateLayer(nn.Module):
    """Emulates an mHC decoder layer.

    Records the deferred state it received and hands a fresh one to the next
    layer, mirroring ``Glm5NextDecoderLayer``'s 4-tuple contract.
    """

    def __init__(self) -> None:
        super().__init__()
        self.incoming: tuple[torch.Tensor | None, torch.Tensor | None] | None = None

    def forward(self, positions, hidden_states, residual, post, comb):
        del positions
        self.incoming = (post, comb)
        return (
            hidden_states + 1,
            residual,
            torch.full((hidden_states.shape[0], N_STREAMS, 1), 2.0),
            torch.full((hidden_states.shape[0], N_STREAMS, N_STREAMS), 3.0),
        )


@pytest.mark.parametrize("mhc", [True, False])
def test_make_empty_intermediate_tensors_matches_pp_payload(mhc):
    model = _stub_model(mhc=mhc)

    tensors = model.make_empty_intermediate_tensors(batch_size=5, dtype=torch.bfloat16, device=torch.device("cpu"))

    assert tensors["hidden_states"].shape == (5, HIDDEN_SIZE)
    assert tensors["hidden_states"].dtype == torch.bfloat16
    if mhc:
        # The per-token stream stays 2D; the hyper-connection streams live in
        # ``residual`` and the deferred state is FP32 (hc_pre kernel output).
        assert tensors["residual"].shape == (5, N_STREAMS, HIDDEN_SIZE)
        assert tensors["post"].shape == (5, N_STREAMS, 1)
        assert tensors["comb"].shape == (5, N_STREAMS, N_STREAMS)
        assert tensors["post"].dtype == torch.float32
        assert tensors["comb"].dtype == torch.float32
    else:
        assert set(tensors.tensors) == {"hidden_states", "residual"}
        assert tensors["residual"].shape == (5, HIDDEN_SIZE)


@pytest.mark.parametrize("mhc", [True, False])
def test_non_last_rank_ships_state_the_next_rank_loads(monkeypatch, mhc):
    model = _stub_model(mhc=mhc)
    model._active_layers = [_DeferredStateLayer()]
    monkeypatch.setattr(glm5next_model, "get_pp_group", _pp_group(is_first_rank=True, is_last_rank=False))
    inputs_embeds = torch.zeros(3, HIDDEN_SIZE, dtype=torch.bfloat16)

    out = model(input_ids=None, positions=torch.arange(3), intermediate_tensors=None, inputs_embeds=inputs_embeds)

    # Next rank indexes these keys directly, so the shipped set must match the
    # placeholder factory key-for-key.
    expected_keys = {"hidden_states", "residual"} | ({"post", "comb"} if mhc else set())
    assert set(out.tensors) == expected_keys
    torch.testing.assert_close(out["hidden_states"], inputs_embeds + 1)
    if mhc:
        assert out["post"].dtype == torch.float32
        assert out["comb"].dtype == torch.float32


def test_non_first_rank_continues_deferred_mhc_state(monkeypatch):
    model = _stub_model(mhc=True)
    layer = _DeferredStateLayer()
    model._active_layers = [layer]
    model.norm = nn.Identity()
    monkeypatch.setattr(glm5next_model, "get_pp_group", _pp_group(is_first_rank=False, is_last_rank=True))
    hidden_states = torch.zeros(3, HIDDEN_SIZE, dtype=torch.bfloat16)
    residual = torch.zeros(3, N_STREAMS, HIDDEN_SIZE, dtype=torch.bfloat16)
    post = torch.ones(3, N_STREAMS, 1)
    comb = torch.ones(3, N_STREAMS, N_STREAMS)
    payload = IntermediateTensors({"hidden_states": hidden_states, "residual": residual, "post": post, "comb": comb})

    out = model(input_ids=None, positions=torch.arange(3), intermediate_tensors=payload)

    # The receiving first layer must run hc_post_pre on the previous rank's
    # state instead of falling back to a standalone pre.
    assert layer.incoming is not None
    assert layer.incoming[0] is post
    assert layer.incoming[1] is comb
    torch.testing.assert_close(out, hidden_states + 1)


def test_non_first_rank_without_mhc_reads_plain_payload(monkeypatch):
    model = _stub_model(mhc=False)
    layer = _DeferredStateLayer()
    model._active_layers = [layer]
    model.norm = nn.Identity()
    monkeypatch.setattr(glm5next_model, "get_pp_group", _pp_group(is_first_rank=False, is_last_rank=True))
    hidden_states = torch.zeros(3, HIDDEN_SIZE, dtype=torch.bfloat16)
    payload = IntermediateTensors(
        {"hidden_states": hidden_states, "residual": torch.zeros(3, HIDDEN_SIZE, dtype=torch.bfloat16)}
    )

    out = model(input_ids=None, positions=torch.arange(3), intermediate_tensors=payload)

    assert layer.incoming == (None, None)
    torch.testing.assert_close(out, hidden_states + 1)


class _LocalStateLayer(nn.Module):
    def __init__(self, *, fail=False):
        super().__init__()
        self.fail = fail
        self.incoming = None

    def forward(self, positions, hidden_states, residual, post, comb):
        self.incoming = (hidden_states, residual, post, comb)
        if self.fail:
            raise RuntimeError("injected layer failure")
        count = hidden_states.shape[0]
        return (
            hidden_states + 1,
            torch.full((count, N_STREAMS, HIDDEN_SIZE), 4.0, dtype=hidden_states.dtype),
            torch.full((count, N_STREAMS, 1), 2.0),
            torch.full((count, N_STREAMS, N_STREAMS), 3.0),
        )


def _local_model(monkeypatch, *, tp_rank=2, tp_size=4):
    from vllm_ascend.models.glm5next.model import Glm5NextSPPPSpec, _normalize_sp_pp_state_impl

    model = _stub_model(mhc=True)
    model.is_sequence_parallel = True
    model._sp_pp_enabled = True
    model._glm5next_sp_pp_spec = Glm5NextSPPPSpec(tp_size, tp_rank, HIDDEN_SIZE, N_STREAMS, torch.bfloat16)
    model._active_layers = [_LocalStateLayer()]
    model.norm = nn.Identity()
    monkeypatch.setattr(glm5next_model, "normalize_sp_pp_state", _normalize_sp_pp_state_impl)
    monkeypatch.setattr(glm5next_model.envs, "VLLM_MOE_SKIP_PADDING", False)
    return model


def _local_payload(count=2):
    return IntermediateTensors(
        {
            "hidden_states": torch.arange(count * HIDDEN_SIZE).reshape(count, HIDDEN_SIZE).bfloat16(),
            "residual": torch.ones(count, N_STREAMS, HIDDEN_SIZE, dtype=torch.bfloat16),
            "post": torch.ones(count, N_STREAMS, 1),
            "comb": torch.ones(count, N_STREAMS, N_STREAMS),
        }
    )


def test_sp_pp_factory_allocates_local_capacity(monkeypatch):
    model = _local_model(monkeypatch)
    tensors = model.make_empty_intermediate_tensors(17, torch.bfloat16, torch.device("cpu"))
    assert set(tensors.tensors) == {"hidden_states", "residual", "post", "comb"}
    for key, value in tensors.tensors.items():
        assert value.shape[0] == 5
        assert value.untyped_storage().nbytes() == value.numel() * value.element_size()
        assert value.dtype == (torch.float32 if key in {"post", "comb"} else torch.bfloat16)


@pytest.mark.parametrize("tp_rank,valid_rows", [(0, 2), (2, 1), (3, 0)])
def test_sp_pp_receiver_preserves_four_states_without_resharding(monkeypatch, tp_rank, valid_rows):
    model = _local_model(monkeypatch, tp_rank=tp_rank)
    monkeypatch.setattr(glm5next_model, "get_pp_group", _pp_group(is_first_rank=False, is_last_rank=False))
    monkeypatch.setattr(glm5next_model, "sp_shard", lambda _: pytest.fail("receiver must not re-shard"))
    monkeypatch.setattr(glm5next_model, "sp_all_gather", lambda _: pytest.fail("PP boundary must not gather"))
    payload = _local_payload()
    out = model(None, torch.arange(5), payload)

    incoming = model._active_layers[0].incoming
    for actual, key in zip(incoming, ("hidden_states", "residual", "post", "comb")):
        assert actual is payload[key]
    for value in out.tensors.values():
        assert value.shape[0] == 2
        assert torch.count_nonzero(value[valid_rows:]) == 0
    torch.testing.assert_close(out["hidden_states"][:valid_rows], payload["hidden_states"][:valid_rows] + 1)
    if valid_rows:
        assert torch.count_nonzero(out["post"][:valid_rows]) > 0
    # The outgoing normalization must not overwrite receive backing.
    assert payload["post"].eq(1).all()


def test_sp_pp_first_stage_shards_embeddings_once(monkeypatch):
    model = _local_model(monkeypatch)
    monkeypatch.setattr(glm5next_model, "get_pp_group", _pp_group(is_first_rank=True, is_last_rank=False))
    calls = []

    def shard(value):
        calls.append(value)
        return torch.nn.functional.pad(value, (0, 0, 0, 3))[4:6]

    monkeypatch.setattr(glm5next_model, "sp_shard", shard)
    embeddings = torch.ones(5, HIDDEN_SIZE, dtype=torch.bfloat16)
    model(None, torch.arange(5), None, inputs_embeds=embeddings)
    assert len(calls) == 1 and calls[0] is embeddings
    assert model._active_layers[0].incoming[0].shape == (2, HIDDEN_SIZE)


def test_sp_pp_last_stage_returns_full_hidden_states(monkeypatch):
    model = _local_model(monkeypatch)
    monkeypatch.setattr(glm5next_model, "get_pp_group", _pp_group(is_first_rank=False, is_last_rank=True))
    monkeypatch.setattr(glm5next_model, "sp_shard", lambda _: pytest.fail("receiver must not re-shard"))
    gathered = torch.arange(8 * HIDDEN_SIZE).reshape(8, HIDDEN_SIZE).bfloat16()
    calls = []

    def gather(value):
        calls.append(value)
        return gathered

    monkeypatch.setattr(glm5next_model, "sp_all_gather", gather)
    out = model(None, torch.arange(5), _local_payload())
    assert len(calls) == 1 and calls[0].shape == (2, HIDDEN_SIZE)
    torch.testing.assert_close(out, gathered[:5])


@pytest.mark.parametrize("fail", [False, True])
def test_sp_pp_local_padding_mask_is_restored(monkeypatch, fail):
    model = _local_model(monkeypatch)
    layer = _LocalStateLayer(fail=fail)
    model._active_layers = [layer]
    monkeypatch.setattr(glm5next_model, "get_pp_group", _pp_group(is_first_rank=False, is_last_rank=False))
    monkeypatch.setattr(glm5next_model.envs, "VLLM_MOE_SKIP_PADDING", True)
    original_mask = torch.tensor([False, False, True, False, True])
    context = SimpleNamespace(is_padding=original_mask)
    monkeypatch.setattr(glm5next_model, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(glm5next_model, "get_forward_context", lambda: context)

    def shard_mask(mask, positions):
        assert mask is original_mask and positions.shape[0] == 5
        return torch.tensor([True, True])

    monkeypatch.setattr(glm5next_model, "sp_padding_mask", shard_mask)
    seen = []
    layer.register_forward_pre_hook(lambda *_: seen.append(context.is_padding.clone()))
    if fail:
        with pytest.raises(RuntimeError, match="injected layer failure"):
            model(None, torch.arange(5), _local_payload())
    else:
        model(None, torch.arange(5), _local_payload())
    assert seen[0].tolist() == [True, True]
    assert context.is_padding is original_mask


def _sp_pp_config():
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            use_sequence_parallel_moe=True,
            tensor_parallel_size=4,
            pipeline_parallel_size=2,
            use_ubatching=False,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            distributed_executor_backend="mp",
            nnodes=1,
        ),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(mhc=True), dtype=torch.bfloat16, runner_type="generate"
        ),
        use_v2_model_runner=True,
        speculative_config=None,
        lora_config=None,
    )


def _ascend_sp_pp_config():
    return SimpleNamespace(
        enable_dsa_cp=False,
        finegrained_tp_config=SimpleNamespace(
            oproj_tensor_parallel_size=0,
            lmhead_tensor_parallel_size=0,
            embedding_tensor_parallel_size=0,
            mlp_tensor_parallel_size=0,
        ),
    )


@pytest.mark.parametrize(
    "field,value",
    [("use_sequence_parallel_moe", False), ("tensor_parallel_size", 1), ("pipeline_parallel_size", 1)],
)
def test_sp_pp_gate_leaves_existing_modes_unchanged(monkeypatch, field, value):
    config = _sp_pp_config()
    setattr(config.parallel_config, field, value)
    config.use_v2_model_runner = False
    config.speculative_config = object()
    config.model_config.runner_type = "pooling"
    monkeypatch.setattr(
        glm5next_model, "get_ascend_config", lambda: pytest.fail("legacy mode must not inspect new gates")
    )
    assert not glm5next_model._is_glm_sp_pp_enabled(config)


@pytest.mark.parametrize(
    "section,field,value,match",
    [
        (None, "use_v2_model_runner", False, "Model Runner V2"),
        ("hf", "mhc", False, "mHC"),
        ("model_config", "runner_type", "pooling", "generation runner"),
        ("model_config", "dtype", torch.float16, "BF16"),
        ("parallel_config", "use_ubatching", True, "DBO"),
        ("parallel_config", "prefill_context_parallel_size", 2, "context parallelism"),
        ("parallel_config", "decode_context_parallel_size", 2, "context parallelism"),
        ("parallel_config", "distributed_executor_backend", "ray", "single-node mp"),
        ("parallel_config", "nnodes", 2, "single-node mp"),
        (None, "speculative_config", object(), "speculative decoding"),
        (None, "lora_config", object(), "LoRA"),
    ],
)
def test_sp_pp_gate_rejects_unvalidated_new_combinations(monkeypatch, section, field, value, match):
    config = _sp_pp_config()
    target = config.model_config.hf_text_config if section == "hf" else getattr(config, section) if section else config
    setattr(target, field, value)
    monkeypatch.setattr(glm5next_model, "get_ascend_config", _ascend_sp_pp_config)
    with pytest.raises(NotImplementedError, match=match):
        glm5next_model._is_glm_sp_pp_enabled(config)


def test_sp_pp_gate_accepts_only_basic_candidate(monkeypatch):
    monkeypatch.setattr(glm5next_model, "get_ascend_config", _ascend_sp_pp_config)
    assert glm5next_model._is_glm_sp_pp_enabled(_sp_pp_config())


@pytest.mark.parametrize("finegrained", [False, True])
def test_sp_pp_gate_rejects_ascend_parallel_extensions(monkeypatch, finegrained):
    config = _ascend_sp_pp_config()
    if finegrained:
        config.finegrained_tp_config.embedding_tensor_parallel_size = 2
    else:
        config.enable_dsa_cp = True
    monkeypatch.setattr(glm5next_model, "get_ascend_config", lambda: config)
    with pytest.raises(NotImplementedError):
        glm5next_model._is_glm_sp_pp_enabled(_sp_pp_config())


@pytest.mark.parametrize("enabled", [False, True])
def test_only_new_combination_exposes_model_state_on_both_wrappers(monkeypatch, enabled):
    from vllm_ascend.models.glm5next.model import Glm5NextSPPPSpec

    inner = _stub_model(mhc=True)
    inner._sp_pp_enabled = enabled
    if enabled:
        inner._glm5next_sp_pp_spec = Glm5NextSPPPSpec(4, 2, HIDDEN_SIZE, N_STREAMS, torch.bfloat16)
    monkeypatch.setattr(glm5next_model, "Glm5NextModel", lambda **_: inner)
    monkeypatch.setattr(glm5next_model, "get_pp_group", _pp_group(is_first_rank=False, is_last_rank=False))
    monkeypatch.setattr(glm5next_model, "LogitsProcessor", lambda *_, **__: nn.Identity())
    text_config = SimpleNamespace(vocab_size=16, logit_scale=1.0)
    config = SimpleNamespace(model_config=SimpleNamespace(hf_text_config=text_config), quant_config=None)
    text = glm5next_model.Glm5NextForCausalLM(vllm_config=config)
    assert hasattr(text, "get_model_state_cls") is enabled
    assert hasattr(text, "_glm5next_sp_pp_spec") is enabled

    wrapper_cls = glm5next_model.Glm5NextForConditionalGeneration
    monkeypatch.setattr(wrapper_cls, "_mark_tower_model", lambda *_, **__: nullcontext())
    monkeypatch.setattr(wrapper_cls, "_mark_language_model", lambda *_, **__: nullcontext())
    monkeypatch.setattr(glm5next_model, "Glm5NextVisionTransformer", lambda *_, **__: nn.Identity())
    monkeypatch.setattr(glm5next_model, "init_vllm_registered_model", lambda **_: text)
    full_config = SimpleNamespace(text_config=text_config, vision_config=SimpleNamespace(rms_norm_eps=1e-6))
    mm_config = SimpleNamespace(mm_encoder_tp_mode="weights", is_multimodal_pruning_enabled=lambda: False)
    config.model_config = SimpleNamespace(
        hf_config=full_config, hf_text_config=text_config, multimodal_config=mm_config
    )
    outer = wrapper_cls(vllm_config=config)
    assert hasattr(outer, "get_model_state_cls") is enabled
    assert hasattr(outer, "_glm5next_sp_pp_spec") is enabled
    if enabled:
        assert outer._glm5next_sp_pp_spec is inner._glm5next_sp_pp_spec
        assert outer.get_model_state_cls is text.get_model_state_cls


@pytest.mark.parametrize("partition", ["0,4,4", "5,-1,4", "4,4,0"])
@pytest.mark.parametrize("pp_rank", range(3))
def test_sp_pp_rejects_any_nonpositive_stage_before_constructing_layers(monkeypatch, partition, pp_rank):
    monkeypatch.setattr(glm5next_model.envs, "VLLM_PP_LAYER_PARTITION", partition)
    monkeypatch.setattr(glm5next_model, "_is_glm_sp_pp_enabled", lambda _: True)
    monkeypatch.setattr(glm5next_model, "get_pp_group", lambda: SimpleNamespace(world_size=3, rank_in_group=pp_rank))
    monkeypatch.setattr(
        glm5next_model, "make_layers", lambda *_, **__: pytest.fail("validation must precede make_layers")
    )
    monkeypatch.setattr(
        glm5next_model,
        "get_tensor_model_parallel_world_size",
        lambda: pytest.fail("validation must precede device/layout construction"),
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(num_hidden_layers=8, index_topk=None)),
        parallel_config=SimpleNamespace(use_sequence_parallel_moe=True),
    )
    with pytest.raises(NotImplementedError, match="strictly positive layer counts on every stage"):
        Glm5NextModel(vllm_config=config)


@pytest.mark.parametrize("partition,num_layers,pp_size", [(None, 8, 3), ("1,2,5", 8, 3)])
def test_sp_pp_partition_validation_reuses_upstream_for_every_stage(monkeypatch, partition, num_layers, pp_size):
    monkeypatch.setattr(glm5next_model.envs, "VLLM_PP_LAYER_PARTITION", partition)
    upstream = glm5next_model.get_pp_indices
    calls = []

    def resolve(count, rank, size):
        calls.append((count, rank, size))
        return upstream(count, rank, size)

    monkeypatch.setattr(glm5next_model, "get_pp_indices", resolve)
    glm5next_model._validate_glm_sp_pp_partitions(num_layers, pp_size)
    assert calls == [(num_layers, rank, pp_size) for rank in range(pp_size)]


def test_sp_pp_partition_validation_rejects_empty_default_partition(monkeypatch):
    monkeypatch.setattr(glm5next_model.envs, "VLLM_PP_LAYER_PARTITION", None)
    with pytest.raises(NotImplementedError, match="strictly positive layer counts on every stage"):
        glm5next_model._validate_glm_sp_pp_partitions(2, 4)
