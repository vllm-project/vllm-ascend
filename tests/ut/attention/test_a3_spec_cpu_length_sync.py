# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Check the target-side K3.1 speculative length policy without a native provider."""

import ast
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


def ast_method(path, owner, name, scope):
    module = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == owner)
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    fn.decorator_list = []
    # Keep implementation intact; remove annotations requiring unavailable vLLM types.
    fn.returns = None
    for arg in (*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs):
        arg.annotation = None
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(ROOT / path), "exec"), scope)
    return scope[name]


def common(lengths=(15, 7), prefixes=(0, 2, 3), capacity=8):
    return SimpleNamespace(
        num_actual_tokens=prefixes[-1],
        num_input_tokens=capacity,
        causal=False,
        seq_lens=torch.tensor(lengths, dtype=torch.int32),
        query_start_loc=torch.tensor(prefixes, dtype=torch.int32),
        block_table_tensor=torch.tensor([[0], [1]], dtype=torch.int32),
        slot_mapping=torch.arange(prefixes[-1]),
        positions=torch.arange(prefixes[-1]),
    )


def runner(ready=True):
    backend = type("UnknownBackend", (), {})
    state = SimpleNamespace()
    speculator = SimpleNamespace(
        use_flashmla=False, use_a3_mla_sink=True, draft_attn_layer_names={"draft.0"}, attn_backends={"draft.0": backend}
    )
    spec = SimpleNamespace(
        method="dspark",
        draft_model_config=SimpleNamespace(architectures=["K3DSparkModel"]),
        enable_adaptive_verification=False,
    )
    instance = SimpleNamespace(
        sync_spec_pp_cpu_counts=False,
        model_state=state,
        _a3_target_device_lengths=ready,
        speculator=speculator,
        speculative_config=spec,
        model_config=SimpleNamespace(hf_config=SimpleNamespace(), architecture="KimiK31ForConditionalGeneration"),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1, prefill_context_parallel_size=1, decode_context_parallel_size=8
        ),
        ascend_config=SimpleNamespace(sparse_kv_offload_config=SimpleNamespace(enabled=False)),
        vllm_config=SimpleNamespace(kv_transfer_config=None),
    )
    scope = dict(
        torch=torch,
        is_deepseek_v41=lambda cfg: False,
        AscendFlashAttentionBackend=backend,
    )
    path = "vllm_ascend/worker/v2/model_runner.py"
    fn = ast_method(path, "NPUModelRunner", "_can_skip_spec_cpu_length_sync", scope)
    instance._can_skip_spec_cpu_length_sync = lambda: fn(instance)
    return instance, scope


@pytest.mark.parametrize("ready", [False, True])
def test_real_runner_copy_and_wait_share_the_same_predicate(monkeypatch, ready):
    instance, scope = runner(ready)
    assert instance._can_skip_spec_cpu_length_sync() is ready
    instance.req_states = SimpleNamespace(
        num_computed_tokens=SimpleNamespace(gpu=torch.tensor([98, 195])),
        num_computed_tokens_cpu=torch.tensor([100, 200]),
        req_id_to_index={"a": 0, "b": 1},
    )
    instance.num_computed_tokens_cpu = torch.full((2,), -99)
    events = []
    instance.num_computed_tokens_stream = SimpleNamespace(wait_stream=lambda stream: events.append("wait"))
    instance.num_computed_tokens_event = SimpleNamespace(
        record=lambda: events.append("record"), synchronize=lambda: events.append("synchronize")
    )
    instance.input_buffers = SimpleNamespace(seq_lens_cpu=torch.zeros(2, dtype=torch.int32))
    scheduler = SimpleNamespace(
        num_scheduled_tokens={"a": 4, "b": 4}, scheduled_cached_reqs=SimpleNamespace(req_ids=["b", "a"])
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: object())
    monkeypatch.setattr(torch, "npu", SimpleNamespace(stream=lambda stream: nullcontext()), raising=False)
    path = "vllm_ascend/worker/v2/model_runner.py"
    copy = ast_method(path, "NPUModelRunner", "_copy_num_computed_tokens_to_cpu", scope)
    update = ast_method(path, "NPUModelRunner", "_update_seq_lens_cpu", scope)
    copy(instance)
    update(instance, scheduler, ["b", "a"])
    assert events.count("synchronize") == int(not ready)
    assert events.count("record") == int(not ready)
    assert instance.num_computed_tokens_cpu.tolist() == ([-99, -99] if ready else [98, 195])
    assert instance.input_buffers.seq_lens_cpu.tolist() == ([204, 104] if ready else [199, 102])


@pytest.mark.parametrize("reason", ["pp", "pcp", "offload", "adaptive", "fia"])
def test_sync_fallback_for_every_unsupported_consumer(reason):
    instance, _ = runner()
    if reason == "pp":
        instance.parallel_config.pipeline_parallel_size = 2
    elif reason == "pcp":
        instance.parallel_config.prefill_context_parallel_size = 2
    elif reason == "offload":
        instance.ascend_config.sparse_kv_offload_config.enabled = True
    elif reason == "adaptive":
        instance.speculative_config.enable_adaptive_verification = True
    else:
        instance.speculator.use_a3_mla_sink = False
    assert instance._can_skip_spec_cpu_length_sync() is False


@pytest.mark.parametrize(
    "case",
    [
        "supported",
        "unknown",
        "empty",
        "draft_only",
        "state",
        "recover",
        "custom_recover",
        "unknown_recover_builder",
        "pcp",
    ],
)
def test_bind_target_lengths_once(case, monkeypatch):
    from types import ModuleType

    modules = {}
    for path, name in (
        ("attention.context_parallel.sfa_cp", "AscendSFADCPMetadataBuilder"),
        ("attention.sfa_v1", "AscendSFAMetadataBuilder"),
        ("attention.indexer", "AscendSFAIndexerMetadataBuilder"),
        ("ops.kimi_kda", "KimiKDAHostMetadataBuilder"),
        ("models.kimi_k3.recoverssm_metadata", "AscendKimiKDARecoverSSMMetadataBuilder"),
        ("worker.v2.model_states.mamba_hybrid", "AscendMambaHybridModelState"),
    ):
        module = ModuleType("vllm_ascend." + path)
        modules[name] = type(name, (), {})
        setattr(module, name, modules[name])
        monkeypatch.setitem(sys.modules, module.__name__, module)
    state = modules["AscendMambaHybridModelState"]()
    recover_module = ModuleType("vllm.v1.worker.gpu.model_states.recoverssm")
    recover_type = type("RecoverSSMState", (), {})
    recover_module.RecoverSSMState = recover_type
    monkeypatch.setitem(sys.modules, recover_module.__name__, recover_module)
    state.recoverssm = None
    kda = modules["KimiKDAHostMetadataBuilder"]
    if case in ("recover", "custom_recover", "unknown_recover_builder"):
        state.recoverssm = recover_type()
        kda = modules["AscendKimiKDARecoverSSMMetadataBuilder"]
    if case == "custom_recover":
        state.recoverssm = type("CustomRecoverState", (recover_type,), {})()
    if case == "unknown_recover_builder":
        kda = type("CustomRecoverBuilder", (kda,), {})
    builders = [
        modules[name]()
        for name in ("AscendSFADCPMetadataBuilder", "AscendSFAMetadataBuilder", "AscendSFAIndexerMetadataBuilder")
    ]
    builders.append(kda())
    if case == "unknown":
        builders.append(object())
    group = SimpleNamespace(layer_names=["draft" if case == "draft_only" else "target"], metadata_builders=builders)
    runner = SimpleNamespace(
        model_state=object() if case == "state" else state,
        attn_groups=[] if case == "empty" else [[group]],
        parallel_config=SimpleNamespace(prefill_context_parallel_size=2 if case == "pcp" else 1),
    )
    configure = ast_method(
        "vllm_ascend/worker/v2/model_runner.py", "NPUModelRunner", "_configure_a3_target_lengths", {}
    )
    expected = case in ("supported", "recover")
    assert configure(runner, {"draft"}) is expected
    for sfa in builders[:2]:
        assert getattr(sfa, "allow_a3_device_lengths", False) is expected


@pytest.mark.parametrize("has_device_metadata", [False, True])
def test_a5_sync_policy_retains_original_target_producer_requirement(has_device_metadata):
    instance, _ = runner(False)
    instance.speculator.use_a3_mla_sink = False
    instance.speculator.use_flashmla = True
    instance.model_state = SimpleNamespace(device_metadata=object() if has_device_metadata else None)
    # A5 must not inspect the A3 consumer declarations or require A3 groups.
    assert instance._can_skip_spec_cpu_length_sync() is has_device_metadata


@pytest.mark.parametrize("host_mirror", [None, 9999])
def test_sfa_build_consumes_device_lengths_without_exact_cpu_mirror(monkeypatch, host_mirror):
    cls = type("AscendSFAMetadataBuilder", (), {})
    scope = {
        "AscendSFAMetadataBuilder": cls,
        "get_cos_and_sin_mla": lambda *args, **kwargs: (torch.zeros(3, 64), torch.zeros(3, 64)),
        "split_decodes_and_prefills": lambda *args, **kwargs: (2, 0, 3, 0),
    }
    build = ast_method("vllm_ascend/attention/sfa_v1.py", cls.__name__, "_build", scope)
    instance = cls()
    instance.use_pcp = instance.nope = instance.use_rope = False
    instance.allow_a3_device_lengths = True
    instance.kernel_block_size = 128
    instance.decode_threshold = 8
    instance.metadata_cls = SimpleNamespace
    instance.model_config = SimpleNamespace(get_head_size=lambda: 576)
    instance.attn_mask_builder = SimpleNamespace(get_attention_mask=lambda *args: None)
    inputs = common(capacity=3)
    inputs.num_reqs = 2
    inputs._seq_lens_cpu = None if host_mirror is None else torch.full((2,), host_mirror)
    inputs.seq_lens_cpu = None
    inputs.attn_state = object()
    inputs.max_query_len = 2
    inputs.max_seq_len = 16384
    original_to = torch.Tensor.to

    def no_cpu_copy(tensor, *args, **kwargs):
        if args and str(args[0]) == "cpu" or str(kwargs.get("device")) == "cpu":
            pytest.fail("SFA rebuilt an exact CPU sequence-length mirror")
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", no_cpu_copy)
    actual = build(instance, inputs)
    assert actual.seq_lens.tolist() == [15, 7]
    assert actual.cum_query_lens.tolist() == [2, 3]
    assert actual.seq_lens.data_ptr() == inputs.seq_lens.data_ptr()
    if host_mirror is None:
        assert actual.seq_lens_cpu is None
        # Other model/hardware paths retain the pre-existing fallback.
        del instance.allow_a3_device_lengths
        monkeypatch.setattr(torch.Tensor, "to", original_to)
        fallback = build(instance, inputs)
        assert fallback.seq_lens_cpu.tolist() == [15, 7]
    else:
        assert actual.seq_lens_cpu.tolist() == [9999, 9999]


def test_initialize_length_policy_tracks_actual_draft_layers_on_reinitialization():
    initialize = ast_method(
        "vllm_ascend/worker/v2/model_runner.py", "NPUModelRunner", "initialize_spec_cpu_length_sync", {}
    )
    instance, _ = runner()

    # The target and draft may coexist in the static forward context.
    def layer(selected):
        return SimpleNamespace(impl=SimpleNamespace(use_a3_mla_sink=selected))

    context = {"draft.0": layer(False), "target": layer(True)}
    instance.compilation_config = SimpleNamespace(static_forward_context=context)
    audits = []

    def audit(names):
        audits.append(set(names))
        return True

    instance._configure_a3_target_lengths = audit
    initialize(instance)
    assert not instance._a3_target_device_lengths and not audits
    context["draft.0"] = layer(True)
    initialize(instance)
    assert instance._a3_target_device_lengths and audits == [{"draft.0"}]
    context.pop("draft.0")
    initialize(instance)
    assert not instance._a3_target_device_lengths and audits == [{"draft.0"}]
