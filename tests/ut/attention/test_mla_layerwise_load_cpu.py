# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run production MLA dispatch and pool event handling with CPU kernel doubles."""

import ast
from pathlib import Path
from threading import Event
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
MLA = "vllm_ascend/attention/mla_v1.py"
POOL = "vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/layerwise/pool_worker.py"


def load(path, name, scope, owner=None):
    # Avoid importing NPU operators: execute the unchanged production methods.
    nodes = ast.parse((ROOT / path).read_text(encoding="utf-8")).body
    if owner is not None:
        nodes = next(node for node in nodes if isinstance(node, ast.ClassDef) and node.name == owner).body
    node = next(node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == name)
    node.decorator_list = []
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
    exec(compile(module, path, "exec"), scope)
    return scope[name]


def make_pool(has_load):
    event = Event()
    tracked_event = Mock(wraps=event)

    def wait(timeout):
        # A second consumption must fail immediately rather than hang the UT.
        assert event.is_set(), "load completion was consumed more than once"
        return event.wait(timeout=0)

    tracked_event.wait.side_effect = wait
    worker = SimpleNamespace(
        current_layer=3,
        num_layers=4,
        use_kvpp=False,
        layer_load_finished_events=[Event(), Event(), Event(), tracked_event],
        layer_load_tasks=[[], [], [], [object()] if has_load else []],
        prefetch_layer_map={},
        external_slot_release_waiter=None,
        kv_recv_thread=SimpleNamespace(raise_if_failed=Mock()),
        _submit_ready_layer_loads=Mock(),
    )
    scope = {"reset_attention_compute_start_gate": Mock(), "logger": Mock()}
    worker.wait_for_layer_load = MethodType(load(POOL, "wait_for_layer_load", scope, "LayerwiseKVPoolWorker"), worker)
    return worker, event, tracked_event


def make_forward(path, batch, wait):
    decode_tokens = 0 if batch == "prefill" else 4
    prefill_tokens = 0 if batch == "decode" else 2
    tokens = decode_tokens + prefill_tokens
    metadata = SimpleNamespace(
        num_actual_tokens=tokens,
        num_decode_tokens=decode_tokens,
        num_decodes=int(decode_tokens > 0),
        num_prefills=int(prefill_tokens > 0),
        uses_pcp_sharded_layout=False,
        external_flashmla=None,
        slot_mapping=torch.arange(tokens),
        decode=SimpleNamespace(cos=torch.zeros(tokens, 2), sin=torch.zeros(tokens, 2)),
        prefill=SimpleNamespace(cos=torch.zeros(prefill_tokens, 2), sin=torch.zeros(prefill_tokens, 2)),
    )
    if batch == "decode":
        # A 3076-token prompt with 3072 cached tokens is still prefilling,
        # but its four scheduled rows use the decode path at spec=3.
        common = SimpleNamespace(
            context_parallel_metadata=None,
            num_reqs=1,
            num_actual_tokens=4,
            max_query_len=4,
            query_start_loc_cpu=torch.tensor([0, 4]),
            is_prefilling=torch.tensor([True]),
        )
        classify = load(
            "vllm_ascend/attention/utils.py",
            "split_decodes_and_prefills",
            {"is_pd_decode_recompute_scheduler_enabled": lambda: False},
        )
        decodes, prefills, decode_rows, prefill_rows = classify(common, decode_threshold=4)
        assert (decodes, prefills, decode_rows, prefill_rows) == (1, 0, 4, 0)
        metadata.num_decodes, metadata.num_prefills, metadata.num_decode_tokens = decodes, prefills, decode_rows
    cache = (torch.zeros(2, 4, 1, 2), torch.zeros(2, 4, 1, 2))
    if path == "strided_fallback":
        cache = tuple(tensor.transpose(0, 1) for tensor in cache)
    writes = []
    execution_order = []

    def connector_wait(layer_name):
        execution_order.append("connector_wait")
        wait(layer_name)

    def project(hidden):
        execution_order.append("projection")
        return (torch.zeros(hidden.shape[0], 6),)

    def project_kv(hidden):
        execution_order.append("projection")
        return (torch.zeros(hidden.shape[0], 4),)

    def decode_result(q, pe, k, kr, **kwargs):
        return SimpleNamespace(ql_nope=q, q_pe=pe, k_nope=k, k_pe=kr, **kwargs)

    def prefill_result(q, pe, k, kr, value):
        return SimpleNamespace(q_nope=q, q_pe=pe, k_nope=k, k_pe=kr, value=value)

    def record_write(kind):
        assert wait.call_count == 1, "KV access must follow exactly one connector wait"
        execution_order.append("cache_access")
        writes.append(kind)

    def decode(*_args):
        record_write("decode")
        return decode_result(None, None, *cache)

    def prefill(*_args):
        record_write("prefill")
        return prefill_result(None, None, *cache, None)

    def prolog(hidden, *_args, **_kwargs):
        record_write("prolog")
        q = torch.zeros(hidden.shape[0], 1, 2)
        return q, q.clone(), None

    scope = {
        "torch": torch,
        "_EXTRA_CTX": SimpleNamespace(num_tokens=tokens),
        "MLAPO_MAX_SUPPORTED_TOKENS": 16,
        "wait_for_kv_layer_from_connector": connector_wait,
        "execution_order": execution_order,
        "notify_kv_cache_written": Mock(),
        "maybe_save_kv_layer_to_connector": Mock(),
        "DecodeMLAPreprocessResult": decode_result,
        "PrefillMLAPreprocessResult": prefill_result,
    }
    impl = SimpleNamespace(
        layer_name="language_model.model.layers.3.self_attn.attn",
        num_heads=1,
        num_kv_heads=1,
        v_head_dim=2,
        kv_lora_rank=2,
        q_lora_rank=2,
        qk_rope_head_dim=2,
        qk_nope_head_dim=2,
        qk_head_dim=4,
        use_output_gate=False,
        fa_quant_layer=False,
        enable_mlapo=path == "fused",
        use_mla_rope=True,
        enable_kv_nz=False,
        use_pcp=False,
        use_dcp=False,
        pcp_enabled=False,
        dcp_enabled=False,
        is_pcp_decode_sharded=False,
        layerwise_kv_cache_hook=None,
        get_num_actual_tokens=lambda meta: meta.num_actual_tokens,
        _decode_requires_current_kv=lambda _: False,
        _prefill_requires_current_kv=lambda _: False,
        _cache_requires_prolog_v3_writer=lambda _: path == "prolog",
        _uses_strided_kv_adapter=lambda: path == "strided_fallback",
        fused_qkv_a_proj=project,
        _split_fused_qkv_lora=lambda qkv: (*qkv.split([2, 4], dim=-1), None),
        kv_a_proj_with_mqa=project_kv,
        q_a_layernorm=lambda x: x,
        q_proj=lambda x: (torch.zeros(x.shape[0], 4),),
        kv_b_proj=lambda x: (torch.zeros(x.shape[0], 4),),
        rope_single=lambda x, *_args: x,
        mla_preprocess_decode=decode,
        mla_preprocess_prefill=prefill,
        mla_preprocess_only_decode=lambda *_args: (decode(), None),
        _run_prolog_v3=prolog,
        _prolog_v3_weights={"num_heads": 1},
        reorg_decode_q=lambda q, pe: (q, pe),
        _gather_current_kv_from_cache=lambda _, slots: (
            torch.zeros(slots.numel(), 1, 2),
            torch.zeros(slots.numel(), 1, 2),
        ),
        _forward_decode=lambda *_args: torch.ones(decode_tokens, 2),
        _forward_prefill=lambda *_args: torch.ones(prefill_tokens, 2),
        o_proj=lambda x, **_kwargs: (x,),
    )
    for name in ("forward", "_mla_preprocess", "mla_preprocess_prolog_v3"):
        setattr(impl, name, MethodType(load(MLA, name, scope, "AscendMLAImpl"), impl))
    return impl, metadata, cache, writes, scope


@pytest.mark.parametrize(
    "path,batch",
    [
        ("native", "decode"),
        ("native", "prefill"),
        ("native", "mixed"),
        ("strided_fallback", "decode"),
        ("strided_fallback", "prefill"),
        ("strided_fallback", "mixed"),
        ("prolog", "decode"),
        ("prolog", "prefill"),
        ("prolog", "mixed"),
        ("fused", "decode"),
    ],
)
@pytest.mark.parametrize("has_load", [False, True])
def test_layer_load_consumed_once_before_any_mla_cache_access(path, has_load, batch):
    worker, event, tracked_event = make_pool(has_load)

    def wait_for_layer(_layer_name):
        worker.wait_for_layer_load()
        assert not event.is_set()

    wait = Mock(side_effect=wait_for_layer)
    impl, metadata, cache, writes, scope = make_forward(path, batch, wait)
    for _step in range(2):
        # Reproduce a completed prefetch at MLA layer 3 followed by another load.
        assert not event.is_set(), "previous step left a stale completion event"
        if has_load:
            event.set()
        wait.reset_mock()
        scope["execution_order"].clear()
        output = torch.empty(metadata.num_actual_tokens, 2)
        impl.forward(impl.layer_name, torch.zeros(metadata.num_actual_tokens, 2), cache, metadata, output)
        wait.assert_called_once_with(impl.layer_name)
        order = scope["execution_order"]
        assert order.index("connector_wait") < order.index("cache_access")
        if path in ("native", "strided_fallback"):
            assert order.index("projection") < order.index("connector_wait")
        assert not event.is_set()
        torch.testing.assert_close(output, torch.ones_like(output))
    assert writes
    assert tracked_event.wait.call_count == (2 if has_load else 0)
    assert tracked_event.clear.call_count == 2
    assert scope["maybe_save_kv_layer_to_connector"].call_count == 2


@pytest.mark.parametrize("path", ["native", "strided_fallback"])
@pytest.mark.parametrize("fused_projection", [False, True])
@pytest.mark.parametrize("batch", ["decode", "prefill", "mixed"])
def test_native_projections_stay_before_connector_and_kvpp_waits(path, fused_projection, batch):
    wait = Mock()
    impl, metadata, cache, _, scope = make_forward(path, batch, wait)
    if not fused_projection:
        impl.fused_qkv_a_proj = None
    order = scope["execution_order"]
    impl.layerwise_kv_cache_hook = SimpleNamespace(wait_for_layer=lambda _: order.append("kvpp_wait"))
    impl.forward(
        impl.layer_name,
        torch.zeros(metadata.num_actual_tokens, 2),
        cache,
        metadata,
        torch.empty(metadata.num_actual_tokens, 2),
    )
    assert order[:3] == ["projection", "connector_wait", "kvpp_wait"]
    assert order[3:] == ["cache_access"] * (2 if batch == "mixed" else 1)
    wait.assert_called_once_with(impl.layer_name)


def test_profile_run_does_not_wait_for_connector():
    wait = Mock()
    impl, _, _, writes, _ = make_forward("native", "decode", wait)
    output = torch.ones(4, 2)
    impl.forward(impl.layer_name, torch.zeros(4, 2), (), None, output)
    wait.assert_not_called()
    assert not writes
    assert torch.count_nonzero(output) == 0


@pytest.mark.parametrize("path", ["native", "strided_fallback", "prolog", "fused"])
def test_failed_load_stops_mla_before_cache_access(path):
    wait = Mock(side_effect=RuntimeError("asynchronous transfer failed"))
    impl, metadata, cache, writes, scope = make_forward(path, "decode", wait)
    with pytest.raises(RuntimeError, match="asynchronous transfer failed"):
        impl.forward(impl.layer_name, torch.zeros(4, 2), cache, metadata, torch.empty(4, 2))
    assert not writes
    scope["maybe_save_kv_layer_to_connector"].assert_not_called()


@pytest.mark.parametrize("has_group,has_metadata", [(False, True), (True, False)])
def test_connector_wait_noops_without_transfer_metadata(has_group, has_metadata):
    connector = SimpleNamespace(has_connector_metadata=lambda: has_metadata, wait_for_layer_load=Mock())
    scope = {
        "has_kv_transfer_group": lambda: has_group,
        "is_v1_kv_transfer_group": lambda: True,
        "get_kv_transfer_group": lambda: connector,
        "get_forward_context": lambda: SimpleNamespace(attn_metadata={}),
    }
    wait = load("vllm_ascend/attention/utils.py", "wait_for_kv_layer_from_connector", scope)
    wait("mla")
    connector.wait_for_layer_load.assert_not_called()


@pytest.mark.parametrize("speculative", [False, True])
def test_decode_attention_opens_gate_for_next_layer_prefetch(speculative):
    # A real receiver waits on this boundary before copying the next layer.
    # Without the signal, adding the MLA load wait deadlocks on layer 1.
    gate = Event()
    attention_calls = []

    def attention(query, *_args, **_kwargs):
        assert gate.is_set(), "decode left subsequent layer prefetch blocked"
        attention_calls.append(True)
        return query, None

    state = SimpleNamespace(DecodeOnly=0, SpecDecoding=1, ChunkedPrefill=2, PrefillNoCache=3)
    scope = {
        "torch": torch,
        "torch_npu": SimpleNamespace(npu_fused_infer_attention_score_v2=attention),
        "AscendAttentionState": state,
        "_EXTRA_CTX": SimpleNamespace(is_draft_model=False, capturing=False),
        "get_graph_params": lambda: None,
        "record_attention_compute_start": gate.set,
    }
    impl = SimpleNamespace(
        fa_quant_layer=False,
        enable_kv_nz=False,
        kv_lora_rank=2,
        qk_rope_head_dim=2,
        num_heads=1,
        num_heads_padded=1,
        num_kv_heads=1,
        head_padding=0,
        scale=1.0,
        speculative_config=object() if speculative else None,
        _v_up_proj=lambda value: value,
    )
    result = SimpleNamespace(
        ql_nope=torch.ones(4, 1, 2),
        q_pe=torch.ones(4, 1, 2),
        k_nope=torch.ones(1, 1, 128, 2),
        k_pe=torch.ones(1, 1, 128, 2),
        dequant_scale_q_nope=None,
    )
    metadata = SimpleNamespace(
        attn_state=state.DecodeOnly,
        causal=True,
        decode=SimpleNamespace(block_table=None, seq_lens_list=[4], actual_seq_lengths_q=[4], attn_mask=None),
    )
    decode = load(MLA, "_forward_decode", scope, "AscendMLAImpl")
    decode(impl, result, 128, metadata)
    assert attention_calls == [True]
