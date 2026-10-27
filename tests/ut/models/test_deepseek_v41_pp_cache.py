# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# ruff: noqa: E402

from types import SimpleNamespace

import pytest

pytest.importorskip("vllm.transformers_utils.configs.deepseek_v41")

import torch
from vllm.sequence import IntermediateTensors

from tests.deepseek_v41_utils import allocate_cache_views, make_cache_config
from vllm_ascend.models.deepseek_v41 import model as v41_model
from vllm_ascend.models.deepseek_v41.model import CANDIDATE_LENGTHS, CANDIDATES, TOPK, TOPK_LENGTHS, build_layer_plan
from vllm_ascend.mrv2_utils import use_v2_model_runner
from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.v2 import pp_transport


def _config():
    return SimpleNamespace(
        num_hidden_layers=40,
        compress_ratios=[0, 0] + [2] * 18 + [1] * 20,
        kv_source_layer_ids=[2, 8, 14, 20],
        index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
        candidate_source_layer_id=20,
        candidate_topk_blocks=3,
        candidate_block_size=8,
        index_topk=5,
        engram_layer_ids=[],
        head_dim=8,
        index_head_dim=4,
    )


def _partitions(lengths):
    ends = [0]
    for length in lengths:
        ends.append(ends[-1] + length)
    return tuple(zip(ends, ends[1:]))


@pytest.mark.parametrize("boundary", range(1, 40))
def test_every_two_stage_boundary_resolves_exact_dependencies(boundary):
    config = _config()
    (plan,) = build_layer_plan(config).pp_boundaries(((0, boundary), (boundary, 40)))
    # Independent source lifetimes for the actual checkpoint topology.
    expected_cache = next(
        (source for source, end in [(2, 8), (8, 14), (14, 20), (20, 40)] if source < boundary < end), None
    )
    assert plan.cache_sources == (() if expected_cache is None else (expected_cache,))
    assert plan.topk == (boundary >= 3 and boundary not in config.index_source_layer_ids)
    assert plan.candidates == (20 < boundary <= 36)


@pytest.mark.parametrize("lengths", [(0, 40), (20, 19), (41,), (20, 0, 20)])
def test_invalid_partitions_fail_before_constructing_mirrors(lengths):
    with pytest.raises(ValueError, match="partition"):
        build_layer_plan(_config()).pp_boundaries(_partitions(lengths))


def test_multi_stage_plan_forwards_dependencies_past_stages_with_no_source():
    plans = build_layer_plan(_config()).pp_boundaries(_partitions([1] * 40))
    assert len(plans) == 39
    assert all(plan.cache_sources == (20,) and plan.candidates for plan in plans[20:36])
    assert plans[23].boundary == 24 and not plans[23].topk
    assert plans[24].topk
    assert not plans[36].candidates and plans[36].topk


def _plane(blocks, rows, width, dtype, fill):
    stride = rows * width + 11
    backing = torch.full((blocks * stride + 13,), fill, dtype=dtype)
    return torch.as_strided(backing, (blocks, rows, 1, width), (stride, width, width, 1), 13)


def _reference_gather(cache, slots, output):
    physical = slots.clamp_min(0).long()
    values = cache.squeeze(-2)[physical // cache.shape[1], physical % cache.shape[1]]
    output.copy_(torch.where((slots >= 0)[:, None], values, 0))


def _reference_scatter(cache, coordinates, values):
    valid = (coordinates[:, 0] >= 0) & (coordinates[:, 1] >= 0)
    cache.squeeze(-2)[coordinates[valid, 0].long(), coordinates[valid, 1].long()] = values[valid]


def _shared(fill=0):
    return SimpleNamespace(
        candidates=torch.arange(24, dtype=torch.int32).reshape(8, 1, 3) + fill,
        candidate_lengths=torch.arange(8, dtype=torch.int32).reshape(8, 1) + fill,
        topk_indices=torch.arange(40, dtype=torch.int32).reshape(8, 5) + fill,
        topk_lengths=torch.arange(8, dtype=torch.int32).reshape(8, 1) + fill,
    )


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.setattr("vllm_ascend.models.deepseek_v41.cache_config.uses_a5_packed_cache", lambda: False)
    monkeypatch.setattr(v41_model, "uses_a5_packed_cache", lambda: False)
    monkeypatch.setattr(pp_transport, "gather_pp_cache_rows", _reference_gather)
    monkeypatch.setattr(v41_model, "scatter_cache_sk", _reference_scatter)
    context = SimpleNamespace(attn_metadata=None, no_compile_layers={})
    monkeypatch.setattr(v41_model, "get_forward_context", lambda: context)
    monkeypatch.setattr(pp_transport, "get_forward_context", lambda: context)
    monkeypatch.setattr(pp_transport, "is_forward_context_available", lambda: True)
    library = torch.library.Library("vllm", "IMPL", "CPU")
    library.impl("pp_dependency_export", pp_transport._export_dependencies)
    library.impl("pp_dependency_import", pp_transport._import_dependencies)
    yield context
    library._destroy()
    torch._dynamo.reset()


def _stage(monkeypatch, partitions: tuple[tuple[int, int], ...], rank: int, shared, pcp_size=1):
    config = _config()
    runtime = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=4),
        parallel_config=SimpleNamespace(pipeline_parallel_size=len(partitions), prefill_context_parallel_size=pcp_size),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    monkeypatch.setattr(v41_model, "get_pp_indices", lambda layers, r, size: partitions[r])
    transport = v41_model.make_pp_cache_transport(
        runtime, config, "model", *partitions[rank], shared, build_layer_plan(config)
    )
    return transport, runtime.compilation_config.static_forward_context


def _metadata(slots, rows=4):
    slots = torch.tensor(slots, dtype=torch.int64)
    coordinates = torch.stack((slots // rows, slots % rows), dim=-1).int()
    coordinates[slots < 0] = -1
    return SimpleNamespace(flat_slot_mapping=slots, slot_mapping=coordinates)


def _bind_source(layers, handler, source, fill=0):
    planes = [_plane(8, 4, w, d, fill) for w, d in [(8, torch.bfloat16), (4, torch.int8), (1, torch.float16)]]
    long_prefix, index_prefix = handler.cache_prefixes(source)
    for prefix, cache in [(long_prefix, planes[0]), (index_prefix, tuple(planes[1:]))]:
        if prefix in layers:
            layers[prefix].bind_kv_cache(cache)
        else:
            layers[prefix] = SimpleNamespace(kv_cache=[cache])
    return planes


@pytest.mark.parametrize("boundary", [3, 7, 10, 15, 21, 23, 24, 25, 37, 39])
def test_transfer_retains_history_and_copies_all_live_states(runtime, monkeypatch, boundary):
    partitions = ((0, boundary), (boundary, 40))
    shared = [_shared(), _shared(-77)]
    sender, source_layers = _stage(monkeypatch, partitions, 0, shared[0])
    receiver, mirrors = _stage(monkeypatch, partitions, 1, shared[1])
    source = receiver.cache_handler.incoming.cache_sources[0]
    sources = _bind_source(source_layers, sender.cache_handler, source)
    for plane in sources:
        plane.copy_(torch.arange(plane.numel()).reshape(plane.shape).remainder(97))
    dest = _bind_source(mirrors, receiver.cache_handler, source, -77)
    for source_slots, dest_slots in [([9, 1, -1, 19, 6], [5, 17, -1, 9, 3]), ([22, -1, 10], [6, -1, 17])]:
        n = len(source_slots)
        before = [p.clone() for p in dest]
        runtime.no_compile_layers = source_layers
        runtime.attn_metadata = {
            prefix: _metadata(source_slots) for prefix in sender.cache_handler.cache_prefixes(source)
        }
        payload = sender.send(IntermediateTensors({"hidden_states": torch.zeros(n, 8)}), n)
        snapshots = {name: tensor.clone() for name, tensor in payload.tensors.items()}
        shared[0].candidates.add_(1)
        for name, tensor in snapshots.items():
            torch.testing.assert_close(payload[name], tensor)
        runtime.no_compile_layers = mirrors
        runtime.attn_metadata = {
            prefix: _metadata(dest_slots) for prefix in receiver.cache_handler.cache_prefixes(source)
        }
        receiver.receive(payload, n)
        for source_plane, mirror, previous in zip(sources, dest, before):
            for src, dst in zip(source_slots, dest_slots):
                if src >= 0 and dst >= 0:
                    previous[dst // 4, dst % 4] = source_plane[src // 4, src % 4]
            torch.testing.assert_close(mirror, previous, rtol=0, atol=0)
        for name, buffer in receiver.incoming_states.items():
            torch.testing.assert_close(buffer[:n], payload[name])
    assert not list(receiver.parameters())
    assert not any("state_cache" in name for name in mirrors), "Compressor ring ownership must stay on the producer"


@pytest.mark.parametrize("receives", [False, True], ids=["send", "receive"])
def test_compile_without_metadata_keeps_all_runtime_dependencies(runtime, monkeypatch, receives):
    # CPU torch builds must not discover the mocked torch.npu accelerator.
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    shared = _shared()
    partitions = ((0, 25), (25, 40))
    transport, layers = _stage(monkeypatch, partitions, int(receives), shared)
    source = 20
    planes = _bind_source(layers, transport.cache_handler, source, 12)
    hidden = torch.zeros(5, 8)
    incoming = transport.add_receive_buffers(IntermediateTensors({"hidden_states": hidden}), 5, torch.device("cpu"))
    if receives:
        for spec in transport.incoming_caches:
            incoming[spec.name].fill_(12)
        for name in transport.incoming_states:
            incoming[name].fill_(37)
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    def forward(activations):
        payload = IntermediateTensors({**incoming.tensors, "hidden_states": activations})
        if receives:
            transport.receive(payload, activations.shape[0])
            return activations
        return tuple(transport.send(payload, activations.shape[0]).tensors.values())

    compiled = torch.compile(forward, backend=backend, fullgraph=True)
    compiled(hidden)
    # Change slots after compilation, and again on the very same graph.
    for slots in ([9, 1, -1, 19, 6], [22, 3, -1, 10, 5]):
        runtime.attn_metadata = {prefix: _metadata(slots) for prefix in transport.cache_handler.cache_prefixes(source)}
        runtime.no_compile_layers = layers
        output = compiled(hidden)
        if receives:
            for plane in planes:
                for slot in slots:
                    if slot >= 0:
                        assert (plane[slot // 4, slot % 4] == 12).all()
            assert all((buffer[:5] == 37).all() for buffer in transport.incoming_states.values())
        else:
            for plane, actual in zip(planes, output[1:4]):
                expected = torch.empty_like(actual)
                _reference_gather(plane, torch.tensor(slots), expected)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert len(graphs) == 1
    assert ("pp_dependency_import" if receives else "pp_dependency_export") in str(graphs[0].graph)


@pytest.mark.parametrize("lengths", [(10, 10, 10, 10), (1, 9, 15, 15), (21, 1, 1, 1, 1, 15), tuple([1] * 40)])
def test_all_dependencies_survive_multiple_stage_forwarding(runtime, monkeypatch, lengths):
    partitions = _partitions(lengths)
    stages = [_stage(monkeypatch, partitions, rank, _shared(-77)) for rank in range(len(partitions))]
    for transport, layers in stages:
        for source in set(
            transport.cache_handler.incoming.cache_sources + transport.cache_handler.outgoing.cache_sources
        ):
            _bind_source(layers, transport.cache_handler, source, -77)
    # Two forwards exercise retained state and overwrites, with C2 PAD rows.
    for slots in ([1, -1, 9, -1, 19], [3, -1, 11, -1, 21]):
        payload = None
        for rank, (transport, layers) in enumerate(stages):
            start, end = partitions[rank]
            n = len(slots)
            runtime.no_compile_layers = layers
            runtime.attn_metadata = {
                prefix: _metadata(slots)
                for source in set(
                    transport.cache_handler.incoming.cache_sources + transport.cache_handler.outgoing.cache_sources
                )
                for prefix in transport.cache_handler.cache_prefixes(source)
            }
            if payload is not None:
                transport.receive(payload, n)
                for source in transport.cache_handler.incoming.cache_sources:
                    for plane in layers[transport.cache_handler.cache_prefixes(source)[0]].kv_cache:
                        for slot in slots:
                            if slot >= 0:
                                assert (plane[slot // 4, slot % 4] == source + 1).all()
                for name, buffer in transport.incoming_states.items():
                    torch.testing.assert_close(buffer[:n], payload[name])
            for source in transport.cache_handler.outgoing.cache_sources:
                if start <= source < end:
                    long_prefix, index_prefix = transport.cache_handler.cache_prefixes(source)
                    for plane in [layers[long_prefix].kv_cache[0], *layers[index_prefix].kv_cache[0]]:
                        plane.fill_(source + 1)
            # Stand in for producer layers, only overwriting their own outputs.
            for name, buffer in transport.outgoing_states.items():
                if name not in transport.incoming_states or (
                    name in (TOPK, TOPK_LENGTHS) and any(start <= i < end for i in _config().index_source_layer_ids)
                ):
                    buffer[:n].fill_(rank + 17)
            payload = transport.send(IntermediateTensors({"hidden_states": torch.zeros(n, 8)}), n)


@pytest.mark.parametrize("key", [CANDIDATES, CANDIDATE_LENGTHS, TOPK, TOPK_LENGTHS, "pp_transport_v41_cache_20_long"])
def test_receive_checks_entire_manifest_before_mutating(runtime, monkeypatch, key):
    receiver, _ = _stage(monkeypatch, ((0, 25), (25, 40)), 1, _shared())
    payload = receiver.add_receive_buffers(
        IntermediateTensors({"hidden_states": torch.zeros(3, 8)}), 3, torch.device("cpu")
    )
    payload.tensors[key] = payload[key].to(torch.float64)
    with pytest.raises(ValueError, match="Invalid PP dependency"):
        receiver.receive(payload, 3)


@pytest.mark.parametrize("lengths", [(10, 10, 10, 10), (1, 9, 15, 15), (25, 15), tuple([1] * 40)])
def test_mirrors_share_global_block_ids_without_local_page_aliases(runtime, monkeypatch, lengths):
    if vllm_version_is("0.32.0"):
        from vllm.v1.core.kv_cache_utils import _project_kv_cache_groups_to_worker
    else:
        from vllm.v1.core.kv_cache_planning import _project_kv_cache_groups_to_worker

    from vllm_ascend.models.deepseek_v41.cache_config import get_deepseek_v41_kv_cache_config

    config = make_cache_config(17, head_size=8, index_size=4, draft_layers=3)
    global_specs = {n: s for g in config.kv_cache_groups for n, s in g.kv_cache_spec.kv_cache_specs.items()}
    partitions = _partitions(lengths)
    planner = SimpleNamespace(
        cache_config=SimpleNamespace(num_gpu_blocks_override=None, prefix_cache_retention_interval=None),
        parallel_config=SimpleNamespace(pipeline_parallel_size=len(lengths)),
    )
    draft_names = {f"mtp.{index}.self_attn.swa_cache" for index in range(3)}
    for rank, (start, end) in enumerate(partitions):
        transport, layers = _stage(monkeypatch, partitions, rank, _shared())
        mirrors = {name for name, layer in layers.items() if isinstance(layer, v41_model.DeepseekV41CacheLayer)}
        local = {
            name: spec
            for name, spec in global_specs.items()
            if name in mirrors
            or (".mtp." in "." + name and end == 40)
            or (".layers." in name and start <= int(name.split(".layers.")[1].split(".")[0]) < end)
        }
        groups = _project_kv_cache_groups_to_worker(config.kv_cache_groups, local)
        planned = get_deepseek_v41_kv_cache_config(planner, groups, 10**7)
        _, views = allocate_cache_views(planned)
        assert set(views) == set(local)
        assert draft_names.intersection(views) == (draft_names if end == 40 else set())
        for group in groups:
            pages = []
            for i, name in enumerate(group.layer_names):
                for plane in views[name] if isinstance(views[name], tuple) else (views[name],):
                    plane[1].fill_(i + 1)
                    pages.append((plane[1], i + 1))
            for page, value in pages:
                torch.testing.assert_close(page, torch.full_like(page, value))


@pytest.mark.parametrize("boundary", [15, 23])
@pytest.mark.parametrize("pcp_size", [2, 4])
@pytest.mark.parametrize("global_slots", [[9, 1, -1, 19, 6, 23, -1], [22]])
@pytest.mark.parametrize("sharded", [True, False], ids=["sharded-prefill-or-decode", "replicated-decode"])
def test_pcp_transfer_copies_global_cache_rows_but_keeps_query_states_local(
    runtime, monkeypatch, boundary, pcp_size, global_slots, sharded
):
    partitions = ((0, boundary), (boundary, 40))
    n = (len(global_slots) + pcp_size - 1) // pcp_size if sharded else len(global_slots)
    sender, source_layers = _stage(monkeypatch, partitions, 0, _shared(), pcp_size)
    receiver, mirrors = _stage(monkeypatch, partitions, 1, _shared(-77), pcp_size)
    source = receiver.cache_handler.incoming.cache_sources[0]
    sources = _bind_source(source_layers, sender.cache_handler, source)
    for plane in sources:
        plane.copy_(torch.arange(plane.numel()).reshape(plane.shape).remainder(97))
    dest = _bind_source(mirrors, receiver.cache_handler, source, -77)
    dest_slots = [slot + 1 if slot >= 0 else -1 for slot in global_slots]

    def metadata(slots):
        canonical = _metadata(slots)
        if not sharded:
            return canonical
        # The last peer has no live queries when the global batch has one token.
        local_slots = slots[pcp_size - 1 :: pcp_size]
        local_slots += [-1] * (n - len(local_slots))
        local = _metadata(local_slots)
        local.global_metadata = canonical
        return local

    runtime.no_compile_layers = source_layers
    runtime.attn_metadata = {prefix: metadata(global_slots) for prefix in sender.cache_handler.cache_prefixes(source)}
    payload = sender.send(IntermediateTensors({"hidden_states": torch.zeros(n, 8)}), n)
    for spec in sender.outgoing_caches:
        assert payload[spec.name].shape[0] == n
    for name in sender.outgoing_states:
        assert payload[name].shape[0] == n
    runtime.no_compile_layers = mirrors
    runtime.attn_metadata = {prefix: metadata(dest_slots) for prefix in receiver.cache_handler.cache_prefixes(source)}
    receiver.receive(payload, n)
    for source_plane, mirror in zip(sources, dest):
        expected = torch.full_like(mirror, -77)
        for src, dst in zip(global_slots, dest_slots):
            if src >= 0:
                expected[dst // 4, dst % 4] = source_plane[src // 4, src % 4]
        torch.testing.assert_close(mirror, expected, rtol=0, atol=0)


@pytest.mark.parametrize("env_override,additional", [(False, {}), (None, {"enable_dsa_cp": True})])
def test_pp_model_rejects_explicit_v1_or_runner_fallback_before_allocating_weights(
    monkeypatch, env_override, additional
):
    monkeypatch.setattr("vllm_ascend.mrv2_utils.envs_vllm.VLLM_USE_V2_MODEL_RUNNER", env_override)
    runtime = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=2),
        additional_config=additional,
    )
    runtime.use_v2_model_runner = use_v2_model_runner(runtime)
    # No model/weight config is provided: reject V1 before accessing either.
    with pytest.raises(ValueError, match="requires Model Runner V2"):
        v41_model.DeepseekV41Model(vllm_config=runtime)
