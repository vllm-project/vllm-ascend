# SPDX-License-Identifier: Apache-2.0

import ast
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

PATCH_PATH = Path(__file__).resolve().parents[1] / "patch_gdn_v023.py"
RUNNER_PATCH_PATH = Path(__file__).resolve().parents[1] / "patch_runner.py"


def _load_prefill_routers():
    tree = ast.parse(PATCH_PATH.read_text(encoding="utf-8"))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {"_dcut_gdn_has_prefill", "_dcut_gdn_use_native_core"}
    ]
    module = ast.Module(body=functions, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(PATCH_PATH), "exec"), namespace)
    return (
        namespace["_dcut_gdn_has_prefill"],
        namespace["_dcut_gdn_use_native_core"],
    )


def test_prefill_batch_routes_to_native_gdn_core() -> None:
    has_prefill, use_native = _load_prefill_routers()
    context = SimpleNamespace(
        attn_metadata={
            "layers.0.mixer": SimpleNamespace(num_prefills=1),
            "layers.1.mixer": SimpleNamespace(num_prefills=1),
        }
    )

    assert has_prefill(context)
    assert has_prefill(context, "layers.0.mixer")
    assert use_native(context, "layers.0.mixer")


def test_pure_spec_batch_keeps_dcut_gdn_core() -> None:
    has_prefill, use_native = _load_prefill_routers()
    context = SimpleNamespace(
        attn_metadata={
            "layers.0.mixer": SimpleNamespace(num_prefills=0),
        }
    )

    assert not has_prefill(context)
    assert not has_prefill(context, "layers.0.mixer")
    assert not use_native(context, "layers.0.mixer")


def test_uniform_full_graph_keeps_stock_gdn_core() -> None:
    _, use_native = _load_prefill_routers()
    context = SimpleNamespace(
        cudagraph_runtime_mode=SimpleNamespace(name="FULL"),
        batch_descriptor=SimpleNamespace(uniform=True),
        _dcut_gdn_native_batch=False,
        attn_metadata={
            "layers.0.mixer": SimpleNamespace(num_prefills=0),
        },
    )

    assert use_native(context, "layers.0.mixer")
    context.batch_descriptor.uniform = False
    assert not use_native(context, "layers.0.mixer")


def test_scheduler_non_prefill_overrides_synthetic_gdn_prefill() -> None:
    has_prefill, use_native = _load_prefill_routers()
    context = SimpleNamespace(
        _dcut_gdn_native_batch=False,
        attn_metadata={
            # The native GDN builder reports ordinary decode rows here when
            # speculative and non-speculative decode coexist.
            "layers.0.mixer": SimpleNamespace(num_prefills=3),
        },
    )

    assert has_prefill(context, "layers.0.mixer")
    assert not use_native(context, "layers.0.mixer")


def test_reused_context_can_return_to_pure_spec_route() -> None:
    has_prefill, use_native = _load_prefill_routers()
    context = SimpleNamespace(
        _dcut_gdn_native_batch=True,
        attn_metadata={
            "layers.0.mixer": SimpleNamespace(num_prefills=0),
        },
    )

    assert not has_prefill(context)
    assert use_native(context, "layers.0.mixer")

    context._dcut_gdn_native_batch = False
    assert not use_native(context, "layers.0.mixer")


def test_prefill_metadata_uses_native_builder() -> None:
    tree = ast.parse(PATCH_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_patch_gdn_spec_metadata_builder"
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"replace": replace}
    exec(compile(module, str(PATCH_PATH), "exec"), namespace)

    calls = []

    def native_builder(self, attn_metadata):
        calls.append(attn_metadata)
        return "native"

    def native_build(
        self,
        common_prefix_len,
        common_attn_metadata,
        num_accepted_tokens=None,
        num_decode_draft_tokens_cpu=None,
        fast_build=False,
    ):
        return common_attn_metadata

    class MetadataBuilder:
        build = native_build
        _attach_spec_decode_metadata = native_builder

    fake_module = SimpleNamespace(
        AscendGDNAttentionMetadataBuilder=MetadataBuilder
    )
    namespace["_patch_gdn_spec_metadata_builder"](fake_module)
    metadata = SimpleNamespace(num_prefills=1)

    builder = MetadataBuilder()
    assert builder._attach_spec_decode_metadata(metadata) == "native"

    # A recompute handoff is decode-only from the metadata builder's point of
    # view, but must still keep the native metadata paired with the native core.
    handoff_metadata = SimpleNamespace(num_prefills=0)
    builder._dcut_force_native_gdn_metadata = True
    assert builder._attach_spec_decode_metadata(handoff_metadata) == "native"
    assert calls == [metadata, handoff_metadata]


def test_fia_dummy_request_is_removed_from_gdn_metadata() -> None:
    tree = ast.parse(PATCH_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_patch_gdn_spec_metadata_builder"
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)

    calls = []

    def native_build(
        self,
        common_prefix_len,
        common_attn_metadata,
        num_accepted_tokens=None,
        num_decode_draft_tokens_cpu=None,
        fast_build=False,
    ):
        calls.append(
            (
                common_attn_metadata.num_reqs,
                len(common_attn_metadata.query_start_loc_cpu),
                len(num_accepted_tokens),
                len(num_decode_draft_tokens_cpu),
            )
        )
        return common_attn_metadata

    def native_attach(self, attn_metadata):
        return attn_metadata

    class MetadataBuilder:
        vllm_config = SimpleNamespace(
            compilation_config=SimpleNamespace(
                cudagraph_mode=SimpleNamespace(name="FULL_DECODE_ONLY")
            )
        )
        build = native_build
        _attach_spec_decode_metadata = native_attach

    namespace = {"replace": replace}
    exec(compile(module, str(PATCH_PATH), "exec"), namespace)
    namespace["_patch_gdn_spec_metadata_builder"](
        SimpleNamespace(AscendGDNAttentionMetadataBuilder=MetadataBuilder)
    )

    @dataclass
    class CommonMetadata:
        query_start_loc_cpu: list[int]
        num_reqs: int
        num_actual_tokens: int

        def unpadded(self, num_actual_tokens, num_actual_reqs):
            return replace(
                self,
                query_start_loc_cpu=self.query_start_loc_cpu[
                    : num_actual_reqs + 1
                ],
                num_reqs=num_actual_reqs,
                num_actual_tokens=num_actual_tokens,
            )

    common = CommonMetadata(
        query_start_loc_cpu=[0, 2, 5, 5],
        num_reqs=3,
        num_actual_tokens=5,
    )
    accepted = [1, 1, 1]
    draft_lens = [1, 2, -1]
    result = MetadataBuilder().build(
        0, common, accepted, draft_lens, False
    )

    assert result.num_reqs == 2
    assert calls == [(2, 3, 2, 2)]

    # The same wrapper is inert under PIECEWISE: no FIA-specific request
    # unpadding or per-request tensor slicing is allowed to leak across modes.
    MetadataBuilder.vllm_config.compilation_config.cudagraph_mode.name = (
        "PIECEWISE"
    )
    piecewise_result = MetadataBuilder().build(
        0, common, accepted, draft_lens, False
    )
    assert piecewise_result.num_reqs == 3
    assert calls[-1] == (3, 4, 3, 3)

    # FULL_DECODE_ONLY normally removes the FIA dummy request. Recompute
    # handoff routing must bypass that D-Cut wrapper as well.
    MetadataBuilder.vllm_config.compilation_config.cudagraph_mode.name = (
        "FULL_DECODE_ONLY"
    )
    native_builder = MetadataBuilder()
    native_builder._dcut_force_native_gdn_metadata = True
    native_result = native_builder.build(
        0, common, accepted, draft_lens, False
    )
    assert native_result.num_reqs == 3
    assert calls[-1] == (3, 4, 3, 3)


def _load_native_recompute_handoff_router():
    tree = ast.parse(RUNNER_PATCH_PATH.read_text(encoding="utf-8"))
    names = {
        "_dcut_execute_with_gdn_prefill_route",
        "_dcut_execute_native_recompute_handoff",
    }
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    module = ast.Module(body=functions, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(RUNNER_PATCH_PATH), "exec"), namespace)
    return namespace["_dcut_execute_native_recompute_handoff"]


def test_recompute_handoff_pairs_native_core_and_metadata() -> None:
    def patched_attach(self, metadata):
        return metadata

    patched_attach._dcut_patched = True

    class Builder:
        _attach_spec_decode_metadata = patched_attach

    builder = Builder()
    group = SimpleNamespace(get_metadata_builder=lambda _ubid=0: builder)
    runner = SimpleNamespace(attn_groups=[[group]])
    route = _load_native_recompute_handoff_router()

    def execute(runner_arg, scheduler_output, intermediate_tensors):
        assert runner_arg is runner
        assert scheduler_output == "scheduler"
        assert intermediate_tensors == "intermediate"
        assert runner._dcut_recompute_handoff_active is True
        assert runner._dcut_gdn_scheduler_has_prefill is True
        assert builder._dcut_force_native_gdn_metadata is True
        return "native-result"

    assert route(runner, execute, "scheduler", "intermediate") == (
        "native-result"
    )
    assert not hasattr(runner, "_dcut_recompute_handoff_active")
    assert not hasattr(runner, "_dcut_gdn_scheduler_has_prefill")
    assert not hasattr(builder, "_dcut_force_native_gdn_metadata")


def test_recompute_handoff_restores_route_flags_after_failure() -> None:
    def patched_attach(self, metadata):
        return metadata

    patched_attach._dcut_patched = True

    class Builder:
        _attach_spec_decode_metadata = patched_attach

    builder = Builder()
    builder._dcut_force_native_gdn_metadata = "previous-builder"
    group = SimpleNamespace(get_metadata_builder=lambda _ubid=0: builder)
    runner = SimpleNamespace(
        attn_groups=[[group]],
        _dcut_recompute_handoff_active="previous-runner",
        _dcut_gdn_scheduler_has_prefill="previous-core-route",
    )
    route = _load_native_recompute_handoff_router()

    def fail(*_args):
        raise RuntimeError("expected")

    try:
        route(runner, fail, None, None)
    except RuntimeError as exc:
        assert str(exc) == "expected"
    else:
        raise AssertionError("expected recompute handoff execution to fail")

    assert runner._dcut_recompute_handoff_active == "previous-runner"
    assert runner._dcut_gdn_scheduler_has_prefill == "previous-core-route"
    assert builder._dcut_force_native_gdn_metadata == "previous-builder"
