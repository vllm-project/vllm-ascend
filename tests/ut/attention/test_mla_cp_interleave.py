# SPDX-License-Identifier: Apache-2.0

"""Startup coverage for the MLA DCP block-size / interleave contract.

The DCP interleave requirement is `cp_kv_cache_interleave_size` divides the
cache block size. The block size is not final when the platform config is
validated (`update_block_size_for_backend` can still move it), so the MLA DCP
metadata builder resolves the value it actually uses and validates that one.
These tests pin both halves: the resolved-value check in the builder helper,
and the fact that the platform defers MLA targets to it.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_ascend.attention.context_parallel.mla_cp import resolve_dcp_block_layout
from vllm_ascend.attention.context_parallel.topology import CPTopology, validate_dcp_block_interleave
from vllm_ascend.platform import _validate_target_dcp_block_interleave


def _vc(interleave_size: int, dcp_size: int = 2, pcp_size: int = 1) -> SimpleNamespace:
    """Minimal current config: only the fields the topology/layout helpers read."""
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=2,
            prefill_context_parallel_size=pcp_size,
            decode_context_parallel_size=dcp_size,
            cp_kv_cache_interleave_size=interleave_size,
        )
    )


def test_layout_helper_accepts_an_aligned_interleave() -> None:
    # lcm(128, 32) keeps the cache block size: the DCP virtual block already
    # tiles it, which is the layout the documented slot mapping assumes.
    local, virtual, resolved = resolve_dcp_block_layout(_vc(16), 128, 2)
    assert (local, virtual) == (16, 32)
    assert resolved == 128


def test_layout_helper_inflates_only_when_dcp_needs_a_larger_frame() -> None:
    # interleave * dcp = 256 does not tile 128, so the attention block size
    # becomes least_common_multiple(128, 256) == 256 even though 64 divides 128.
    local, virtual, resolved = resolve_dcp_block_layout(_vc(64, dcp_size=4), 128, 4)
    assert (local, virtual) == (64, 256)
    assert resolved == 256


def test_layout_helper_rejects_a_non_divisible_interleave() -> None:
    with pytest.raises(
        ValueError,
        match=r"cp_kv_cache_interleave_size\(96\) must divide the cache block_size\(128\)",
    ) as exc_info:
        resolve_dcp_block_layout(_vc(96), 128, 2)
    # The message must name every quantity needed to find the offending flag.
    assert "dcp_size=2" in str(exc_info.value)


def test_layout_helper_rejects_a_non_positive_interleave() -> None:
    with pytest.raises(ValueError, match=r"cp_kv_cache_interleave_size\(0\)"):
        resolve_dcp_block_layout(_vc(0), 128, 2)


def test_layout_helper_skips_the_check_when_dcp_is_off() -> None:
    # Without decode context parallelism there is no interleaved layout to frame.
    local, virtual, resolved = resolve_dcp_block_layout(_vc(96, dcp_size=1), 128, 1)
    assert (local, virtual, resolved) == (96, 96, 384)


def test_resolved_block_size_accepts_a_config_the_stale_size_rejects() -> None:
    """The backend-resolved value is what decides, not the pre-resolution one."""
    topology = CPTopology(tp_size=2, pcp_size=1, dcp_size=2, interleave_size=96)
    with pytest.raises(ValueError, match=r"block_size\(128\)"):
        validate_dcp_block_interleave(topology, SimpleNamespace(block_size=128))
    # The backend settles on 384, which 96 divides.
    validate_dcp_block_interleave(topology, resolved_block_size=384)


def test_platform_defers_mla_targets_to_the_backend() -> None:
    """MLA keeps starting; the builder raises later on the settled size."""
    config = _vc(96)
    config.model_config = SimpleNamespace(use_mla=True)
    config.cache_config = SimpleNamespace(block_size=128)
    _validate_target_dcp_block_interleave(config)


def test_platform_still_rejects_non_divisible_non_mla_interleave() -> None:
    config = _vc(96)
    config.model_config = SimpleNamespace(use_mla=False)
    config.cache_config = SimpleNamespace(block_size=128)
    with pytest.raises(ValueError, match=r"must divide the cache block_size\(128\)"):
        _validate_target_dcp_block_interleave(config)


def test_resolved_block_size_frames_the_interleave_and_the_virtual_block() -> None:
    """The resolved size the builder uses must be a multiple of both DCP units."""
    for dcp_size in (2, 4, 8):
        for interleave in (1, 16, 32):
            local, virtual, resolved = resolve_dcp_block_layout(_vc(interleave, dcp_size=dcp_size), 128, dcp_size)
            assert resolved % virtual == 0
            assert resolved % local == 0
            assert resolved >= 128


def test_mla_dcp_builder_validates_through_the_layout_helper() -> None:
    """Guard the wiring and the order: validate the cache block, then derive lcm."""
    module_path = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/context_parallel/mla_cp.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    builder = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendMlaDCPMetadataBuilder"
    )
    init = next(node for node in builder.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    layout_call = next(
        node
        for node in ast.walk(init)
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "resolve_dcp_block_layout"
    )
    # The validated size must be the cache block size the builder already resolved.
    block_args = [arg for arg in layout_call.args if isinstance(arg, ast.Attribute) and arg.attr == "block_size"]
    assert block_args, "resolve_dcp_block_layout must receive the builder block size"
    # The helper validates before it derives the least common multiple.
    helper = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "resolve_dcp_block_layout"
    )
    validate_line = min(
        node.lineno
        for node in ast.walk(helper)
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "validate_dcp_block_interleave"
    )
    lcm_line = min(
        node.lineno
        for node in ast.walk(helper)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "gcd"
    )
    assert validate_line < lcm_line, "the interleave check must run before the lcm derivation"
