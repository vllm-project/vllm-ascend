"""Verify concrete Layerwise formulas without hiding request-time branches."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.projection import (
    GVALayerwiseProjection,
    GVALayerwiseProjectionBinder,
    KeyRangeLayerwiseProjection,
    KeyRangeLayerwiseProjectionBinder,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.projection.layerwise.gva import (
    gva_layer_ranges,
    gva_local_keys,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.projection.layerwise.key_range import (
    key_range_layer_ranges,
    key_range_local_keys,
)

from .helpers import make_topology


def _registration(topology, *, block_length: int = 32):
    bases = {}
    lengths = {}
    strides = {}
    layer_offsets = {}
    for group in topology.transfer_groups:
        group_bases = []
        group_lengths = []
        group_strides = []
        group_offsets = [0]
        for layer in group.layers:
            group_bases.append(1000 + group.group_id * 10000 + layer.physical_layer_id * 1000)
            group_lengths.append(block_length)
            group_strides.append(64)
            group_offsets.append(len(group_bases))
        bases[group.group_id] = group_bases
        lengths[group.group_id] = group_lengths
        strides[group.group_id] = group_strides
        layer_offsets[group.group_id] = group_offsets
    return bases, lengths, strides, layer_offsets


def test_layerwise_binders_reject_unsupported_static_compositions() -> None:
    for topology, message in (
        (make_topology(tp_mismatch=True), "TP mismatch"),
        (make_topology(consumer_pipeline_partitions=(1, 1)), "consumer pipeline"),
    ):
        for binder_type in (KeyRangeLayerwiseProjectionBinder, GVALayerwiseProjectionBinder):
            with pytest.raises(ValueError, match=message):
                binder_type(topology, 64, lambda *_args: "key")


def test_key_range_projection_bind_terminal_layer_formulas() -> None:
    topology = make_topology()
    projection = KeyRangeLayerwiseProjectionBinder(
        topology,
        64,
        lambda group, value, head, stage: f"g{group}:p{stage}:h{head}:{value}",
    ).bind(*_registration(topology))
    assert isinstance(projection, KeyRangeLayerwiseProjection)
    assert not hasattr(projection, "load_rows")
    assert not hasattr(projection, "store_candidate_rows")

    group = projection.groups[0]
    block_ids = np.asarray([1, 3], dtype=np.uint64)
    counts = np.asarray([4, 4], dtype=np.uint64)
    first = key_range_layer_ranges(group, block_ids, counts, layer_id=0)
    second = key_range_layer_ranges(group, block_ids, counts, layer_id=1)

    assert first[0].tolist() == [[1064], [1192]]
    assert first[2].tolist() == [[0], [0]]
    assert second[0].tolist() == [[2064], [2192]]
    assert second[2].tolist() == [[32], [32]]
    assert first[1].tolist() == second[1].tolist() == [[32], [32]]
    assert group.object_size == 64
    assert key_range_local_keys(group, ("a", "b")) == (("g0:p0:h0:a", "g0:p0:h0:b"),)


def test_gva_projection_bind_global_layout_and_terminal_copy_formula() -> None:
    topology = replace(make_topology(), tp_rank=1, tp_size=2, put_step=2)

    def full_key(group: int, value: str, head: int, stage: int) -> str:
        return f"g{group}:p{stage}:h{head}:{value}"

    binder = GVALayerwiseProjectionBinder(topology, 64, full_key)
    with pytest.raises(ValueError, match="global object sizes"):
        binder.bind(*_registration(topology))

    parallel = replace(make_topology(group_ids=(0, 1)), pp_size=2)
    parallel_binder = GVALayerwiseProjectionBinder(parallel, 64, full_key)
    with pytest.raises(ValueError, match="group 1.*global object offset"):
        parallel_binder.bind(
            *_registration(parallel),
            object_sizes={0: 128, 1: 128},
            object_offsets={0: 0},
        )

    projection = binder.bind(
        *_registration(topology),
        object_sizes={0: 128},
        object_offsets={0: 16},
    )
    assert isinstance(projection, GVALayerwiseProjection)
    assert not hasattr(projection, "load_rows")
    assert not hasattr(projection, "store_candidate_rows")

    group = projection.groups[0]
    remote, local, sizes = gva_layer_ranges(
        group,
        np.asarray([1, 3], dtype=np.uint64),
        np.asarray([2, 3], dtype=np.uint64),
        np.asarray([20_000, 30_000], dtype=np.uint64),
        layer_id=1,
    )
    assert remote.tolist() == [20_048, 30_048]
    assert local.tolist() == [2064, 2192]
    assert sizes.tolist() == [16, 24]
    assert group.object_size == 128
    assert gva_local_keys(group, ("a",)) == (("g0:p0:h0:a",),)


def test_layerwise_projection_has_no_generic_functional_transport_layer() -> None:
    projection_dir = (
        Path(__file__).parents[5] / "vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/v1/projection"
    )
    assert not (projection_dir / "identity.py").exists()
    assert not (projection_dir / "memory.py").exists()
    assert not (projection_dir / "layerwise/common.py").exists()

    arguments = (projection_dir.parent / "worker/io/arguments.py").read_text()
    assert "RangeBatch" not in arguments
    assert "projection.memory" not in arguments


def test_execution_ownership_has_no_legacy_runtime_package() -> None:
    v1_dir = Path(__file__).parents[5] / "vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/v1"

    assert not (v1_dir / "runtime").exists()
    assert (v1_dir / "projection/bulk/rows.py").is_file()
    assert (v1_dir / "worker/io/io.py").is_file()
    runtime_imports = [path for path in v1_dir.rglob("*.py") if ".runtime" in path.read_text()]
    assert runtime_imports == []


def test_worker_implementation_is_grouped_by_execution_role() -> None:
    worker_dir = Path(__file__).parents[5] / "vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/v1/worker"

    assert (worker_dir / "bulk/worker.py").is_file()
    assert (worker_dir / "layerwise/worker.py").is_file()
    assert (worker_dir / "layerwise/gva.py").is_file()
    assert (worker_dir / "layerwise/key_range.py").is_file()
    assert (worker_dir / "transfer/batch.py").is_file()
    assert (worker_dir / "transfer/evidence.py").is_file()
    assert (worker_dir / "transfer/result.py").is_file()
    assert (worker_dir / "transfer/state.py").is_file()

    legacy_root_modules = (
        "batch.py",
        "bulk.py",
        "evidence.py",
        "gva.py",
        "key_range.py",
        "layerwise.py",
        "result.py",
        "state.py",
    )
    assert all(not (worker_dir / module).exists() for module in legacy_root_modules)
