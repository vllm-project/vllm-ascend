# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from dataclasses import replace

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pool_metadata import (
    LayerwisePoolConnectorMetadata,
    PoolCommitTracker,
    PoolObjectRef,
    PoolObjectSlice,
    PoolSourceProfile,
    PoolTransferMetadata,
    PoolWorkerReport,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.key_codec import PoolKeyIdentity
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.restore_planner import CacheSlice


def transfer():
    identity = PoolKeyIdentity("memcache-v1", "source-model", 2, 2)
    cache = CacheSlice("layer.7", "indexer", "int8-scale-v1", 0, 4, 0, 1, 8, 8, 32)
    item = PoolObjectRef("source-profile", 1, "source-hash", 3, 1, 32, (PoolObjectSlice(cache, 0),))
    return PoolTransferMetadata(
        "transfer",
        "attempt",
        "request-digest",
        0,
        4,
        "retention",
        (PoolSourceProfile("source-profile", "engine-incarnation", identity),),
        (item,),
    )


def test_json_handoff_reconstructs_source_keys_without_local_configuration():
    original = transfer()
    restored = PoolTransferMetadata.from_wire(json.loads(json.dumps(original.to_wire())))
    assert restored == original
    assert restored.sources()[0].key == "source-model@1@pp1@source-hash@3"
    assert restored.sources()[0].cache.layer == "layer.7"
    assert "remote_block_ids" not in restored.to_wire()
    assert "destination" not in json.dumps(restored.to_wire())


def test_source_logical_block_sizes_roundtrip_without_changing_legacy_profile():
    profile = transfer().profiles[0]
    extended = replace(profile, group_block_sizes=(8, 16))
    assert PoolSourceProfile.from_wire(json.loads(json.dumps(extended.to_wire()))) == extended
    assert "group_block_sizes" not in profile.to_wire()
    assert PoolSourceProfile.from_wire(profile.to_wire()) == profile


def test_compact_layouts_preserve_source_shards_components_offsets_and_partial_pages():
    original = transfer()
    base = original.objects[0]
    profiles = (original.profiles[0], replace(original.profiles[0], profile_id="replica"))
    objects = []
    for page in range(64):
        for rank in range(2):
            cache = replace(
                base.slices[0].cache,
                token_start=page * 4,
                token_end=page * 4 + (2 if page == 63 else 4),
                head_start=rank,
                head_end=rank + 1,
            )
            objects.append(
                replace(
                    base,
                    object_id=f"page-{page}",
                    profile_id=profiles[rank].profile_id,
                    head_or_tp_rank=rank,
                    pp_rank=rank,
                    size=64,
                    slices=(PoolObjectSlice(cache, 0), PoolObjectSlice(replace(cache, component="scale"), 32)),
                )
            )
    original = replace(original, profiles=profiles, objects=tuple(objects), token_end=254)
    wire = original.to_wire()
    assert len(wire["objects"]["layouts"]) == 2
    restored = PoolTransferMetadata.from_wire(json.loads(json.dumps(wire)))
    assert restored == original
    assert restored.sources() == original.sources()
    expanded = wire | {"objects": [item.to_wire() for item in original.objects]}
    assert PoolTransferMetadata.from_wire(expanded) == original
    assert len(json.dumps(wire)) < len(json.dumps(expanded)) * 0.3
    # Caller mutations cannot modify an immutable report or a later encoding.
    wire["objects"]["layouts"][0][5][0][0] = "wrong-layer"
    assert original.to_wire()["objects"]["layouts"][0][5][0][0] == "layer.7"


@pytest.mark.parametrize(
    "path,value",
    [
        (("encoding",), "unknown"),
        (("layouts",), {}),
        (("rows",), {}),
        (("layouts", 0), []),
        (("layouts", 0, 5), []),
        (("layouts", 0, 5, 0), []),
        (("layouts", 0, 5, 0, 0), 1),
        (("layouts", 0, 5, 0, 5), True),
        (("rows", 0), []),
        (("rows", 0, 0), -1),
        (("rows", 0, 0), 1),
        (("rows", 0, 0), True),
        (("rows", 0, 1), ""),
        (("rows", 0, 2), []),
        (("rows", 0, 2, 0), []),
        (("rows", 0, 2, 0, 0), -1),
        (("rows", 0, 2, 0, 1), 0),
        (("rows", 0, 2, 0, 2), 1),  # Offset exceeds object bounds.
    ],
)
def test_compact_object_table_rejects_malformed_rows_and_invalid_ranges(path, value):
    wire = transfer().to_wire()
    target = wire["objects"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        PoolTransferMetadata.from_wire(wire)


def test_worker_reports_use_the_same_codec_including_empty_owners_and_legacy_reads():
    for objects in ((), transfer().objects):
        report = PoolWorkerReport("transfer", "attempt", "worker", True, objects)
        wire = json.loads(json.dumps(report.to_wire()))
        assert PoolWorkerReport.from_wire(wire) == report
        assert PoolWorkerReport.from_wire(wire | {"objects": [item.to_wire() for item in objects]}) == report
    wire["objects"]["layouts"].append(wire["objects"]["layouts"][0])
    with pytest.raises(ValueError, match="Unused"):
        PoolWorkerReport.from_wire(wire)


def test_scheduler_to_worker_copy_keeps_pool_metadata_and_detaches_local_blocks():
    blocks = ([1, 2], [3])
    metadata = LayerwisePoolConnectorMetadata()
    metadata.add_load("request", {"pool_transfer": transfer().to_wire()}, blocks)
    blocks[0].append(99)
    assert metadata.local_block_ids["request"] == ([1, 2], [3])
    assert metadata.loads["request"].objects[0].group_id == 1
    with pytest.raises(ValueError, match="twice"):
        metadata.add_load("request", {"pool_transfer": transfer().to_wire()}, blocks)


@pytest.mark.parametrize(
    "patch",
    [
        {"status": "writing"},
        {"version": 2},
        {"version": True},
        {"token_end": -1},
        {"profiles": []},
        {"objects": []},
        {"unknown": 1},
        {"retention_ref": ""},
    ],
)
def test_rejects_incomplete_or_unsupported_handoff(patch):
    with pytest.raises(ValueError):
        PoolTransferMetadata.from_wire(transfer().to_wire() | patch)


def test_object_range_and_profile_coordinates_are_validated():
    original = transfer()
    item = original.objects[0]
    for changed in (replace(item, pp_rank=2), replace(item, group_id=2), replace(item, profile_id="missing")):
        with pytest.raises(ValueError):
            replace(original, objects=(changed,))
    with pytest.raises(ValueError, match="exceeds"):
        replace(item, size=31)
    with pytest.raises(ValueError, match="Duplicate"):
        replace(original, objects=original.objects * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        replace(original, profiles=original.profiles * 2)


def test_all_required_workers_must_commit_and_stale_attempts_are_ignored():
    tracker = PoolCommitTracker("transfer", "attempt", frozenset({"pp0/tp0", "pp1/tp0"}))
    first = PoolWorkerReport("transfer", "attempt", "pp0/tp0", True)
    assert not tracker.record(replace(first, attempt_id="old"))
    assert tracker.record(first)
    assert tracker.record(first)  # Idempotent duplicate completion.
    assert not tracker.committed
    tracker.record(PoolWorkerReport("transfer", "attempt", "pp1/tp0", True))
    assert tracker.committed
    assert tracker.failure is None


def test_worker_failure_prevents_commit_and_conflicting_reports_are_errors():
    tracker = PoolCommitTracker("transfer", "attempt", frozenset({"worker"}))
    failed = PoolWorkerReport("transfer", "attempt", "worker", False, error="native copy failed")
    tracker.record(failed)
    assert not tracker.committed
    assert tracker.failure == "native copy failed"
    with pytest.raises(ValueError, match="Conflicting"):
        tracker.record(PoolWorkerReport("transfer", "attempt", "worker", True))
    with pytest.raises(ValueError, match="Unexpected"):
        tracker.record(replace(failed, worker_id="other"))
