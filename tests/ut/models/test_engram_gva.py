# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch.distributed as dist

from vllm_ascend.models.deepseek_v41.engram.gva import (
    EngramGvaPool,
    make_engram_gva_local_tables,
    make_engram_gva_manifest,
    make_engram_gva_tables,
    use_engram_gva,
    validate_engram_gva_manifest,
)


@pytest.mark.parametrize(
    "eligible,mode,expected",
    [
        (True, "auto", True),
        (False, "auto", False),
        (True, "gva", True),
        (True, "uva", False),
        (False, "uva", False),
    ],
)
def test_gva_backend_mode(eligible, mode, expected):
    assert use_engram_gva(eligible, mode) is expected


def test_gva_backend_mode_env_override(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_ENGRAM_GVA_MODE", "uva")
    assert not use_engram_gva(True)


@pytest.mark.parametrize(
    "eligible,mode,error",
    [
        (False, "gva", "prerequisites"),
        (True, "other", "one of auto, gva, or uva"),
    ],
)
def test_gva_backend_mode_rejects_invalid_requests(eligible, mode, error):
    with pytest.raises((ValueError, RuntimeError), match=error):
        use_engram_gva(eligible, mode)


def test_gva_layout_balances_rows_and_encodes_each_table_in_slot_order():
    tables, rows_per_slot = make_engram_gva_tables([11, 4], world_size=3)
    assert rows_per_slot == 6
    assert [table.owner_ranges for table in tables] == [
        ((0, 4), (4, 8), (8, 11)),
        ((0, 2), (2, 3), (3, 4)),
    ]
    assert [table.max_rows_per_owner for table in tables] == [4, 2]
    assert [table.block_offset for table in tables] == [0, 4]
    assert [tables[0].encode(row, 0, 11, rows_per_slot) for row in range(11)] == [0, 1, 2, 3, 6, 7, 8, 9, 12, 13, 14]
    assert [tables[1].encode(row, 0, 4, rows_per_slot) for row in range(4)] == [4, 5, 10, 16]
    assert tables[1].encode(4, 0, 4, rows_per_slot) == 0


def test_gva_layout_keeps_all_ranks_in_the_balanced_row_map():
    tables, _ = make_engram_gva_tables([7, 5], world_size=4)
    assert tables[0].owner_ranges == ((0, 2), (2, 4), (4, 6), (6, 7))
    assert tables[1].owner_ranges == ((0, 2), (2, 3), (3, 4), (4, 5))
    for table in tables:
        shard_sizes = [end - start for start, end in table.owner_ranges]
        assert max(shard_sizes) - min(shard_sizes) <= 1
        assert min(shard_sizes) > 0


def test_gva_local_tables_keep_tp_head_sharding_separate_from_dp_owner_rows():
    rows, table_ids, heads = make_engram_gva_local_tables(
        ["layer-1", "layer-14"],
        [(11, 13, 17, 19), (23, 29, 31, 37)],
        tp_rank=1,
        tp_size=2,
    )
    assert rows == (36, 68)
    assert table_ids == ("layer-1/head-cols-2-4", "layer-14/head-cols-2-4")
    assert heads == 2


def test_gva_local_tables_reject_uneven_tp_head_layout():
    with pytest.raises(ValueError, match="divided evenly"):
        make_engram_gva_local_tables(["layer-1"], [(11, 13, 17)], tp_rank=0, tp_size=2)


@pytest.mark.parametrize("rows,world_size", [([], 2), ([0], 2), ([1, -1], 2)])
def test_gva_layout_rejects_empty_or_invalid_tables(rows, world_size):
    with pytest.raises(ValueError, match="positive table rows"):
        make_engram_gva_tables(rows, world_size)


@pytest.mark.parametrize("world_size", [0, 1])
def test_gva_layout_requires_multiple_local_owners(world_size):
    with pytest.raises(ValueError, match="at least two local row owners"):
        make_engram_gva_tables([5], world_size)


def test_gva_layout_rejects_tables_smaller_than_local_owner_count():
    with pytest.raises(ValueError, match="rows for 4 local row owners"):
        make_engram_gva_tables([3], world_size=4)


def _valid_manifest():
    tables, _ = make_engram_gva_tables([11, 4], world_size=3)
    return make_engram_gva_manifest(
        tables,
        generation="generation-a",
        checkpoint_identity="checkpoint-a",
        owner_ids=["node/rank-0", "node/rank-1", "node/rank-2"],
        table_ids=["layer-0", "layer-1"],
        row_bytes=264,
        weight_bytes=256,
    )


def test_gva_manifest_describes_all_balanced_owner_shards_and_packed_planes():
    manifest = _valid_manifest()
    assert manifest["schema"] == 3
    assert [member["role"] for member in manifest["members"]] == ["owner"] * 3
    assert manifest["tables"][0]["format"] == "F8_E4M3/F8_E8M0"
    assert manifest["tables"][0]["shards"][1]["row_start"] == 4
    assert manifest["tables"][0]["shards"][1]["row_end"] == 8
    assert manifest["tables"][0]["shards"][1]["objects"]["weight"]["byte_offset"] == 0
    assert manifest["tables"][1]["shards"][0]["objects"]["weight"]["byte_offset"] == 4 * 264
    assert manifest["tables"][1]["shards"][0]["objects"]["scale"]["intra_row_offset"] == 256
    validate_engram_gva_manifest(manifest, expected_generation="generation-a")


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda manifest: manifest.update(schema=2), "schema"),
        (lambda manifest: manifest["tables"][0]["shards"].pop(), "shard count"),
        (lambda manifest: manifest["tables"][0]["shards"][1].update(row_start=5), "ordered"),
        (lambda manifest: manifest["tables"][0].update(format="F8_E4M3/FP32"), "format"),
        (
            lambda manifest: manifest["tables"][0]["shards"][1]["objects"]["weight"].update(object_id=0),
            "descriptor",
        ),
    ],
)
def test_gva_manifest_rejects_incompatible_layouts(mutation, match):
    manifest = _valid_manifest()
    mutation(manifest)
    with pytest.raises(ValueError, match=match):
        validate_engram_gva_manifest(manifest, expected_generation="generation-a")


def test_gva_manifest_rejects_consumer_members():
    manifest = _valid_manifest()
    manifest["members"][1]["role"] = "consumer"
    with pytest.raises(ValueError, match="every local pool member to own rows"):
        validate_engram_gva_manifest(manifest)


def test_gva_manifest_rejects_nonuniform_row_shards():
    manifest = _valid_manifest()
    shards = manifest["tables"][0]["shards"]
    shards[0]["row_end"] = 5
    shards[1]["row_start"] = 5
    shards[1]["row_end"] = 9
    shards[2]["row_start"] = 9
    with pytest.raises(ValueError, match="balanced"):
        validate_engram_gva_manifest(manifest)


def test_gva_manifest_rejects_generation_mismatch():
    with pytest.raises(ValueError, match="generation"):
        validate_engram_gva_manifest(_valid_manifest(), expected_generation="generation-b")


def test_gva_collective_stage_reports_failures_from_all_ranks(monkeypatch):
    pool = object.__new__(EngramGvaPool)
    pool.world_size = 2
    pool.group = type("Group", (), {"cpu_group": object()})()

    def all_gather(errors, error, group):
        assert group is pool.group.cpu_group
        errors[:] = [None, error]

    monkeypatch.setattr(dist, "all_gather_object", all_gather)
    failures = pool._collective_failures("pool allocation", "RuntimeError: no host memory")
    assert failures == "pool allocation failed (rank 1: RuntimeError: no host memory)"


def test_gva_collective_stage_runs_and_returns_local_result(monkeypatch):
    pool = object.__new__(EngramGvaPool)
    pool.world_size = 2
    pool.group = type("Group", (), {"cpu_group": object()})()

    def all_gather(errors, error, group):
        errors[:] = [None] * len(errors)

    monkeypatch.setattr(dist, "all_gather_object", all_gather)
    result = pool._run_collective_stage("layout registration", lambda: 7)
    assert result == 7
