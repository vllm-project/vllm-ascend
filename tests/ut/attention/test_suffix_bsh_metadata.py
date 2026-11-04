# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Real-tensor CPU regressions for native suffix BSH metadata.

For a standalone run without vLLM or the repository-wide UT fixtures:
    pytest --confcutdir=tests/ut/attention tests/ut/attention/test_suffix_bsh_metadata.py
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture(scope="module")
def bsh_module():
    # Load the actual module without importing the hardware-dependent package.
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/suffix_bsh_attention.py"
    module_name = "_test_native_suffix_bsh_attention"
    spec = importlib.util.spec_from_file_location(module_name, source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    previous_module = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module


def _table(rows):
    return torch.tensor(rows, dtype=torch.int32, device="cpu")


def _metadata(**changes):
    fields = {
        "num_actual_tokens": 10,
        "actual_seq_lengths_q": [5, 10],
        "seq_lens_list": [7, 129],
        "block_tables": _table([[11, 12], [21, 22]]),
    }
    fields.update(changes)
    return SimpleNamespace(**fields)


def _assert_table(actual, rows):
    assert actual.device.type == "cpu"
    assert actual.dtype == torch.int32
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, _table(rows), rtol=0, atol=0)


@pytest.mark.parametrize("sequence_type", [list, tuple], ids=["list", "tuple"])
def test_geometry_returns_only_real_lengths_and_original_table(bsh_module, sequence_type):
    metadata = _metadata(
        actual_seq_lengths_q=sequence_type([5, 10, 25]),
        seq_lens_list=sequence_type([7, 129, 999]),
        block_tables=_table([[11, 12], [21, 22], [91, 92]]),
    )
    rows, lengths, table = bsh_module._metadata_geometry(metadata, 5, 5, 256)

    assert rows == 2
    assert lengths == [7, 129]
    assert lengths is not metadata.seq_lens_list
    assert table is metadata.block_tables
    _assert_table(table, [[11, 12], [21, 22], [91, 92]])
    assert metadata.seq_lens_list == sequence_type([7, 129, 999])
    assert metadata.actual_seq_lengths_q == sequence_type([5, 10, 25])


@pytest.mark.parametrize("extra_table_row", [False, True], ids=["exact-table", "extra-row"])
def test_full_capacity_preserves_lengths_and_uses_only_real_table_rows(
    bsh_module, extra_table_row
):
    rows = [[11, 12], [21, 22]]
    if extra_table_row:
        rows.append([91, 92])
    metadata = _metadata(block_tables=_table(rows))
    original_table = metadata.block_tables.clone()
    original_lengths = metadata.seq_lens_list
    original_boundaries = metadata.actual_seq_lengths_q

    result = bsh_module.prepare_metadata(metadata, 2, 5, max_model_len=256)
    lengths, table = result

    assert lengths == [7, 129]
    assert lengths is not original_lengths
    _assert_table(table, [[11, 12], [21, 22]])
    assert table.data_ptr() == metadata.block_tables.data_ptr()
    assert metadata.seq_lens_list is original_lengths
    assert metadata.seq_lens_list == [7, 129]
    assert metadata.actual_seq_lengths_q is original_boundaries
    assert metadata.actual_seq_lengths_q == [5, 10]
    torch.testing.assert_close(metadata.block_tables, original_table, rtol=0, atol=0)
    assert bsh_module.prepare_metadata(metadata, 2, 5, max_model_len=256) is result


@pytest.mark.parametrize(
    "boundaries,lengths,rows",
    [
        pytest.param(
            [5, 10], [7, 129], [[11, 12], [21, 22]], id="unpadded-input"
        ),
        pytest.param(
            [5, 10, 25], [7, 129, 999],
            [[11, 12], [21, 22], [91, 92]], id="combined-synthetic-tail"
        ),
        pytest.param(
            [5, 10, 15, 20, 25], [7, 129, 0, -99, True],
            [[11, 12], [21, 22], [91, 92], [81, 82], [71, 72]],
            id="distributed-padding"
        ),
        pytest.param(
            [5, 10, 10, 10, 25], [7, 129, 0, -99, True],
            [[11, 12], [21, 22], [91, 92], [81, 82], [71, 72]],
            id="empty-padded-segments"
        ),
    ],
)
def test_partial_capacity_gives_each_dummy_its_own_length_and_zero_row(
    bsh_module, boundaries, lengths, rows
):
    metadata = _metadata(
        actual_seq_lengths_q=boundaries.copy(),
        seq_lens_list=lengths.copy(),
        block_tables=_table(rows),
    )
    original_lengths = metadata.seq_lens_list
    original_boundaries = metadata.actual_seq_lengths_q
    original_table = metadata.block_tables
    table_before = original_table.clone()

    result = bsh_module.prepare_metadata(metadata, 5, 5, max_model_len=256)
    output_lengths, output_table = result

    assert output_lengths == [7, 129, 5, 5, 5]
    _assert_table(output_table, [[11, 12], [21, 22], [0, 0], [0, 0], [0, 0]])
    assert output_table.data_ptr() != original_table.data_ptr()
    assert output_lengths is not original_lengths
    assert metadata.seq_lens_list is original_lengths
    assert metadata.seq_lens_list == lengths
    assert metadata.actual_seq_lengths_q is original_boundaries
    assert metadata.actual_seq_lengths_q == boundaries
    assert metadata.block_tables is original_table
    torch.testing.assert_close(metadata.block_tables, table_before, rtol=0, atol=0)
    assert bsh_module.prepare_metadata(metadata, 5, 5, max_model_len=256) is result


@pytest.mark.parametrize(
    "length,max_model_len,columns",
    [
        pytest.param(5, 5, 1, id="minimum-query-and-model-length"),
        pytest.param(127, 256, 2, id="before-block-boundary"),
        pytest.param(128, 256, 2, id="exact-block-boundary"),
        pytest.param(129, 256, 2, id="after-block-boundary"),
        pytest.param(200, 200, 2, id="model-limited-maximum"),
        pytest.param(256, 512, 2, id="table-limited-maximum"),
    ],
)
def test_valid_kv_length_boundaries(bsh_module, length, max_model_len, columns):
    rows = [[11] if columns == 1 else [11, 12]]
    metadata = _metadata(
        num_actual_tokens=5,
        actual_seq_lengths_q=[5],
        seq_lens_list=[length],
        block_tables=_table(rows),
    )
    real_rows, real_lengths, original_table = bsh_module._metadata_geometry(
        metadata, 1, 5, max_model_len
    )
    assert real_rows == 1
    assert real_lengths == [length]
    assert original_table is metadata.block_tables
    output_lengths, output_table = bsh_module.prepare_metadata(
        metadata, 1, 5, max_model_len=max_model_len
    )
    assert output_lengths == [length]
    _assert_table(output_table, rows)


@pytest.mark.parametrize("layer_name", ["decoder.0", None], ids=["named-layer", "anonymous-layer"])
def test_provider_fresh_snapshots_never_reuse_previous_lengths_or_tables(
    bsh_module, layer_name
):
    provider = bsh_module.BSHParamProvider(layer_name, 4, 5, 256, 2)
    snapshots = [
        (_metadata(), [7, 129, 5, 5], [[11, 12], [21, 22], [0, 0], [0, 0]]),
        (
            _metadata(seq_lens_list=[9, 197], block_tables=_table([[31, 32], [41, 42]])),
            [9, 197, 5, 5], [[31, 32], [41, 42], [0, 0], [0, 0]],
        ),
        (
            _metadata(
                num_actual_tokens=5, actual_seq_lengths_q=[5, 20],
                seq_lens_list=[12, 999], block_tables=_table([[51, 52], [91, 92]]),
            ),
            [12, 5, 5, 5], [[51, 52], [0, 0], [0, 0], [0, 0]],
        ),
        (
            _metadata(
                num_actual_tokens=20, actual_seq_lengths_q=[5, 10, 15, 20],
                seq_lens_list=[5, 64, 128, 200],
                block_tables=_table([[61, 62], [71, 72], [81, 82], [91, 92]]),
            ),
            [5, 64, 128, 200], [[61, 62], [71, 72], [81, 82], [91, 92]],
        ),
    ]
    retained = []
    for metadata, expected_lengths, expected_rows in snapshots:
        result = provider.resolve({layer_name: metadata})
        assert set(result) == {"actual_seq_lengths", "actual_seq_lengths_kv", "block_table"}
        assert result["actual_seq_lengths"] is None
        lengths, table = result["actual_seq_lengths_kv"], result["block_table"]
        repeated = provider.resolve({layer_name: metadata})
        assert repeated["actual_seq_lengths_kv"] is lengths
        assert repeated["block_table"] is table
        assert lengths == expected_lengths
        _assert_table(table, expected_rows)
        for old_lengths, old_table, old_expected_lengths, old_expected_rows in retained:
            assert lengths is not old_lengths
            assert table.data_ptr() != old_table.data_ptr()
            assert old_lengths == old_expected_lengths
            _assert_table(old_table, old_expected_rows)
        retained.append((lengths, table, expected_lengths, expected_rows))


@pytest.mark.parametrize(
    "changes",
    [
        pytest.param({"capacity": 0}, id="zero-capacity"),
        pytest.param({"capacity": -1}, id="negative-capacity"),
        pytest.param({"capacity": 1}, id="capacity-overflow"),
        pytest.param({"capacity": True}, id="boolean-capacity"),
        pytest.param({"capacity": 4.0}, id="float-capacity"),
        pytest.param({"capacity": "4"}, id="string-capacity"),
        pytest.param({"width": 4}, id="short-width"),
        pytest.param({"width": 6}, id="long-width"),
        pytest.param({"width": 0}, id="zero-width"),
        pytest.param({"width": True}, id="boolean-width"),
        pytest.param({"width": 5.0}, id="float-width"),
        pytest.param({"max_model_len": 4}, id="short-model"),
        pytest.param({"max_model_len": 0}, id="zero-model"),
        pytest.param({"max_model_len": -1}, id="negative-model"),
        pytest.param({"max_model_len": True}, id="boolean-model"),
        pytest.param({"max_model_len": 256.0}, id="float-model"),
        pytest.param({"max_model_len": None}, id="missing-model"),
    ],
)
def test_invalid_scalar_geometry_is_rejected(bsh_module, changes):
    options = {"capacity": 4, "width": 5, "max_model_len": 256}
    options.update(changes)
    with pytest.raises(ValueError):
        bsh_module.prepare_metadata(_metadata(), **options)


@pytest.mark.parametrize(
    "field,value",
    [
        pytest.param("num_actual_tokens", 0, id="zero-token-count"),
        pytest.param("num_actual_tokens", -5, id="negative-token-count"),
        pytest.param("num_actual_tokens", 11, id="incomplete-query-width"),
        pytest.param("num_actual_tokens", 10.0, id="float-token-count"),
        pytest.param("num_actual_tokens", True, id="boolean-token-count"),
        pytest.param("actual_seq_lengths_q", None, id="missing-boundaries"),
        pytest.param("actual_seq_lengths_q", "5,10", id="string-boundaries"),
        pytest.param("actual_seq_lengths_q", [], id="empty-boundaries"),
        pytest.param("actual_seq_lengths_q", [5], id="missing-real-boundary"),
        pytest.param("actual_seq_lengths_q", [5, 10, 15, 20, 25], id="too-many-boundaries"),
        pytest.param("actual_seq_lengths_q", [4, 10], id="wrong-first-boundary"),
        pytest.param("actual_seq_lengths_q", [5, 11], id="wrong-second-boundary"),
        pytest.param("actual_seq_lengths_q", [5.0, 10], id="float-real-boundary"),
        pytest.param("actual_seq_lengths_q", [True, 10], id="boolean-real-boundary"),
        pytest.param("actual_seq_lengths_q", [5, 10, 16, 15], id="decreasing-padding"),
        pytest.param("actual_seq_lengths_q", [5, 10, 9, 20], id="padding-before-real-tail"),
        pytest.param("actual_seq_lengths_q", [5, 10, 21], id="padding-exceeds-capacity"),
        pytest.param("actual_seq_lengths_q", [5, 10, 15], id="padding-ends-short"),
        pytest.param("actual_seq_lengths_q", [5, 10, 20.0], id="float-padded-boundary"),
        pytest.param("actual_seq_lengths_q", [5, 10, False, 20], id="boolean-padded-boundary"),
        pytest.param("seq_lens_list", None, id="missing-length-mirror"),
        pytest.param("seq_lens_list", "7,129", id="string-length-mirror"),
        pytest.param("seq_lens_list", [7], id="missing-real-kv-length"),
        pytest.param("seq_lens_list", [0, 129], id="zero-kv-length"),
        pytest.param("seq_lens_list", [-1, 129], id="negative-kv-length"),
        pytest.param("seq_lens_list", [4, 129], id="kv-shorter-than-query"),
        pytest.param("seq_lens_list", [7, 257], id="kv-over-table-and-model-limit"),
        pytest.param("seq_lens_list", [7, 129.0], id="float-real-kv-length"),
        pytest.param("seq_lens_list", [True, 129], id="boolean-real-kv-length"),
        pytest.param("seq_lens_list", [7, "129"], id="string-real-kv-length"),
    ],
)
def test_malformed_metadata_is_rejected(bsh_module, field, value):
    with pytest.raises(ValueError):
        bsh_module.prepare_metadata(_metadata(**{field: value}), 4, 5, max_model_len=256)


@pytest.mark.parametrize("field", ["actual_seq_lengths_q", "seq_lens_list"])
def test_cpu_tensor_does_not_replace_required_python_sequence(bsh_module, field):
    values = [5, 10] if field == "actual_seq_lengths_q" else [7, 129]
    metadata = _metadata(**{field: torch.tensor(values, dtype=torch.int32, device="cpu")})
    with pytest.raises(ValueError):
        bsh_module.prepare_metadata(metadata, 4, 5, max_model_len=256)


@pytest.mark.parametrize(
    "table_factory",
    [
        pytest.param(lambda: [[11, 12], [21, 22]], id="not-a-tensor"),
        pytest.param(lambda: torch.tensor(11, dtype=torch.int32, device="cpu"), id="scalar-table"),
        pytest.param(lambda: _table([11, 12]), id="one-dimensional-table"),
        pytest.param(lambda: _table([[[11, 12]], [[21, 22]]]), id="three-dimensional-table"),
        pytest.param(lambda: torch.empty((0, 2), dtype=torch.int32, device="cpu"), id="zero-rows"),
        pytest.param(lambda: _table([[11, 12]]), id="missing-real-row"),
        pytest.param(lambda: torch.empty((2, 0), dtype=torch.int32, device="cpu"), id="zero-columns"),
        pytest.param(lambda: _table([[11, 12], [21, 22]]).to(torch.int64), id="int64-table"),
        pytest.param(lambda: _table([[11, 12], [21, 22]]).to(torch.float32), id="float-table"),
        pytest.param(lambda: _table([[11, 12], [21, 22]]).to(torch.bool), id="boolean-table"),
        pytest.param(
            lambda: torch.arange(8, dtype=torch.int32, device="cpu").reshape(2, 4)[:, ::2],
            id="noncontiguous-table"
        ),
    ],
)
def test_malformed_block_tables_are_rejected(bsh_module, table_factory):
    metadata = _metadata(block_tables=table_factory())
    with pytest.raises(ValueError):
        bsh_module.prepare_metadata(metadata, 4, 5, max_model_len=256)


@pytest.mark.parametrize(
    "length,max_model_len",
    [
        pytest.param(201, 200, id="model-limit-with-spare-table-capacity"),
        pytest.param(257, 512, id="table-limit-with-spare-model-capacity"),
    ],
)
def test_model_and_block_table_length_limits_are_independent(
    bsh_module, length, max_model_len
):
    metadata = _metadata(seq_lens_list=[7, length])
    with pytest.raises(ValueError):
        bsh_module.prepare_metadata(metadata, 4, 5, max_model_len=max_model_len)


@pytest.mark.parametrize(
    "rows",
    [
        pytest.param([[11], [21]], id="fewer-captured-columns"),
        pytest.param([[11, 12, 13], [21, 22, 23]], id="more-captured-columns"),
        pytest.param([11, 12], id="lost-table-dimension"),
        pytest.param([[[11, 12]], [[21, 22]]], id="extra-table-dimension"),
    ],
)
def test_provider_rejects_column_geometry_drift_on_fresh_metadata(bsh_module, rows):
    provider = bsh_module.BSHParamProvider("decoder.0", 4, 5, 256, 2)
    first = provider.resolve({"decoder.0": _metadata()})
    assert first["actual_seq_lengths_kv"] == [7, 129, 5, 5]
    _assert_table(first["block_table"], [[11, 12], [21, 22], [0, 0], [0, 0]])

    changed = _metadata(block_tables=_table(rows))
    with pytest.raises(ValueError):
        provider.resolve({"decoder.0": changed})
    assert first["actual_seq_lengths_kv"] == [7, 129, 5, 5]
    _assert_table(first["block_table"], [[11, 12], [21, 22], [0, 0], [0, 0]])


@pytest.mark.parametrize(
    "changes",
    [
        pytest.param({"capacity": True}, id="bool-int-capacity-key-collision"),
        pytest.param({"capacity": 1.0}, id="float-int-capacity-key-collision"),
        pytest.param({"width": 5.0}, id="float-int-width-key-collision"),
        pytest.param({"max_model_len": 5.0}, id="float-int-model-key-collision"),
    ],
)
def test_cached_valid_metadata_cannot_bypass_argument_type_checks(bsh_module, changes):
    metadata = _metadata(
        num_actual_tokens=5, actual_seq_lengths_q=[5], seq_lens_list=[5],
        block_tables=_table([[11, 12]]),
    )
    options = {"capacity": 1, "width": 5, "max_model_len": 5}
    good = bsh_module.prepare_metadata(metadata, **options)
    assert good[0] == [5]
    _assert_table(good[1], [[11, 12]])
    options.update(changes)

    with pytest.raises(ValueError):
        bsh_module.prepare_metadata(metadata, **options)

    assert bsh_module.prepare_metadata(metadata, 1, 5, max_model_len=5) is good
    assert good[0] == [5]
    _assert_table(good[1], [[11, 12]])
