# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Native band intersection contracts; device correctness is tested separately."""

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from vllm_ascend.worker.v2.spec_decode.draft_window_read import (
    NativeSWASource,
    materialize_direct_prefix,
    materialize_native_swa_intersection,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_swa import NativeSWABuilder


@dataclass
class NativeMetadata:
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor
    seq_lens_gpu: torch.Tensor
    seq_lens_list: list[int]
    block_tables: torch.Tensor
    slot_mapping: torch.Tensor
    positions: torch.Tensor
    causal: bool = True
    native_window: int = 4096


def prepared(span=128, continuing=(False, False), window=1024):
    # Persistent rows differ from the compact builder batch; old null columns
    # are legal for native SWA and must never certify a resident Full prefix.
    canonical = torch.arange(80, dtype=torch.int32).reshape(4, 20)
    canonical[:, :2] = 0
    rows = np.array([3, 1])
    device_exact = torch.tensor([1151, 1152], dtype=torch.int32)
    exact = device_exact.clone()
    common = SimpleNamespace(
        num_reqs=2,
        seq_lens=device_exact,
        seq_lens_cpu_upper_bound=torch.tensor([1152, 1159], dtype=torch.int32),
        block_table_tensor=canonical[rows].clone(),
        query_start_loc_cpu=torch.tensor([0, 8, 16], dtype=torch.int32),
    )
    metadata = NativeMetadata(
        device_exact,
        device_exact,
        device_exact,
        exact.tolist(),
        common.block_table_tensor,
        torch.arange(16),
        torch.arange(16) + 1136,
    )
    batch = SimpleNamespace(
        num_reqs=2,
        req_ids=["new", "other"],
        idx_mapping_np=rows,
        idx_mapping=torch.from_numpy(rows),
        num_computed_prefill_tokens_np=np.zeros(2, dtype=np.int32),
        prefill_len_np=np.where(continuing, 100, 8),
        num_scheduled_tokens=np.full(2, 8),
        num_draft_tokens_per_req=None,
    )
    retained = []
    owner = object()
    runtime = SimpleNamespace(
        _owner=owner,
        _input_batch=batch,
        gids=(1,),
        window_size=window,
        pool=SimpleNamespace(retain=lambda actual_owner, *values: retained.extend(values)),
        speculator=SimpleNamespace(
            num_query_per_req=8,
            block_tables=SimpleNamespace(
                block_tables=[SimpleNamespace(gpu=canonical)],
                kernel_block_sizes=[span],
                num_blocks=SimpleNamespace(np=np.full((1, 4), 20)),
            ),
        ),
    )

    class Builder:
        calls = 0

        def build(self, common_prefix_len, actual_common, **kwargs):
            assert actual_common is common
            self.calls += 1
            return metadata

    builder = Builder()
    return runtime, owner, common, metadata, canonical, retained, builder


@pytest.mark.parametrize("span,starts", [(128, [0, 128]), (64, [64, 128])])
def test_current_exact_rejection_and_group_geometry(span, starts):
    runtime, owner, common, original, table, retained, builder = prepared(span)
    before = table.clone()
    proxy = NativeSWABuilder(builder, runtime, owner, 0, 4096)
    # No additional D2H, including hidden .cpu() or .to("cpu"). Host source
    # checks supplement the independent native test, not a timing guarantee.
    with patch.object(torch.Tensor, "cpu", side_effect=AssertionError("new D2H")):
        result = proxy.build(0, common)
    assert builder.calls == 1
    assert result is not original
    assert result.slot_mapping is original.slot_mapping
    assert result.positions is original.positions
    assert result.native_window == original.native_window
    assert result.seq_lens_list == [1151 - starts[0], 1152 - starts[1]]
    assert original.seq_lens_list == [1151, 1152]
    for i, row in enumerate([3, 1]):
        columns = (int(original.seq_lens_cpu[i]) + span - 1) // span
        expected = before[row, starts[i] // span : columns]
        torch.testing.assert_close(result.block_tables[i, : len(expected)], expected, rtol=0, atol=0)
    torch.testing.assert_close(table, before, rtol=0, atol=0)
    assert any(item is result for item in retained)
    assert any(item is original for item in retained)
    assert any(item is result.block_tables for item in retained)
    assert common.block_table_tensor is original.block_tables
    assert common.seq_lens is original.seq_lens_gpu


def test_native_band_read_set_intersection_uses_actual_query_positions():
    for end in [8, 1151, 1152, 4095, 4096, 33023]:
        for span in [64, 128]:
            for window in [8, 1000, 1024, 4096, 4097]:
                start = max(end - window, 0) // span * span
                for q in range(end - 8, end):
                    native_absolute = set(range(max(q - 4096, 0), q + 1))
                    local_q = q - start
                    actual = {start + k for k in range(max(local_q - 4096, 0), local_q + 1)}
                    assert actual == {k for k in native_absolute if k >= start}


@pytest.mark.parametrize("continuing", [(True, True), (True, False)])
def test_continuing_prefill_retains_original_native_read(continuing):
    runtime, owner, common, original, _, _, builder = prepared(continuing=continuing)
    result = NativeSWABuilder(builder, runtime, owner, 0, 4096).build(0, common)
    assert result.seq_lens_list[0] == 1151
    if all(continuing):
        assert result is original
    else:
        assert result.seq_lens_list[1] == 1024


def test_stale_owner_mapping_and_repeated_build_are_rejected():
    runtime, owner, common, _, _, _, builder = prepared()
    proxy = NativeSWABuilder(builder, runtime, owner, 0, 4096)
    runtime._owner = object()
    with pytest.raises(RuntimeError, match="stale"):
        proxy.build(0, common)
    runtime._owner = owner
    proxy.build(0, common)
    with pytest.raises(RuntimeError, match="repeated"):
        proxy.build(0, common)
    proxy = NativeSWABuilder(builder, runtime, owner, 0, 4096)
    build = builder.build

    def changed(*args, **kwargs):
        result = build(*args, **kwargs)
        runtime._input_batch.idx_mapping_np[:] = [1, 3]
        return result

    builder.build = changed
    with pytest.raises(RuntimeError, match="mapping changed"):
        proxy.build(0, common)


@pytest.mark.parametrize("corruption", ["gpu", "seq_input", "cpu_list", "table", "causal", "query_width"])
def test_builder_result_must_be_current_unprojected_pair(corruption):
    runtime, owner, common, original, _, _, builder = prepared()
    if corruption == "gpu":
        original.seq_lens_gpu = common.seq_lens.clone()
    elif corruption == "seq_input":
        original.seq_lens = common.seq_lens.clone()
    elif corruption == "cpu_list":
        original.seq_lens_list = None
    elif corruption == "table":
        original.block_tables = common.block_table_tensor.clone()
    elif corruption == "causal":
        original.causal = False
    else:
        common.query_start_loc_cpu[-1] += 1
    with pytest.raises(ValueError, match="fresh unprojected|query width"):
        NativeSWABuilder(builder, runtime, owner, 0, 4096).build(0, common)


def test_native_source_does_not_acquire_full_residency_by_duck_typing():
    runtime, _, _, _, _, _, _ = prepared()
    source = NativeSWASource.from_block_tables(runtime.speculator.block_tables, 0, np.array([3, 1]), 4096)
    lengths = torch.tensor([1151, 1152], dtype=torch.int32)
    with pytest.raises(ValueError, match="resident-prefix"):
        materialize_direct_prefix(
            source,
            exact_cpu=lengths,
            exact_device=lengths,
            upper_cpu=lengths,
            starts_cpu=torch.tensor([0, 128]),
            padded_rows=2,
        )
    with pytest.raises(ValueError, match="cover every native query"):
        materialize_native_swa_intersection(
            source,
            exact_cpu=lengths,
            exact_device=lengths,
            upper_cpu=lengths,
            starts_cpu=lengths,
            padded_rows=2,
            query_width=8,
        )


def test_request_reorder_and_replacement_never_reuses_old_projected_table():
    runtime, owner, common, original, table, _, builder = prepared()
    first = NativeSWABuilder(builder, runtime, owner, 0, 4096).build(0, common)
    first_table = first.block_tables.clone()
    runtime._input_batch.req_ids[:] = ["replacement", "new"]
    runtime._input_batch.idx_mapping_np[:] = [1, 3]
    table[1] += 500
    common.block_table_tensor = table[[1, 3]].clone()
    original.block_tables = common.block_table_tensor
    second = NativeSWABuilder(builder, runtime, owner, 0, 4096).build(0, common)
    assert second.block_tables[0, 0] == table[1, 0]
    torch.testing.assert_close(first.block_tables, first_table, rtol=0, atol=0)
    assert first.block_tables.untyped_storage().data_ptr() != second.block_tables.untyped_storage().data_ptr()


def test_misnamed_cpu_field_is_not_a_new_exact_consumer():
    runtime, owner, common, original, _, _, builder = prepared()
    original.seq_lens_cpu = object()
    result = NativeSWABuilder(builder, runtime, owner, 0, 4096).build(0, common)
    assert result.seq_lens_list == [1151, 1024]
    assert result.seq_lens_cpu.device.type == "cpu"


def test_noncausal_native_window_preserves_declared_mask_and_full_writes():
    runtime, owner, common, original, table, retained, builder = prepared()
    original.causal = False
    before = table.clone()
    result = NativeSWABuilder(builder, runtime, owner, 0, 4096, causal=False).build(0, common)
    assert result.causal is False
    assert result.seq_lens_list == [1151, 1024]
    assert result.slot_mapping is original.slot_mapping
    assert result.positions is original.positions
    assert any(item is result for item in retained)
    torch.testing.assert_close(table, before, atol=0, rtol=0)


def test_noncausal_native_window_requires_resident_read_geometry():
    runtime, owner, _, _, _, _, builder = prepared(window=4096)
    with pytest.raises(NotImplementedError, match="resident"):
        NativeSWABuilder(builder, runtime, owner, 0, 4096, causal=False)
    assert builder.calls == 0


def test_noncausal_metadata_cannot_change_its_declared_mask():
    runtime, owner, common, _, _, _, builder = prepared()
    with pytest.raises(ValueError, match="fresh unprojected"):
        NativeSWABuilder(builder, runtime, owner, 0, 4096, causal=False).build(0, common)


@pytest.mark.parametrize("continuing", [False, True])
@pytest.mark.parametrize("causal", [False, True])
def test_native_only_runtime_accounts_for_the_actual_builder_and_retains_readers(monkeypatch, continuing, causal):
    from vllm_ascend.worker.v2.spec_decode import draft_window_runtime as runtime_module
    from vllm_ascend.worker.v2.spec_decode.draft_window_policy import resolve_draft_window_policy

    prepared_runtime, _, common, original, table, _, builder = prepared(continuing=(continuing, continuing))
    original.causal = causal
    speculator = prepared_runtime.speculator
    speculator.draft_model_config = SimpleNamespace(enforce_eager=True)
    speculator.speculative_config = SimpleNamespace(enforce_eager=True)
    speculator.vllm_config = SimpleNamespace(max_concurrent_batches=2)
    speculator.num_speculative_steps = 7

    def native_only(runtime, spec):
        runtime.native_swa_intersections = {0: 4096}
        runtime.native_swa_causal = {0: causal}
        return ()

    class Completion:
        def is_complete(self):
            return False

        def wait(self):
            raise AssertionError("No reader wait is needed with a free lane")

    monkeypatch.setattr(runtime_module.DraftWindowRuntime, "_validate_route", native_only)
    monkeypatch.setattr(runtime_module, "_NPUCompletion", Completion)
    runtime = runtime_module.DraftWindowRuntime(
        speculator, resolve_draft_window_policy(requested_window_size=1024, method="dflash")
    )
    before = table.clone()
    with runtime.proposal_context(speculator, prepared_runtime._input_batch):
        owner = runtime._owner
        wrapped = runtime.metadata_builder_callback(0)(0, builder, ["draft.0"])
        result = wrapped.build(0, common)
        assert runtime._build_count == 1
        assert result.slot_mapping is original.slot_mapping
        assert result.positions is original.positions
        if continuing:
            assert result is original
        else:
            assert any(value is result for value in owner.retained)
        torch.testing.assert_close(table, before, rtol=0, atol=0)
    assert owner.completion is not None
    assert not owner.completion.is_complete()
    assert runtime.pool.reclaim_completed() == 0


def test_native_only_proposal_cannot_be_certified_by_one_of_two_groups(monkeypatch):
    from vllm_ascend.worker.v2.spec_decode import draft_window_runtime as runtime_module
    from vllm_ascend.worker.v2.spec_decode.draft_window_policy import resolve_draft_window_policy

    prepared_runtime, _, common, _, _, _, builder = prepared()
    speculator = prepared_runtime.speculator
    speculator.draft_model_config = SimpleNamespace(enforce_eager=True)
    speculator.speculative_config = SimpleNamespace(enforce_eager=True)
    speculator.vllm_config = SimpleNamespace(max_concurrent_batches=2)
    speculator.num_speculative_steps = 7

    def native_only(runtime, spec):
        runtime.native_swa_intersections = {0: 4096, 1: 4096}
        return ()

    monkeypatch.setattr(runtime_module.DraftWindowRuntime, "_validate_route", native_only)
    runtime = runtime_module.DraftWindowRuntime(
        speculator, resolve_draft_window_policy(requested_window_size=1024, method="dflash")
    )
    with (
        pytest.raises(RuntimeError, match="did not consume a read projection"),
        runtime.proposal_context(speculator, prepared_runtime._input_batch),
    ):
        runtime.metadata_builder_callback(0)(0, builder, ["draft.0"]).build(0, common)
        assert runtime._build_count == 0
    assert runtime.pool._poisoned
