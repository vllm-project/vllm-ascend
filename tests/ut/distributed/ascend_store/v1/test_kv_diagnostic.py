# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for the temporary byte probe; native transfer is validated only by the NPU CI."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.e2e.pull_request.one_card.ascendstore_v1_kv_diagnostic import (
    KVByteAudit,
    bf16_all_finite,
    bf16_difference,
    compare_byte_audit_phases,
    compare_bytes,
    consecutive_runs,
    read_prefix_bytes,
)


class ByteAuditFixture:
    def __init__(self, implementation: str = "v1") -> None:
        self.caches = {
            "layer.0": tuple((torch.arange(80).reshape(20, 2, 1, 2) + i).to(torch.bfloat16) for i in (0, 100))
        }
        self.source_blocks = [1, 3, 5]
        self.destination_blocks = [7, 9, 11]
        self.payload: list[torch.Tensor] = []
        self.copied = True
        self.corrupt_load = False
        self.runner = SimpleNamespace(
            _prepare_inputs_impl=lambda batch: batch,
            block_tables=SimpleNamespace(block_sizes=[2], kernel_block_sizes=[2], input_block_tables=[]),
        )
        self.connector = SimpleNamespace(start_load_kv=self.load, wait_for_save=self.store)
        self.connector._get_connector_metadata = lambda: self.metadata
        self.audit = KVByteAudit(self.runner, self.connector, self.caches, implementation, lambda: None, 2, 4, 1)
        self.original_prepare = self.runner._prepare_inputs_impl
        self.audit.install()

    def store(self) -> None:
        if self.audit.phase is not None and self.audit.phase["mode"] == "record":
            self.payload = [cache[self.source_blocks[:-1]].clone() for cache in self.caches["layer.0"]]

    def load(self) -> None:
        if self.copied:
            for cache, payload in zip(self.caches["layer.0"], self.payload, strict=True):
                for index, block in enumerate(self.destination_blocks[:-1]):
                    cache[block].copy_(payload[index])
        if self.corrupt_load:
            self.caches["layer.0"][0][self.destination_blocks[0]].view(torch.uint8).flatten()[0] ^= 1

    def begin(self, mode: str) -> None:
        phase_id = f"v1-c1-{mode}-cold" if mode == "record" else f"v1-c1-{mode}-external"
        request_id = f"cmpl-ascendstore-diag-{phase_id}-p0-0-uuid"
        blocks = self.source_blocks if mode == "record" else self.destination_blocks
        self.runner.block_tables.input_block_tables = [torch.tensor([blocks], dtype=torch.int32)]
        command = SimpleNamespace(
            request_id=request_id,
            store_range=SimpleNamespace(start_token=0, end_token=4),
            load_range=SimpleNamespace(start_token=0, end_token=4),
            block_ids_by_group=(tuple(blocks),),
            num_prompt_tokens=5,
        )
        request = SimpleNamespace(
            req_id=request_id,
            can_save=True,
            save_start_token=0,
            save_end_token=4,
            num_prompt_tokens=5,
            block_ids_by_group=[blocks],
            load_spec=SimpleNamespace(can_load=True, vllm_cached_tokens=0, kvpool_cached_tokens=4),
        )
        self.metadata = SimpleNamespace(
            load=SimpleNamespace(commands=(command,)),
            store=SimpleNamespace(source_pending_commands=(command,), source_ready_commands=()),
            requests=[request],
        )
        computed, scheduled = (0, 5) if mode == "record" else (4, 1)
        batch = SimpleNamespace(
            req_ids=[request_id],
            num_reqs=1,
            num_reqs_after_padding=1,
            num_tokens=scheduled,
            num_tokens_after_padding=scheduled,
            idx_mapping_np=np.array([0]),
            num_scheduled_tokens=np.array([scheduled]),
            num_computed_tokens_np=np.array([computed]),
            prefill_len_np=np.array([5]),
            query_start_loc_np=np.array([0, scheduled]),
            is_prefilling_np=np.array([True]),
            attn_state=SimpleNamespace(name="PrefillNoCache"),
        )
        self.audit.begin_phase(phase_id, mode)
        assert self.runner._prepare_inputs_impl(batch) is batch

    def record(self) -> dict:
        self.begin("record")
        self.connector.wait_for_save()
        result = self.audit.end_phase()
        # The independent disk reference and mock backend payload must survive source block reuse.
        for cache in self.caches["layer.0"]:
            for block in self.source_blocks:
                cache[block].fill_(-100)
        return result


@pytest.mark.parametrize("implementation", ("production", "v1"))
@pytest.mark.parametrize("mode", ("observe", "poison"))
def test_audit_compares_relocated_prefix_bytes_and_separates_recomputed_tail(implementation: str, mode: str) -> None:
    fixture = ByteAuditFixture(implementation)
    try:
        reference = fixture.record()
        fixture.begin(mode)
        fixture.connector.start_load_kv()
        # Simulate the next model forward changing only the last, non-transferred prompt token.
        for cache in fixture.caches["layer.0"]:
            cache[fixture.destination_blocks[-1], 0].fill_(0.125)
        fixture.connector.wait_for_save()
        result = fixture.audit.end_phase()
        assert reference["requests"][0]["blocks"] == fixture.source_blocks
        assert [row["stage"] for row in result["requests"]] == ["after_load_before_forward", "after_forward"]
        for row in result["requests"]:
            assert row["different_prefix_bytes"] == 0 and row["prefix_bytes"] == 32
            assert row["source_blocks"] != row["destination_blocks"]
            assert all(entry["prefix"]["equal"] for entry in row["entries"])
        assert result["requests"][0]["poison_verified"] == (mode == "poison")
        assert all(
            entry["recomputed_last_prompt_token"]["different_bytes"] > 0 for entry in result["requests"][1]["entries"]
        )
    finally:
        fixture.audit.close()
    assert fixture.runner._prepare_inputs_impl is fixture.original_prepare
    assert fixture.connector.start_load_kv == fixture.load and fixture.connector.wait_for_save == fixture.store
    assert fixture.audit.reference.closed


@pytest.mark.parametrize("failure", ("skipped_load", "corrupt_byte", "post_forward_write"))
def test_audit_rejects_missing_load_corruption_and_post_forward_prefix_writes(failure: str) -> None:
    fixture = ByteAuditFixture()
    try:
        fixture.record()
        fixture.begin("poison")
        fixture.copied = failure != "skipped_load"
        fixture.corrupt_load = failure == "corrupt_byte"
        if failure == "post_forward_write":
            fixture.connector.start_load_kv()
            fixture.caches["layer.0"][0][fixture.destination_blocks[0], 0, 0, 0] = 999
            with pytest.raises(AssertionError, match="byte-for-byte"):
                fixture.connector.wait_for_save()
        else:
            with pytest.raises(AssertionError, match="byte-for-byte"):
                fixture.connector.start_load_kv()
    finally:
        fixture.audit.close()


def test_transfer_plan_must_match_actual_consumer_table_before_poisoning() -> None:
    fixture = ByteAuditFixture()
    try:
        fixture.record()
        fixture.begin("poison")
        fixture.metadata.load.commands[0].block_ids_by_group = ((8, 10, 12),)
        before = fixture.caches["layer.0"][0].clone()
        with pytest.raises(AssertionError, match="block tables disagree"):
            fixture.connector.start_load_kv()
        assert torch.equal(before, fixture.caches["layer.0"][0])
    finally:
        fixture.audit.close()


def test_incomplete_phase_and_reference_overwrite_are_not_accepted() -> None:
    fixture = ByteAuditFixture()
    try:
        fixture.begin("record")
        with pytest.raises(AssertionError, match="Incomplete"):
            fixture.audit.end_phase()
        fixture.record()
        with pytest.raises(AssertionError, match="Record once"):
            fixture.audit.begin_phase("v1-c1-record-cold", "record")
    finally:
        fixture.audit.close()


def test_byte_comparison_detects_signed_zero_and_rejects_truncation() -> None:
    positive = torch.tensor([0.0, 1.0], dtype=torch.bfloat16).view(torch.uint8).numpy().tobytes()
    negative = torch.tensor([-0.0, 1.0], dtype=torch.bfloat16).view(torch.uint8).numpy().tobytes()
    comparison = compare_bytes(positive, negative)
    assert comparison["equal"] is False and comparison["different_bytes"] == 1
    assert comparison["first_different_byte"] == 1
    assert bf16_difference(positive, negative)["different_elements"] == 0
    with pytest.raises(AssertionError, match="truncated"):
        compare_bytes(positive, negative[:-1])
    assert bf16_all_finite(positive)
    assert not bf16_all_finite(torch.tensor([float("nan")], dtype=torch.bfloat16).view(torch.uint8).numpy().tobytes())


def test_snapshot_preserves_logical_order_for_noncontiguous_physical_blocks() -> None:
    cache = torch.arange(48).reshape(12, 2, 1, 2).to(torch.bfloat16)
    assert consecutive_runs([4, 5, 1, 8, 9]) == [(4, 2), (1, 1), (8, 2)]
    expected = cache[[4, 5, 1, 8, 9]].view(torch.uint8).numpy().tobytes()
    assert read_prefix_bytes(cache, [4, 5, 1, 8, 9]) == expected


def test_output_contrasts_do_not_mix_byte_probe_effect_with_connector_effect() -> None:
    phases = []
    for implementation in ("production", "v1"):
        for source, mode in (
            ("cold", "off"),
            ("external", "off"),
            ("cold", "record"),
            ("external", "observe"),
            ("external", "poison"),
        ):
            phases.append(
                {
                    "id": f"{implementation}-c1-{mode}-{source}",
                    "implementation": implementation,
                    "case": "c1",
                    "source": source,
                    "byte_audit_mode": mode,
                    "batches": [],
                    "observations": [
                        {"request_index": 0, "prompt_sha256": "same", "token_ids": [1 if source == "cold" else 2]}
                    ],
                }
            )
    comparisons = compare_byte_audit_phases(phases)
    assert {row["contrast"] for row in comparisons} == {
        "cold-external",
        "byte-audit-effect",
        "poison-effect",
        "production-v1",
    }
    assert all(bool(row["differences"]) == (row["contrast"] == "cold-external") for row in comparisons)
