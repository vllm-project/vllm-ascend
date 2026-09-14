"""Regression tests for SFA KV-offload attention metadata."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm_ascend.attention.attention_v1 import AscendAttentionState  # noqa: E402
from vllm_ascend.attention.sfa_kv_offload import (  # noqa: E402
    AscendSFAKVOffloadImpl,
    AscendSFAKVOffloadMetadataBuilder,
)
from vllm_ascend.attention.sfa_v1 import AscendSFAMetadataBuilder  # noqa: E402
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (  # noqa: E402
    FSA_EXTERNAL_PLAN_READY_MARKER,
    FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT,
    FSA_SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT,
)


def _make_boundary_decode_metadata():
    return SimpleNamespace(
        context_parallel_metadata=None,
        max_query_len=1,
        num_reqs=1,
        num_actual_tokens=1,
        query_start_loc_cpu=torch.tensor([0, 1]),
        is_prefilling=torch.tensor([True]),
        req_ids_tensor=torch.tensor([7]),
        token_to_req=torch.tensor([0]),
    )


@pytest.mark.parametrize(
    ("kv_transfer_config", "expected"),
    [
        (None, False),
        (SimpleNamespace(is_kv_consumer=False, is_kv_producer=True), False),
        (SimpleNamespace(is_kv_consumer=True, is_kv_producer=True), False),
        (SimpleNamespace(is_kv_consumer=True, is_kv_producer=False), True),
    ],
)
def test_pd_decode_consumer_is_derived_from_kv_role(kv_transfer_config, expected):
    vllm_config = SimpleNamespace(kv_transfer_config=kv_transfer_config)
    with patch.object(AscendSFAMetadataBuilder, "__init__", return_value=None):
        builder = AscendSFAKVOffloadMetadataBuilder(
            kv_cache_spec=None,
            layer_names=[],
            vllm_config=vllm_config,
            device=torch.device("cpu"),
        )

    assert builder.is_pd_decode_consumer is expected


@pytest.mark.parametrize(
    ("is_pd_decode_consumer", "expected_decodes", "expected_prefills"),
    [
        (True, 1, 0),
        (False, 0, 1),
    ],
)
def test_boundary_token_classification_depends_on_pd_decode_role(
    is_pd_decode_consumer,
    expected_decodes,
    expected_prefills,
):
    builder = AscendSFAKVOffloadMetadataBuilder.__new__(AscendSFAKVOffloadMetadataBuilder)
    builder.decode_threshold = 1
    builder.is_pd_decode_consumer = is_pd_decode_consumer
    metadata = SimpleNamespace(attn_state=AscendAttentionState.DecodeOnly)

    with patch(
        "vllm_ascend.attention.utils.is_pd_decode_recompute_scheduler_enabled",
        return_value=False,
    ):
        builder._populate_offload_metadata(metadata, _make_boundary_decode_metadata())

    assert metadata.num_decodes == expected_decodes
    assert metadata.num_prefills == expected_prefills
    assert metadata.num_decode_tokens == expected_decodes
    assert metadata.req_ids_tensor.tolist() == [7]
    assert metadata.token_to_req.tolist() == [0]
    assert AscendSFAKVOffloadImpl._is_decode_only(metadata) is is_pd_decode_consumer


def test_pd_decode_consumer_still_rejects_long_prefill_classification():
    builder = AscendSFAKVOffloadMetadataBuilder.__new__(AscendSFAKVOffloadMetadataBuilder)
    builder.decode_threshold = 1
    builder.is_pd_decode_consumer = True
    metadata = SimpleNamespace()
    common_metadata = _make_boundary_decode_metadata()
    common_metadata.max_query_len = 2
    common_metadata.num_actual_tokens = 2
    common_metadata.query_start_loc_cpu = torch.tensor([0, 2])

    with patch(
        "vllm_ascend.attention.utils.is_pd_decode_recompute_scheduler_enabled",
        return_value=False,
    ):
        builder._populate_offload_metadata(metadata, common_metadata)

    assert metadata.num_decodes == 0
    assert metadata.num_prefills == 1
    assert metadata.num_decode_tokens == 0


def _make_fused_overlap_impl() -> AscendSFAKVOffloadImpl:
    impl = AscendSFAKVOffloadImpl.__new__(AscendSFAKVOffloadImpl)
    impl.block_size = 4
    impl.local_num_heads = 2
    impl.lru_resident_capacity = 8
    impl.sfa_sparse_topk = 4
    impl.max_num_topk_rows = 4
    impl.scale = 0.5
    impl.selection_kv_block_table = None
    impl.selection_kv_block_status = None
    impl.selection_membership_map = None
    impl.fused_overlap_last_req_ids = None
    impl._fused_overlap_selection_capacity = None
    impl._fused_overlap_decode_logged = True
    impl.skip_topk = False
    return impl


def test_mtp_rewrite_invalidates_membership_slot_map():
    impl = _make_fused_overlap_impl()
    topk_count = 4
    selection_status = torch.full((4, 1, 8), -1, dtype=torch.int32)
    selection_status[:, 0, :topk_count] = torch.tensor([8, 9, 10, 11])
    selection_status[:, 0, topk_count] = topk_count
    membership = torch.full(
        (4, FSA_SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT),
        -1,
        dtype=torch.int16,
    )
    control = membership[
        :, FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT : FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT + 8
    ]
    control[:, 0] = 0x5A4D
    last_req_ids = torch.full((4,), 7, dtype=torch.int64)
    metadata = SimpleNamespace(
        token_to_req=torch.zeros(4, dtype=torch.int32),
        req_ids_tensor=torch.tensor([7], dtype=torch.int64),
    )

    with patch(
        "vllm_ascend.attention.sfa_kv_offload.get_forward_context",
        return_value=SimpleNamespace(capturing=False),
    ):
        impl._invalidate_fused_overlap_selection_rows(
            selection_status,
            membership,
            last_req_ids,
            metadata,
            num_tokens=4,
            num_reqs=1,
            topk_count=topk_count,
            seq_lens=torch.tensor([12], dtype=torch.int32),
            cum_query_lens=torch.tensor([4], dtype=torch.int32),
        )

    assert bool((selection_status[:, 0, :topk_count] == -1).all())
    assert bool((control[:, 0] == -1).all())


def test_fused_overlap_external_plan_passes_raw_topk_and_full_selection_state():
    """
    Test that the fused overlap external plan receives the raw topk indices and the full selection state,
    and that the fused op receives the correct inputs.
    """
    impl = _make_fused_overlap_impl()
    ql_nope = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    q_pe = torch.ones((2, 2, 1), dtype=torch.float32)
    topk = torch.tensor([[0, -1, 2, 5], [3, 7, 1, -1]], dtype=torch.int64)
    full_kv_cpu = torch.zeros((3, 4, 1, 3), dtype=torch.float32)
    full_rope_cpu = torch.zeros((3, 4, 1, 1), dtype=torch.float32)
    metadata = SimpleNamespace(
        num_decodes=2,
        token_to_req=torch.tensor([0, 1], dtype=torch.int32),
        req_ids_tensor=torch.tensor([101, 202], dtype=torch.int64),
        block_table=torch.tensor([[0, 1], [1, 2]], dtype=torch.int32),
    )
    call_order = []
    fused_inputs = {}
    plan_inputs = {}

    def allocate_membership(row_capacity):
        membership = torch.full(
            (row_capacity, FSA_SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT),
            -1,
            dtype=torch.int16,
        )
        control = membership[
            :,
            FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT : FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT + 8,
        ]
        control[:, 1] = FSA_EXTERNAL_PLAN_READY_MARKER
        control[:, 2] = 4
        control[:, 3] = FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT - 4
        return membership

    def prepare_plan(**kwargs):
        call_order.append("plan")
        plan_inputs.update(kwargs)
        return True

    def inject_current(**kwargs):
        call_order.append("inject")
        assert kwargs["layer_name"] == "model.layers.0.self_attn"
        assert kwargs["num_tokens"] == 2
        assert kwargs["selection_kv_cache"].shape == (8, 4, 3)
        assert kwargs["selection_k_rope"].shape == (8, 4, 1)
        assert kwargs["capturing"] is False

    def wait_for_writeback(capturing):
        call_order.append("wait_writeback")
        assert capturing is False

    def fused_op(**kwargs):
        call_order.append("fused")
        fused_inputs.update(kwargs)
        return kwargs["query"] + 10

    manager = SimpleNamespace(
        topk_buffers_k=[torch.zeros((4, 8, 1, 3), dtype=torch.float32)],
        topk_buffers_v=[torch.zeros((4, 8, 1, 1), dtype=torch.float32)],
        _get_offload_layer_id=lambda _: 0,
        get_fused_overlap_cpu_kv_inputs=lambda _: (full_kv_cpu, full_rope_cpu),
        allocate_fused_overlap_membership_map=allocate_membership,
        prepare_fused_overlap_external_plan=prepare_plan,
        inject_current_kv_into_selection=inject_current,
        wait_for_current_kv_writeback=wait_for_writeback,
    )

    with (
        patch.object(impl, "_require_custom_op", return_value=fused_op),
        patch(
            "vllm_ascend.attention.sfa_kv_offload.get_sparse_kv_offload_manager",
            return_value=manager,
        ),
        patch(
            "vllm_ascend.attention.sfa_kv_offload.get_forward_context",
            return_value=SimpleNamespace(capturing=False),
        ),
    ):
        output = impl._execute_fused_overlap_offload_decode(
            ql_nope,
            q_pe,
            topk,
            metadata,
            torch.tensor([1, 2], dtype=torch.int32),
            torch.tensor([4, 4], dtype=torch.int32),
            "model.layers.0.self_attn",
        )

    torch.testing.assert_close(output, ql_nope + 10)
    assert call_order == [
        "plan",
        "inject",
        "fused",
        "wait_writeback",
    ]
    assert fused_inputs["selection_kv_cache"].shape == (8, 4, 3)
    assert fused_inputs["selection_k_rope"].shape == (8, 4, 1)
    assert fused_inputs["selection_kv_block_table"].shape == (4, 2)
    assert fused_inputs["selection_kv_block_status"].shape == (4, 1, 8)
    assert fused_inputs["selection_membership_map"].shape == (4, 16400)
    assert fused_inputs["selection_membership_map"].dtype == torch.int16
    torch.testing.assert_close(
        fused_inputs["query"],
        torch.cat([ql_nope, q_pe], dim=-1),
    )
    assert plan_inputs["selection_membership_map"].data_ptr() == fused_inputs["selection_membership_map"].data_ptr()
    expected_topk = torch.tensor(
        [[[0, -1, 2, 5]], [[3, 7, 1, -1]]],
        dtype=torch.int32,
    )
    torch.testing.assert_close(
        fused_inputs["selection_topk_indices"],
        expected_topk,
    )
    torch.testing.assert_close(
        plan_inputs["topk_indices_npu"],
        expected_topk.squeeze(1),
    )
    torch.testing.assert_close(
        plan_inputs["req_ids_npu"],
        torch.tensor([101, 202], dtype=torch.int64),
    )
    torch.testing.assert_close(
        plan_inputs["stable_prefix_lens_npu"],
        torch.tensor([3, 3], dtype=torch.int32),
    )
    torch.testing.assert_close(
        plan_inputs["visible_seq_lens_npu"],
        torch.tensor([4, 4], dtype=torch.int32),
    )
    assert plan_inputs["capturing"] is False


def test_fused_overlap_common_inputs_are_reused_only_within_one_forward():
    impl = _make_fused_overlap_impl()
    metadata = SimpleNamespace(
        num_decodes=2,
        token_to_req=torch.tensor([0, 1], dtype=torch.int32),
        req_ids_tensor=torch.tensor([101, 202], dtype=torch.int64),
        block_table=torch.tensor([[0, 1], [1, 2]], dtype=torch.int32),
    )
    topk = torch.tensor(
        [[[0, 1, 2, 3]], [[3, 2, 1, 0]]],
        dtype=torch.int32,
    )
    query_lens = torch.tensor([1, 2], dtype=torch.int32)
    kv_lens = torch.tensor([4, 5], dtype=torch.int32)
    first_forward = SimpleNamespace(capturing=False)
    second_forward = SimpleNamespace(capturing=False)
    current_forward = [first_forward]

    with patch(
        "vllm_ascend.attention.sfa_kv_offload.get_forward_context",
        side_effect=lambda: current_forward[0],
    ):
        first = impl._prepare_fused_overlap_decode_common_inputs(
            metadata,
            num_tokens=2,
            num_reqs=2,
            topk_indices_decode=topk,
            actual_seq_lengths_query_decode=query_lens,
            actual_seq_lengths_key_decode=kv_lens,
        )
        reused = impl._prepare_fused_overlap_decode_common_inputs(
            metadata,
            num_tokens=2,
            num_reqs=2,
            topk_indices_decode=topk,
            actual_seq_lengths_query_decode=query_lens,
            actual_seq_lengths_key_decode=kv_lens,
        )
        kv_lens.add_(1)
        current_forward[0] = second_forward
        refreshed = impl._prepare_fused_overlap_decode_common_inputs(
            metadata,
            num_tokens=2,
            num_reqs=2,
            topk_indices_decode=topk,
            actual_seq_lengths_query_decode=query_lens,
            actual_seq_lengths_key_decode=kv_lens,
        )

    assert reused is first
    assert refreshed is not first
    torch.testing.assert_close(
        first.seq_len_thresholds.reshape(-1),
        torch.tensor([4, 5], dtype=torch.int32),
    )
    torch.testing.assert_close(
        refreshed.seq_len_thresholds.reshape(-1),
        torch.tensor([5, 6], dtype=torch.int32),
    )
