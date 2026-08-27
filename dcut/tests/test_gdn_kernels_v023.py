# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

DCUT_ROOT = Path(__file__).resolve().parents[1]
KERNEL_ROOT = DCUT_ROOT / "kernel"


def _read(relative_path: str) -> str:
    return (DCUT_ROOT / relative_path).read_text(encoding="utf-8")


def test_piecewise_gdn_core_uses_recurrent_boundary_inputs() -> None:
    patch = _read("patch_gdn_v023.py")
    piecewise = _read("patch_piecewise.py")
    core = _read("gdn_forward_v023.py")
    assert "native_forward = target_class.forward" in patch
    runner = _read("patch_runner.py")

    assert "native_forward_core = target_class._forward_core" in patch
    assert "target_class._forward_core = _dcut_forward_core" in patch
    assert "return dcut_forward_core(" in patch
    assert "target_class.forward = _dcut_forward" in patch
    assert "_patch_gdn_spec_metadata_builder" in patch
    assert "actual_seq_lengths=attn_metadata.spec_query_start_loc" in patch
    assert "_build_actual_seq_lengths(" not in patch
    assert "torch.ops.vllm.qwen_gdn_attention_core" in core
    assert '_WHOLE_GDN_OP = "vllm::qwen_gdn_attention_core"' in piecewise
    assert '_RECURRENT_OP = "vllm::dcut_gdn_recurrent"' in piecewise
    assert "_ensure_gdn_splitting_ops(" in piecewise
    assert "_dcut_get_gdn_piecewise_spec_bufs" in core
    assert "piecewise_spec_bufs[\"token_mask\"]" not in core
    assert "zero_padded_output=piecewise_spec_bufs is not None" in core
    assert "_dcut_prepare_gdn_eager_state" in runner
    assert "_dcut_prepare_gdn_piecewise_replay" in runner
    assert "CUDAGraphMode.NONE" not in runner
    assert "_dcut_gdn_local_graph" not in runner
    assert "_dcut_gdn_piecewise_capture_sizes" not in runner
    assert "torch.npu.NPUGraph" not in core
    assert "graph.replay()" not in core
    assert "clear_unused_rows=True" in runner
    assert "_dcut_gdn_recurrent_piecewise_safe" in runner
    assert "npu_dcut_causal_conv1d" in core
    assert "torch.ops.vllm.dcut_gdn_recurrent" in core
    assert 'op_name="dcut_gdn_recurrent"' in core
    assert 'mutates_args=["state"]' in core
    assert "ssm_state_indices=spec_state_indices_tensor.flatten()" not in core
    assert "if use_recurrent_boundary" in core
    assert "else torch.ops._C_ascend.npu_dcut_recurrent_gated_delta_rule" in core
    assert 'eager_spec_state["query_start_loc"]' in core
    assert 'piecewise_spec_bufs["qsl"]' in core
    assert 'eager_spec_state["conv_state_offsets"]' not in core
    assert 'piecewise_spec_bufs["conv_state_offsets"]' not in core
    assert 'eager_spec_state["num_accepted_tokens"]' in core


def test_piecewise_replay_preserves_previous_accepted_state() -> None:
    buffers = _read("gdn_buffers.py")
    fill_start = buffers.index(
        "def _dcut_fill_gdn_piecewise_spec_bufs("
    )
    fill_end = buffers.index(
        "def _dcut_prepare_gdn_piecewise_replay("
    )
    fill = buffers[fill_start:fill_end]

    assert "nat[:num_spec_decodes].copy_(" in fill
    assert "torch.sub(nat, 1, out=conv_state_offsets)" not in fill
    assert "torch.lt(" not in fill
    assert "qsl.zero_()" not in fill
    assert "if clear_unused_rows:" in fill
    assert "ssi.fill_(PAD_SLOT_ID)" in fill
    assert "torch.minimum(" not in fill
    assert "current segment length" in fill
    assert "fill_shared_batch=index == 0" in buffers


def test_recurrent_kernel_uses_fixed_request_rows() -> None:
    for relative_path in (
        "kernel/dcut_recurrent_gated_delta_rule/vendor/op_kernel/recurrent_gated_delta_rule.h",
        "kernel/dcut_recurrent_gated_delta_rule/vendor/op_kernel/arch35/recurrent_gated_delta_rule.h",
    ):
        kernel = _read(relative_path)
        assert "batch_i * stateIndexStride_ + (seq_i - seq0)" in kernel
        assert "stateTokenIdx += acceptedTokenNum - 1" in kernel
        assert "acceptedTokenNum > static_cast<int32_t>(stateIndexStride_)" in kernel


def test_recurrent_kernel_consumes_query_start_locations() -> None:
    wrapper = _read(
        "kernel/dcut_recurrent_gated_delta_rule/op_kernel/"
        "dcut_recurrent_gated_delta_rule.cpp"
    )
    assert "DCUT_RECURRENT_QUERY_START_LOC" in wrapper

    for relative_path in (
        "kernel/dcut_recurrent_gated_delta_rule/vendor/op_kernel/recurrent_gated_delta_rule.h",
        "kernel/dcut_recurrent_gated_delta_rule/vendor/op_kernel/arch35/recurrent_gated_delta_rule.h",
    ):
        kernel = _read(relative_path)
        assert "#if defined(DCUT_RECURRENT_QUERY_START_LOC)" in kernel
        assert "const int32_t seqLen = seq1 - seq0;" in kernel
        assert (
            "gamaInQueue_.FreeTensor(gamaInUb);\n"
            "            }\n"
            "            seq0 = seq1;"
        ) in kernel


def test_conv_kernel_derives_state_offsets_from_accepted_counts() -> None:
    wrapper = _read("kernel/dcut_causal_conv1d/op_kernel/dcut_causal_conv1d.cpp")
    kernel = _read("kernel/dcut_causal_conv1d/vendor/op_kernel/causal_conv1d.h")

    assert "DCUT_CAUSAL_CONV_DIRECT_STATE_OFFSETS" not in wrapper
    assert "int32_t accepted = ReadNumAcceptedTokensValue(seq);" in kernel
    assert "stateTokenOffset = accepted - 1;" in kernel


def test_torch_registration_has_graph_metadata() -> None:
    binding = _read("kernel/torch_extension/dcut_torch_binding.cpp")

    assert "TORCH_LIBRARY_FRAGMENT(_C_ascend, ops)" in binding
    assert "TORCH_LIBRARY_IMPL(_C_ascend, PrivateUse1, ops)" in binding
    assert "TORCH_LIBRARY_IMPL(_C_ascend, Meta, ops)" in binding
    assert "Tensor(a!) state" in binding
    assert "Tensor(b!) conv_state" in binding
    assert "Tensor? query_start_loc=None" in binding
    assert "Tensor? num_accepted_tokens=None" in binding
    assert "bool zero_padded_output=False" in binding


def test_truncation_has_no_previous_acceptance_floor() -> None:
    truncate = _read("truncate.py")

    assert "_get_gdn_min_draft_lens" not in truncate
    assert "gdn_min_draft_lens" not in truncate
