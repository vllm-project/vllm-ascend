# SPDX-License-Identifier: Apache-2.0
"""Source and CPU regressions for the Qwen4Exp PLE graph-safe path."""

from __future__ import annotations

import ast
import copy
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[4]
NVIDIA_PLE = ROOT / "vllm_ascend" / "models" / "qwen4_exp" / "nvidia" / "ple_layer.py"
NVIDIA_MODEL = ROOT / "vllm_ascend" / "models" / "qwen4_exp" / "nvidia" / "model.py"
MODEL_RUNNER = ROOT / "vllm_ascend" / "worker" / "model_runner_v1.py"
SHORT_CONV_ATTN = ROOT / "vllm_ascend" / "models" / "qwen4_exp" / "short_conv_attn.py"
AMD_PLE = ROOT / "vllm_ascend" / "models" / "qwen4_exp" / "amd" / "ple_layer.py"
AMD_MODEL = ROOT / "vllm_ascend" / "models" / "qwen4_exp" / "amd" / "model.py"


def _method_node(path: Path, class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return child
    raise AssertionError(f"method {class_name}.{method_name} not found in {path}")


def _standalone_method(method_name: str) -> Callable:
    method = copy.deepcopy(_method_node(NVIDIA_PLE, "Qwen4ExpPLELayer", method_name))
    method.decorator_list = []
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"torch": torch}
    exec(compile(module, NVIDIA_PLE, "exec"), namespace)
    return namespace[method_name]


def _standalone_function(path: Path, function_name: str) -> Callable:
    tree = ast.parse(path.read_text())
    function = next(
        copy.deepcopy(node) for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    function.decorator_list = []
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"torch": torch}
    exec(compile(module, path, "exec"), namespace)
    return namespace[function_name]


def test_ple_graph_rows_stay_static_across_128_and_120_token_captures() -> None:
    pad_inputs = _standalone_function(MODEL_RUNNER, "_pad_qwen4_exp_ple_graph_inputs")
    eos = 248044
    context_buffer = torch.empty((32, 2), dtype=torch.int32)
    query_start_loc_buffer = torch.empty((33,), dtype=torch.int32)

    for num_reqs in (32, 30):
        runtime_context = torch.arange(num_reqs * 2, dtype=torch.int32).reshape(num_reqs, 2)
        runtime_query_start_loc = torch.arange(0, (num_reqs + 1) * 4, 4, dtype=torch.int32)
        context, query_start_loc = pad_inputs(
            context_buffer,
            query_start_loc_buffer,
            runtime_context,
            runtime_query_start_loc,
            num_reqs,
            eos,
        )
        assert context.shape == (32, 2)
        assert query_start_loc.shape == (33,)
        torch.testing.assert_close(context[:num_reqs], runtime_context)
        torch.testing.assert_close(query_start_loc[: num_reqs + 1], runtime_query_start_loc)
        if num_reqs < 32:
            assert torch.all(context[num_reqs:] == eos)
            assert torch.all(query_start_loc[num_reqs + 1 :] == runtime_query_start_loc[-1])
        packed = torch.zeros((32, 4096), dtype=torch.int64)
        assert torch.cat((context.long(), packed), dim=-1).shape == (32, 4098)


def test_short_conv_cross_correlation_matches_conv1d() -> None:
    implementation = _standalone_method("_short_conv_cross_correlation")
    torch.manual_seed(20260827)
    channels = 8
    kernel_size = 4
    dilation = 3
    effective_kernel_size = dilation * (kernel_size - 1) + 1
    weights = torch.randn(channels, kernel_size)

    for dtype in (torch.float32, torch.bfloat16):
        dtype_weights = weights.to(dtype)
        for output_len in (1, 4, 16, 17, 33):
            history_len = effective_kernel_size + output_len - 1
            history = torch.randn(2, channels, history_len).to(dtype)
            expected = F.conv1d(
                history.float(),
                dtype_weights.float().unsqueeze(1),
                groups=channels,
                dilation=dilation,
            ).to(dtype)
            actual = implementation(history, dtype_weights, dilation)
            torch.testing.assert_close(actual, expected)


def test_short_conv_prefill_chunks_bound_the_padded_window_peak() -> None:
    source = ast.unparse(_method_node(NVIDIA_PLE, "Qwen4ExpPLELayer", "_short_conv_cross_correlation"))
    assert "chunk_size = 16" in source
    assert "for output_start in range(0, output_len, chunk_size)" in source
    assert "output[..., output_start:output_end].copy_" in source

    # The failed 1.25-GiB allocation proves 16,384 padded row positions for
    # 10,240 PLE channels and four BF16 windows. The original run did not log
    # the separate num_prefills/max_len factors, so keep the byte assertion in
    # terms of their evidenced product. Use max_num_seqs only to bound a
    # single 16-position chunk below.
    padded_row_positions = 16_384
    max_num_prefills = 32
    channels = 10_240
    kernel_size = 4
    dtype_size = 2
    full_window_bytes = padded_row_positions * channels * kernel_size * dtype_size
    chunk_window_bytes = max_num_prefills * channels * 16 * kernel_size * dtype_size
    assert full_window_bytes == 1_342_177_280
    assert chunk_window_bytes == 41_943_040
    assert chunk_window_bytes * 32 == full_window_bytes


def test_ple_phase_masks_use_authoritative_request_phase_only() -> None:
    phase_masks = _standalone_function(SHORT_CONV_ATTN, "_ple_request_phase_masks")
    execution_masks = _standalone_function(
        SHORT_CONV_ATTN, "_ple_decode_execution_masks"
    )

    for decode_len in (1, 2, 3, 4):
        decode, prefill = phase_masks(torch.tensor([False]), 1)
        assert decode.tolist() == [True], f"D{decode_len} must stay decode"
        assert prefill.tolist() == [False]
        regular, variable = execution_masks(decode, torch.tensor([decode_len]))
        assert regular.tolist() == [decode_len == 1]
        assert variable.tolist() == [decode_len > 1]

    # Equal query lengths do not imply equal phases: P4 and D4 remain distinct.
    decode, prefill = phase_masks(torch.tensor([False, True]), 2)
    assert decode.tolist() == [True, False]
    assert prefill.tolist() == [False, True]

    # A one-token chunked prefill is still prefill.
    decode, prefill = phase_masks(torch.tensor([True]), 1)
    assert decode.tolist() == [False]
    assert prefill.tolist() == [True]


def test_ple_phase_masks_exclude_padded_capacity_rows() -> None:
    phase_masks = _standalone_function(SHORT_CONV_ATTN, "_ple_request_phase_masks")
    execution_masks = _standalone_function(
        SHORT_CONV_ATTN, "_ple_decode_execution_masks"
    )
    phase = torch.zeros(32, dtype=torch.bool)
    phase[:4] = torch.tensor([False, False, False, True])
    # Deliberately poison padded rows; they must not enter either mask.
    phase[4:] = True

    decode, prefill = phase_masks(phase, 4)
    assert decode.numel() == prefill.numel() == 4
    assert decode.tolist() == [True, True, True, False]
    assert prefill.tolist() == [False, False, False, True]

    query_lens = torch.tensor([4, 4, 4, 4084])
    regular_decode, variable_decode = execution_masks(decode, query_lens)
    assert int(regular_decode.sum()) == 0
    assert int(variable_decode.sum()) == 3
    assert int(prefill.sum()) == 1
    assert int(query_lens[prefill].max()) == 4084

    # One real prefill row, 10,240 channels, 4 BF16 windows.
    stack_bytes = 1 * 10_240 * 4084 * 4 * 2
    assert stack_bytes == 334_561_280
    assert stack_bytes / 2**20 == 319.0625


def test_ple_phase_masks_cover_all_prefill_and_all_decode_batches() -> None:
    phase_masks = _standalone_function(SHORT_CONV_ATTN, "_ple_request_phase_masks")
    for phase, expected_decode in (
        (torch.ones(4, dtype=torch.bool), 0),
        (torch.zeros(4, dtype=torch.bool), 4),
    ):
        decode, prefill = phase_masks(phase, 4)
        assert int(decode.sum()) == expected_decode
        assert int(prefill.sum()) == 4 - expected_decode
        assert torch.all(decode ^ prefill)


def test_ple_builder_does_not_classify_phase_from_query_length() -> None:
    source = ast.unparse(
        _method_node(
            SHORT_CONV_ATTN,
            "PleShortConvAttentionMetadataBuilder",
            "build",
        )
    )
    assert "decode_phase_mask_cpu, prefill_mask_cpu = _ple_request_phase_masks" in source
    assert "decode_mask_cpu, spec_sequence_masks_cpu = _ple_decode_execution_masks" in source
    assert "prefill_mask_cpu = non_spec_mask_cpu & (query_lens_cpu > 1)" not in source
    assert "decode_mask_cpu = non_spec_mask_cpu & (query_lens_cpu == 1)" not in source
    assert "num_reqs_actual" in source


def test_ple_capture_dummy_populates_mtp_request_metadata() -> None:
    source = ast.unparse(_method_node(MODEL_RUNNER, "NPUModelRunner", "_dummy_run"))
    assert "dummy_accepted_tokens = np.asarray(num_scheduled_tokens_list" in source
    assert "self.num_accepted_tokens.np[:num_reqs] = dummy_accepted_tokens" in source
    assert "self.num_decode_draft_tokens.np[:num_reqs] = dummy_accepted_tokens - 1" in source
    assert "use_spec_decode=dummy_use_spec_decode" in source


def test_conv_state_rows_use_contiguous_storage_layout() -> None:
    gather = _standalone_method("_gather_conv_state_rows")
    scatter = _standalone_method("_scatter_conv_state_rows")
    storage = torch.arange(4 * 7 * 5, dtype=torch.float32).reshape(4, 7, 5)
    conv_state = storage.transpose(-1, -2)
    indices = torch.tensor([3, 1])
    assert conv_state.stride() == (35, 1, 5)
    assert conv_state.transpose(-1, -2).is_contiguous()

    selected = gather(conv_state, indices)
    torch.testing.assert_close(selected, conv_state.index_select(0, indices))

    updated = selected + 1000
    scatter(conv_state, indices, updated)
    torch.testing.assert_close(conv_state.index_select(0, indices), updated)


def test_ple_state_shape_reserves_speculative_history() -> None:
    for path in (NVIDIA_MODEL, AMD_MODEL):
        source = ast.unparse(
            _method_node(
                path,
                "Qwen4ExpForCausalLM",
                "get_ple_mamba_state_shape_from_config",
            )
        )
        assert "vllm_config.num_speculative_tokens" in source

    for path in (NVIDIA_PLE, AMD_PLE):
        source = ast.unparse(_method_node(path, "Qwen4ExpPLELayer", "get_state_shape"))
        assert "self.num_spec_tokens" in source


def test_nvidia_model_does_not_compile_query_start_loc_as_dynamic() -> None:
    tree = ast.parse(NVIDIA_MODEL.read_text())
    model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Qwen4ExpModel")
    compile_decorator = next(
        decorator
        for decorator in model.decorator_list
        if isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Name)
        and decorator.func.id == "support_torch_compile"
    )
    dynamic_arg_dims = next(
        keyword.value for keyword in compile_decorator.keywords if keyword.arg == "dynamic_arg_dims"
    )
    assert isinstance(dynamic_arg_dims, ast.Dict)
    keys = {key.value for key in dynamic_arg_dims.keys if isinstance(key, ast.Constant) and isinstance(key.value, str)}
    assert "query_start_loc" not in keys
