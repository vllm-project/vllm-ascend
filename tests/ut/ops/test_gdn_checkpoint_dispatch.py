# SPDX-License-Identifier: Apache-2.0
"""Exercise the real dispatch preamble and workspace owner without NPU imports."""

import ast
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[3]


def function_from_source(path, name, scope, stop_at=None):
    tree = ast.parse(path.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    node.returns = None
    for arg in node.args.args + node.args.kwonlyargs:
        arg.annotation = None
    if stop_at is not None:
        end = next(
            i
            for i, n in enumerate(node.body)
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == stop_at for t in n.targets)
        )
        node.body = node.body[:end] + [ast.Return(value=ast.Constant(value=None))]
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, str(path), "exec"), scope)
    return scope[name]


class CheckpointDispatchTest(unittest.TestCase):
    def test_checkpoint_policy_excludes_unaligned_scheduler_and_worker_modes(self):
        supports = function_from_source(
            ROOT / "vllm_ascend/ops/gdn_checkpoint.py", "_supports_checkpoint_boundary_policy", {}
        )
        for transfer, spec_drop, lookahead, expected in (
            (False, None, 0, True),
            (False, False, 0, True),
            (False, True, 1, True),
            (False, True, 0, False),
            (True, None, 0, False),
            (True, True, 1, False),
        ):
            with self.subTest(transfer=transfer, spec_drop=spec_drop, lookahead=lookahead):
                config = SimpleNamespace(
                    kv_transfer_config=object() if transfer else None,
                    speculative_config=None
                    if spec_drop is None
                    else SimpleNamespace(use_eagle_block_drop=lambda spec_drop=spec_drop: spec_drop),
                    num_hash_lookahead_tokens=lookahead,
                )
                self.assertIs(supports(config), expected)

    def test_checkpoint_entrypoint_requires_complete_device_metadata(self):
        entry = function_from_source(
            ROOT / "vllm_ascend/ops/triton/fla/chunk.py",
            "chunk_gated_delta_rule_checkpoint",
            {"torch": SimpleNamespace(float32="fp32")},
            stop_at="q",
        )
        fields = {
            name: object()
            for name in (
                "chunk_indices_chunk64",
                "chunk_offsets_chunk64",
                "chunk_indices_large_block",
                "block_indices_cumsum",
            )
        }
        for missing in [None, *fields]:
            metadata = None if missing is None else SimpleNamespace(**{**fields, missing: None})
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                entry(None, None, None, None, None, None, None, metadata, None, None)
        entry(None, None, None, None, None, None, None, SimpleNamespace(**fields), None, None)

    def test_checkpoint_entry_normalizes_read_inputs_and_preserves_output_buffers(self):
        class Tensor:
            shape = (1, 192, 4, 128)

            def __init__(self, contiguous=False):
                self.is_contiguous = contiguous

            def contiguous(self):
                return self if self.is_contiguous else Tensor(True)

        def normalize(tensor):
            self.assertTrue(tensor.is_contiguous)
            return tensor

        states, offsets, cu_seqlens = Tensor(True), Tensor(), Tensor()
        output, final_state = object(), object()

        def forward(q, k, v, g, beta, scale, initial_state, output_final_state, **kwargs):
            self.assertTrue(kwargs["use_triton_recurrence"])
            for tensor in (q, k, v, g, beta, initial_state, kwargs["cu_seqlens"], kwargs["checkpoint_offsets"]):
                self.assertTrue(tensor.is_contiguous)
            self.assertIs(kwargs["checkpoint_output"], states)
            return None, output, None, final_state, None, None, None

        entry = function_from_source(
            ROOT / "vllm_ascend/ops/triton/fla/chunk.py",
            "chunk_gated_delta_rule_checkpoint",
            {"l2norm_fwd": normalize, "chunk_gated_delta_rule_fwd": forward, "torch": SimpleNamespace(float32="fp32")},
        )
        metadata = SimpleNamespace(
            **{
                name: object()
                for name in (
                    "chunk_indices_chunk64",
                    "chunk_offsets_chunk64",
                    "chunk_indices_large_block",
                    "block_indices_cumsum",
                )
            }
        )
        result = entry(*(Tensor() for _ in range(6)), cu_seqlens, metadata, states, offsets)
        self.assertEqual(result, (output, final_state))

    def test_checkpoint_skips_ascendc_w_u_copies_and_preserves_kernel_inputs(self):
        class Tensor:
            dtype = "bf16"

            def __init__(self):
                self.conversions = 0

            def to(self, dtype):
                self.conversions += 1
                return self

            def transpose(self, *dims):
                return self

            def contiguous(self):
                return self

            def __getitem__(self, key):
                return self

            def clone(self):
                raise AssertionError("Uncompacted Triton states must not be scattered again")

        for checkpoint, force_triton in ((False, False), (True, False), (False, True)):
            with self.subTest(checkpoint=checkpoint, force_triton=force_triton):
                uses_triton = checkpoint or force_triton
                q, k, v, g, beta, initial = (Tensor() for _ in range(6))
                w, u, h, v_new, final, output = (Tensor() for _ in range(6))
                states, offsets = object(), object()

                triton_kernel = Mock(return_value=(h, v_new, final))
                ascendc_kernel = Mock(return_value=(h, v_new, final))
                forward = function_from_source(
                    ROOT / "vllm_ascend/ops/triton/fla/chunk.py",
                    "chunk_gated_delta_rule_fwd",
                    {
                        "torch": SimpleNamespace(
                            float32="fp32",
                            bfloat16="bf16",
                            int64="i64",
                            ops=SimpleNamespace(
                                _C_ascend=SimpleNamespace(
                                    chunk_gated_delta_rule_fwd_h_vllm=ascendc_kernel,
                                    chunk_fwd_o_vllm=Mock(return_value=output),
                                )
                            ),
                        ),
                        "get_forward_context": lambda: SimpleNamespace(attn_metadata=None),
                        "get_pcp_group": lambda: SimpleNamespace(world_size=1),
                        "chunk_local_cumsum": lambda tensor, **kwargs: tensor,
                        "chunk_scaled_dot_kkt_fwd": lambda **kwargs: object(),
                        "solve_tril": lambda **kwargs: kwargs["A"],
                        "recompute_w_u_fwd": Mock(return_value=(w, u)),
                        "chunk_gated_delta_rule_fwd_h": triton_kernel,
                        "SUPPRESS_LEVEL": 0,
                    },
                )
                metadata = None
                if force_triton:
                    # Triton already preserves the layout of zero-length rows.
                    metadata = SimpleNamespace(
                        **{
                            name: None
                            for name in (
                                "block_indices_cumsum",
                                "cu_seqlens_host",
                                "chunk_indices_chunk64",
                                "chunk_indices_chunk64_host",
                                "chunk_offsets_chunk64",
                                "update_chunk_offsets_chunk64",
                                "final_chunk_indices_chunk64",
                                "chunk_indices_large_block",
                                "cu_seqlens_kern",
                            )
                        },
                        keep_meta=object(),
                    )
                result = forward(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    0.1,
                    initial,
                    True,
                    checkpoint_output=states if checkpoint else None,
                    checkpoint_offsets=offsets if checkpoint else None,
                    use_triton_recurrence=force_triton,
                    prebuilt_meta=metadata,
                )
                self.assertEqual((w.conversions, u.conversions), (0, 0) if uses_triton else (1, 1))
                self.assertEqual(triton_kernel.call_count, int(uses_triton))
                self.assertEqual(ascendc_kernel.call_count, int(not uses_triton))
                if uses_triton:
                    arguments = triton_kernel.call_args.kwargs
                    self.assertIs(arguments["w"], w)
                    self.assertIs(arguments["u"], u)
                    self.assertIs(arguments["initial_state"], initial)
                    self.assertIs(arguments["checkpoint_output"], states if checkpoint else None)
                    self.assertIs(arguments["checkpoint_offsets"], offsets if checkpoint else None)
                else:
                    self.assertIs(ascendc_kernel.call_args.args[1], w)
                    self.assertIs(ascendc_kernel.call_args.args[2], u)
                self.assertIs(result[1], output)
                self.assertIs(result[3], final)

    def test_checkpoint_prefills_bypass_fused_metadata_but_decode_keeps_fast_path(self):
        for enabled, num_prefills, checkpoint, expected_calls in (
            (False, 1, None, 1),
            (True, 0, None, 1),
            (True, 1, None, 0),
            (True, 1, object(), 0),
        ):
            with self.subTest(num_prefills=num_prefills, checkpoint=checkpoint):
                sentinel = object()
                builder = SimpleNamespace(
                    use_full_cuda_graph=False,
                    use_spec_decode=False,
                    _ascend_gdn_checkpoints=enabled,
                    _build_fused_metadata=Mock(return_value=sentinel),
                )
                module = ModuleType("vllm_ascend.ops.gdn_checkpoint")
                build_checkpoints = Mock(return_value=checkpoint)
                module.__dict__["build_gdn_checkpoints"] = build_checkpoints
                build = function_from_source(
                    ROOT / "vllm_ascend/ops/gdn_attn_builder.py",
                    "build",
                    {
                        "_treat_single_token_prefills_with_state_as_decodes": lambda m: m,
                        "_split_gdn_batch": lambda *args, num_prefills=num_prefills: (
                            1 - num_prefills,
                            num_prefills,
                            1,
                            2,
                            0,
                            None,
                        ),
                    },
                    stop_at="query_start_loc",
                )
                with patch.dict(sys.modules, {module.__name__: module}):
                    result = build(builder, 0, SimpleNamespace(num_reqs=1), num_actual_reqs=1)
                self.assertEqual(builder._build_fused_metadata.call_count, expected_calls)
                self.assertIs(result, sentinel if expected_calls else None)
                self.assertEqual(build_checkpoints.call_count, int(enabled and bool(num_prefills)))

    def test_cold_and_cache_hit_prefill_share_backend_and_cache_dtype(self):
        class Tensor:
            shape = (1, 4, 128, 128)
            dtype = "bf16"

            def transpose(self, *args):
                return self

            def contiguous(self):
                return self

            def __getitem__(self, key):
                return self

        run = function_from_source(
            ROOT / "vllm_ascend/ops/gdn_checkpoint.py",
            "run_gdn_checkpoint_prefill",
            {},
        )
        for creates_checkpoint in (False, True):
            with self.subTest(creates_checkpoint=creates_checkpoint):
                output, final = Tensor(), Tensor()
                backend = Mock(return_value=(output, final))
                run.__globals__["chunk_gated_delta_rule_checkpoint"] = backend
                workspace = Tensor()
                exporter = Mock()
                layer = SimpleNamespace(
                    _ascend_checkpoint_workspace=workspace, kv_cache=(Tensor(), Tensor()), _checkpoint_exporter=exporter
                )
                checkpoint = SimpleNamespace(checkpoint_offsets=object()) if creates_checkpoint else None
                metadata = SimpleNamespace(
                    ascend_checkpoint=checkpoint, non_spec_prefill_metadata=SimpleNamespace(chunk=object())
                )
                result = run(layer, metadata, Tensor(), *(Tensor() for _ in range(7)))
                self.assertEqual(result, (output, final))
                self.assertEqual(backend.call_args.kwargs["checkpoint_state_dtype"], "bf16")
                self.assertIs(backend.call_args.args[8], workspace if creates_checkpoint else None)
                self.assertIs(
                    backend.call_args.args[9], checkpoint.checkpoint_offsets if checkpoint is not None else None
                )
                self.assertEqual(exporter.export.call_count, int(creates_checkpoint))

    def test_checkpoint_workspace_is_shared_only_for_matching_device_and_shape(self):
        context: dict[str, SimpleNamespace] = {}
        config = SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_seqs=32),
            compilation_config=SimpleNamespace(static_forward_context=context),
        )

        class Tensor:
            device = "npu:0"

            def new_empty(self, shape, dtype):
                return SimpleNamespace(shape=shape, dtype=dtype, device=self.device)

        def layer(heads=4):
            return SimpleNamespace(
                _exports_checkpoints=True,
                _ascend_checkpoint_workspace=None,
                vllm_config=config,
                num_v_heads=heads,
                tp_size=1,
                head_v_dim=128,
                head_k_dim=128,
            )

        ensure = function_from_source(
            ROOT / "vllm_ascend/ops/gdn_checkpoint.py",
            "ensure_gdn_checkpoint_workspace",
            {"torch": SimpleNamespace(float32="fp32")},
        )
        first, second, different = layer(), layer(), layer(8)
        context.update(first=first, second=second, different=different)
        tensor = Tensor()
        ensure(first, tensor)
        ensure(second, tensor)
        ensure(different, tensor)
        self.assertIs(first._ascend_checkpoint_workspace, second._ascend_checkpoint_workspace)
        self.assertIsNot(first._ascend_checkpoint_workspace, different._ascend_checkpoint_workspace)
        other_config = layer()
        other_config.vllm_config = SimpleNamespace(
            scheduler_config=config.scheduler_config,
            compilation_config=config.compilation_config,
        )
        context["other_config"] = other_config
        ensure(other_config, tensor)
        self.assertIsNot(first._ascend_checkpoint_workspace, other_config._ascend_checkpoint_workspace)
        other_device = layer()
        context["other"] = other_device
        tensor.device = "npu:1"
        ensure(other_device, tensor)
        self.assertIsNot(first._ascend_checkpoint_workspace, other_device._ascend_checkpoint_workspace)


if __name__ == "__main__":
    unittest.main()
