# SPDX-License-Identifier: Apache-2.0
"""Execute compatibility adapters without importing NPU runtime modules."""

from __future__ import annotations

import ast
import unittest
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _method(path, class_name, method_name, namespace, *, bases=None):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    method.decorator_list = []
    if bases is None:
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias("annotations")], level=0), method],
            type_ignores=[],
        )
    else:
        cls.body = [method]
        cls.bases = [ast.Name(id="Parent", ctx=ast.Load())]
        cls.decorator_list = []
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias("annotations")], level=0), cls], type_ignores=[]
        )
        namespace["Parent"] = bases
    exec(compile(ast.fix_missing_locations(module), str(ROOT / path), "exec"), namespace)
    return namespace[method_name] if bases is None else namespace[class_name]


class ArrayBuffer:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.int64)

    def __getitem__(self, key):
        return ArrayBuffer(self.data[key])

    def numpy(self):
        return self.data

    def numel(self):
        return self.data.size


class TestMain2MainContracts(unittest.TestCase):
    def test_gemma4_copied_constructor_preserves_native_mm_initialization(self):
        for native_norm in (False, True):
            for mode in (None, "missing", "weights", "data"):
                with self.subTest(native_norm=native_norm, mode=mode):
                    mm_config = (
                        None
                        if mode is None
                        else SimpleNamespace(**({} if mode == "missing" else {"mm_encoder_tp_mode": mode}))
                    )
                    model_config = SimpleNamespace(
                        hf_config=SimpleNamespace(
                            vision_config=object(),
                            audio_config=None,
                            text_config=SimpleNamespace(hidden_size_per_layer_input=None),
                        ),
                        multimodal_config=mm_config,
                        dtype="bf16",
                        try_get_generation_config=lambda: {"suppress_tokens": [7]},
                    )
                    config = SimpleNamespace(model_config=model_config, quant_config=object(), lora_config=None)
                    tower = SimpleNamespace(
                        patch_embedder=SimpleNamespace(input_proj=SimpleNamespace(quant_method=None))
                    )
                    norm = object()
                    build_norm = MagicMock(return_value=norm)
                    upstream = SimpleNamespace(build_mm_input_norm=build_norm) if native_norm else SimpleNamespace()
                    language_model = MagicMock()
                    replace_linear = MagicMock()
                    namespace: dict[str, Any] = dict(
                        torch=SimpleNamespace(nn=SimpleNamespace(Module=object)),
                        gemma4_mm=upstream,
                        vllm_version_is=lambda _, native_norm=native_norm: not native_norm,
                        AutoModel=SimpleNamespace(from_config=MagicMock(return_value=tower)),
                        Gemma4MultimodalEmbedder=MagicMock(),
                        recursive_replace_linear=replace_linear,
                        maybe_prefix=lambda prefix, name: f"{prefix}.{name}",
                        init_vllm_registered_model=MagicMock(return_value=language_model),
                        AscendW4A4MXFP4DynamicLinearMethod=type("MXFP4", (), {}),
                        AscendW8A8MXFP8DynamicLinearMethod=type("MXFP8", (), {}),
                    )
                    initialize = _method(
                        "vllm_ascend/models/gemma4_mm.py",
                        "AscendGemma4ForConditionalGeneration",
                        "__init__",
                        namespace,
                    )
                    model = SimpleNamespace(
                        _mark_tower_model=lambda *_args: nullcontext(),
                        _mark_language_model=lambda *_args: nullcontext(),
                    )
                    initialize(model, vllm_config=config, prefix="model")
                    self.assertEqual(model.use_data_parallel, mode == "data")
                    if native_norm:
                        build_norm.assert_called_once_with(model_config)
                        self.assertIs(model.vision_input_norm, norm)
                    else:
                        build_norm.assert_not_called()
                        self.assertFalse(hasattr(model, "vision_input_norm"))
                    self.assertIs(model.vision_tower, tower)
                    replace_linear.assert_called_once_with(tower, config.quant_config, prefix="model.vision_tower")
                    self.assertIs(model.language_model, language_model)
                    self.assertIsNone(model.audio_tower)
                    self.assertIsNone(model.per_layer_embeddings)
                    self.assertFalse(model._enable_mm_lora)
                    self.assertEqual(model._suppress_token_ids, [7])

    def test_split_graph_explicit_values_preserve_default_cap_semantics(self):
        fields = (
            "prefill_cudagraph_capture_token_sizes",
            "max_prefill_cudagraph_capture_tokens",
            "decode_cudagraph_capture_request_sizes",
            "max_decode_cudagraph_capture_requests",
            "max_decode_cudagraph_capture_tokens",
        )
        for release in (False, True):
            namespace = {"vllm_version_is": lambda _, release=release: release}
            tree = ast.parse((ROOT / "vllm_ascend/platform.py").read_text(encoding="utf-8"))
            fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_get_reduced_cg_cap")
            module = ast.Module(
                body=[ast.ImportFrom(module="__future__", names=[ast.alias("annotations")], level=0), fn],
                type_ignores=[],
            )
            exec(compile(ast.fix_missing_locations(module), "<actual-default-cap>", "exec"), namespace)
            compilation = SimpleNamespace(max_cudagraph_capture_size=None, cudagraph_capture_sizes=None)
            if not release:
                for name in fields:
                    setattr(compilation, name, None)
            config = SimpleNamespace(
                compilation_config=compilation,
                scheduler_config=SimpleNamespace(max_num_seqs=7),
                speculative_config=SimpleNamespace(num_speculative_tokens=1),
            )
            cap = namespace[fn.name]
            self.assertEqual(cap(config), 14)
            if not release:
                for name in fields:
                    # Empty lists and zero caps are explicit settings, not None.
                    values = ([], [1, 3, 7]) if name.endswith("sizes") else (0, 7)
                    for value in values:
                        with self.subTest(name=name, value=value):
                            setattr(compilation, name, value)
                            self.assertIsNone(cap(config))
                            self.assertEqual(getattr(compilation, name), value)
                            setattr(compilation, name, None)
            compilation.cudagraph_capture_sizes = []
            self.assertIsNone(cap(config))

    def test_mamba_precopy_temporal_bias_preserves_conv_and_legacy_calls(self):
        path = ROOT / "vllm_ascend/ops/triton/mamba/precopy.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        kernel = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "precopy_mamba_align_fused_kernel"
        )
        kernel.decorator_list = []
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias("annotations")], level=0), kernel],
            type_ignores=[],
        )
        for zero_temporal in (None, False, True):
            for conv_width in (0, 4):
                for mapped in (False, True):
                    with self.subTest(zero_temporal=zero_temporal, conv_width=conv_width, mapped=mapped):
                        req_idx = 2 if mapped else 0
                        memory = {100 + req_idx: 5, 200 + req_idx: 3, 300 + req_idx: 2, 900: conv_width, 1300: 2}
                        copy = MagicMock()
                        namespace: dict[str, Any] = dict(
                            tl=SimpleNamespace(
                                program_id=lambda _axis: np.int64(0),
                                load=lambda address, memory=memory: np.int64(memory[int(address)]),
                                where=np.where,
                            ),
                            _copy_mamba_state_block=copy,
                        )
                        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
                        kwargs = dict(COPY_BLOCK_SIZE=1024, CONV_STATE_DIM_FIRST=True, HAS_IDX_MAPPING=mapped)
                        if zero_temporal is not None:
                            kwargs["ZERO_TEMPORAL_TOKEN_BIAS"] = zero_temporal
                        namespace["precopy_mamba_align_fused_kernel"](
                            100, 200, 300, 400, 32, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1, **kwargs
                        )
                        copy.assert_called_once()
                        args = copy.call_args.args
                        self.assertEqual(args[:4], (0, 0, 3, 5))
                        self.assertEqual(args[4], 0 if zero_temporal and conv_width == 0 else 2)
                        self.assertEqual(args[5:15], (400, 32, 500, 600, 700, 800, 900, 1000, 1100, 1200))
                        self.assertEqual(args[16:], (1024, True, 1))

    def test_replayssm_temporal_guard_retains_conv_copy_and_count_normalization(self):
        path = ROOT / "vllm_ascend/ops/triton/mamba/postprocess.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        kernel = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "postprocess_mamba_fused_kernel"
        )
        # Execute the actual launch-independent tail between address calculation
        # and the memory transfer. Triton compilation/copies remain hardware CI.
        start = next(
            i for i, n in enumerate(kernel.body) if isinstance(n, ast.If) and "state_idx == 0" in ast.unparse(n.test)
        )
        end = next(
            i
            for i, n in enumerate(kernel.body)
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "offsets" for t in n.targets)
        )
        # Parsing supplies version-specific AST fields, including type_params.
        fragment = ast.parse("def before_copy(): pass").body[0]
        assert isinstance(fragment, ast.FunctionDef)
        fragment.body = kernel.body[start : end + 1]
        module = compile(ast.fix_missing_locations(ast.Module(body=[fragment], type_ignores=[])), str(path), "exec")
        for skip_temporal in (False, True):
            for conv_width in (0, 4):
                with self.subTest(skip_temporal=skip_temporal, conv_width=conv_width):
                    output = [2]
                    transfers = MagicMock()

                    def store(_ptr, value, output=output):
                        output[0] = value

                    ns: dict[str, Any] = dict(
                        src_block_idx=1,
                        dest_block_idx=1,
                        state_idx=0,
                        tile_idx=0,
                        accept_token_bias=1,
                        num_accepted_tokens_out_ptr=0,
                        num_accepted_tokens_ptr=0,
                        req_idx=0,
                        conv_width=conv_width,
                        SKIP_TEMPORAL_STATE_COPY=skip_temporal,
                        COPY_BLOCK_SIZE=1024,
                        tl=SimpleNamespace(store=store, arange=transfers),
                    )
                    exec(module, ns)
                    ns["before_copy"]()
                    self.assertEqual(output, [1])
                    if skip_temporal and conv_width == 0:
                        transfers.assert_not_called()
                    else:
                        transfers.assert_called_once_with(0, 1024)

    def test_310p_initialization_forwards_versioned_graph_shapes(self):
        path = ROOT / "vllm_ascend/_310p/worker/v2/model_runner.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assignment = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Name)
            and n.value.func.id == "ModelAclGraphManager310"
        )
        module = ast.fix_missing_locations(ast.Module(body=[assignment], type_ignores=[]))
        for release in (False, True):
            for query_len in (1, 4):
                with self.subTest(release=release, query_len=query_len):
                    shapes = {query_len: 8, query_len + 1: 4}

                    def construct(
                        config,
                        device,
                        mode,
                        query_shapes,
                        runner,
                        *,
                        release=release,
                        query_len=query_len,
                        shapes=shapes,
                        **kwargs,
                    ):
                        if release:
                            self.assertEqual(query_shapes, query_len)
                        else:
                            # The new upstream initializer requires this mapping.
                            self.assertEqual(dict(query_shapes.items()), shapes)
                        self.assertEqual(kwargs["lora_capture_cases"], [0, 2])
                        return object()

                    config = SimpleNamespace(get_decode_cudagraph_query_lens=MagicMock(return_value=shapes))
                    runner = SimpleNamespace(
                        vllm_config=config, device=object(), decode_query_len=query_len, lora_capture_cases=[0, 2]
                    )
                    exec(
                        compile(module, str(path), "exec"),
                        {
                            "self": runner,
                            "cudagraph_mode": object(),
                            "ModelAclGraphManager310": construct,
                            "vllm_version_is": lambda _, release=release: release,
                        },
                    )
                    if release:
                        config.get_decode_cudagraph_query_lens.assert_not_called()
                    else:
                        config.get_decode_cudagraph_query_lens.assert_called_once_with(query_len)

    def test_graph_mapping_preserves_request_limits_and_decode_sizes(self):
        path = "vllm_ascend/worker/v2/spec_decode/target_dependent_ar/aclgraph.py"
        for release in (False, True):
            with self.subTest(release=release):
                calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

                class Parent:
                    def __init__(self, *args, _calls=calls, **kwargs):
                        _calls.append((args, kwargs))
                        self._capture_descs = {}

                    def needs_capture(self):
                        return False

                namespace = {
                    "Mapping": Mapping,
                    "CUDAGraphMode": SimpleNamespace(FULL="full"),
                    "is_drafter_moe_model": lambda _: False,
                    "vllm_version_is": lambda _, release=release: release,
                    "get_draft_prefill_cudagraph_mode": lambda config, mode: mode,
                    "collect_sorted_captured_token_sizes": lambda _: [],
                }
                cls = _method(path, "AutoRegressiveAclGraphManager", "__init__", namespace, bases=Parent)
                mode = SimpleNamespace(decode_mode=lambda: "full")
                shape = 3 if release else {2: 8, 3: 4}
                manager = cls(object(), object(), mode, shape)
                self.assertTrue(manager.is_draft_model_prefill)
                self.assertEqual(calls[0][0][3], shape)
                if not release:
                    calls.clear()
                    manager = cls(object(), object(), mode, decode_query_lens={1: 12}, decode_request_sizes=[1, 4, 12])
                    self.assertFalse(manager.is_draft_model_prefill)
                    self.assertEqual(calls[0][1]["decode_request_sizes"], [1, 4, 12])

    def test_dflash_accepts_new_keyword_and_release_scalar(self):
        for release in (False, True):
            calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

            class Parent:
                def __init__(self, *args, _calls=calls, **kwargs):
                    _calls.append((args, kwargs))
                    self._capture_descs = {}

                def needs_capture(self):
                    return False

            cls = _method(
                "vllm_ascend/worker/v2/spec_decode/dflash/aclgraph.py",
                "DFlashAclGraphManager",
                "__init__",
                {
                    "vllm_version_is": lambda _, release=release: release,
                    "collect_sorted_captured_token_sizes": lambda _: [],
                },
                bases=Parent,
            )
            kwargs: dict[str, object] = {"decode_query_len": 2} if release else {"decode_query_lens": {2: 8}}
            cls(object(), object(), object(), **kwargs)
            self.assertEqual(calls[0][0][3], 2 if release else {2: 8})

    def test_dsa_post_load_accepts_zero_arguments_and_legacy_dtype(self):
        fn = _method("vllm_ascend/models/layer/attention/layer.py", "DSAAttention", "process_weights_after_loading", {})
        layer = SimpleNamespace(act_dtype="bf16", impl=SimpleNamespace(process_weights_after_loading=MagicMock()))
        fn(layer)
        layer.impl.process_weights_after_loading.assert_called_once_with("bf16")
        layer.impl.process_weights_after_loading.reset_mock()
        fn(layer, "fp16")
        layer.impl.process_weights_after_loading.assert_called_once_with("fp16")

    def test_vl_wrapper_does_not_finalize_child_twice_on_main(self):
        for model, cls in [
            ("deepseek_v4", "AscendDeepseekV4ForConditionalGeneration"),
            ("deepseek_v41", "AscendDeepseekV41ForCausalLM"),
        ]:
            for release in (False, True):
                hook = MagicMock()
                fn = _method(
                    f"vllm_ascend/models/{model}/vl_model.py",
                    cls,
                    "process_weights_after_loading",
                    {"vllm_version_is": lambda _, release=release: release},
                )
                fn(SimpleNamespace(language_model=SimpleNamespace(process_weights_after_loading=hook)))
                self.assertEqual(hook.call_count, int(release))

    def test_pointer_descriptors_handle_empty_groups_partial_chunks_and_directions(self):
        # Device addresses may no longer be reconstructed from an NPU block ID.
        # Two refs in group 0, no blocks in group 1, one ref in group 2.
        for store in (False, True):

            def compute_ptrs(ids, chunk, out, tensor, skip_count=0):
                out[:] = [
                    tensor.base
                    + int(ids[(i + skip_count) // chunk]) * tensor.stride
                    + ((i + skip_count) % chunk) * tensor.stride // chunk
                    for i in range(len(out))
                ]

            class BlockSpec:
                block_ids = np.array([2, 4, 7])

            namespace = {
                "BlockIDsLoadStoreSpec": BlockSpec,
                "cdiv": lambda x, y: (x + y - 1) // y,
                "compute_sub_block_ptrs": compute_ptrs,
                "_new_descriptor_buffers": lambda n: tuple(ArrayBuffer(np.zeros(n)) for _ in range(3)),
            }
            fn = _method(
                "vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/native/cpu_npu.py",
                "SingleDirectionNPUOffloadingHandler",
                "_pointer_descriptors",
                namespace,
            )
            refs = [[SimpleNamespace(tensor_idx=0), SimpleNamespace(tensor_idx=1)], [], [SimpleNamespace(tensor_idx=2)]]
            tensors = [
                SimpleNamespace(base=1000, stride=64),
                SimpleNamespace(base=2000, stride=128),
                SimpleNamespace(base=3000, stride=32),
            ]
            handler = SimpleNamespace(
                npu_to_cpu=store,
                _buffer_pool=[],
                src_blocks_per_chunk=1 if store else 2,
                dst_blocks_per_chunk=2 if store else 1,
                src_tensors=tensors,
                dst_tensors=tensors,
                kv_cache_groups_data_refs=refs,
            )
            device = SimpleNamespace(
                ptrs=np.array([11, 29, 47, 101, 131, 167, 199], dtype=np.uint64),
                sizes=np.array([16, 16, 16, 32, 32, 32, 8], dtype=np.uint64),
                group_block_counts=(3, 0, 1),
                group_data_ref_counts=(2, 0, 1),
                block_indices=(1, 0, 0),
            )
            src, dst, sizes, count, num_bytes = fn(handler, device, BlockSpec())
            device_out, cpu_out = (src, dst) if store else (dst, src)
            np.testing.assert_array_equal(device_out.numpy(), device.ptrs)
            np.testing.assert_array_equal(cpu_out.numpy(), [1160, 1256, 1288, 2320, 2512, 2576, 3224])
            np.testing.assert_array_equal(sizes.numpy(), device.sizes)
            self.assertEqual(count, 7)
            self.assertEqual(num_bytes, 152)


if __name__ == "__main__":
    unittest.main()
