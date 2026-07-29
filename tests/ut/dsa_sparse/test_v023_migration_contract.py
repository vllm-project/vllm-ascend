# SPDX-License-Identifier: Apache-2.0
"""Dependency-free source contracts for the v0.23 DSA migration.

These tests intentionally use only the Python standard library.  They can run
on a development host without torch, vLLM, CANN, or an Ascend device and catch
the migration regressions that are otherwise easy to introduce while rebasing
the feature onto a newer framework.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
import types
import unittest
from dataclasses import dataclass, fields
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[3]


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def _load_lightweight_types_module():
    path = REPO_ROOT / "vllm_ascend/dsa_sparse/dsa_types.py"
    spec = importlib.util.spec_from_file_location(
        "_dsa_types_contract",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_build_jobs_module():
    path = REPO_ROOT / "vllm_ascend/build_jobs.py"
    spec = importlib.util.spec_from_file_location(
        "_build_jobs_contract",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_engine_process_entrypoint_contract(engine_core_proc):
    path = (
        REPO_ROOT
        / "vllm_ascend/patch/dsa_sparse/patch_engine_process.py"
    )
    source_module = ast.parse(path.read_text(encoding="utf-8"))
    function_names = {
        "_install_dsa_runtime_patches",
        "is_dsa_run_engine_core_wrapper",
        "_dsa_sparse_run_engine_core",
        "ensure_dsa_engine_core_entrypoint",
    }
    selected_nodes = []
    for node in source_module.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "_DSA_RUN_ENGINE_CORE_WRAPPER_ATTR"
                for target in node.targets
            )
        ):
            selected_nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in function_names:
            selected_nodes.append(node)
        elif (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "setattr"
        ):
            selected_nodes.append(node)

    contract_module = ast.Module(body=selected_nodes, type_ignores=[])
    ast.fix_missing_locations(contract_module)
    namespace = {
        "EngineCoreProc": engine_core_proc,
        "_reattach_dsa_config_from_additional_config": lambda kwargs: None,
        "_is_dsa_enabled_on_config": lambda config: True,
        "verify_dsa_runtime_patches_installed": lambda: None,
    }
    exec(compile(contract_module, str(path), "exec"), namespace)
    return namespace


def _function_node(relative_path: str, function_name: str) -> ast.FunctionDef:
    module = ast.parse(_read(relative_path))
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return node
    raise AssertionError(
        f"Function {function_name!r} not found in {relative_path}"
    )


def _load_indexer_merge_contract():
    path = REPO_ROOT / "vllm_ascend/core/kv_cache_interface.py"
    module = ast.parse(path.read_text(encoding="utf-8"))
    merge_node = None
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "IndexerKVSpec":
            merge_node = next(
                child
                for child in node.body
                if (
                    isinstance(child, ast.FunctionDef)
                    and child.name == "merge"
                )
            )
            break
    assert merge_node is not None
    merge_node.decorator_list = []
    contract_module = ast.Module(body=[merge_node], type_ignores=[])
    ast.fix_missing_locations(contract_module)

    @dataclass(frozen=True, kw_only=True)
    class FakeAttentionSpec:
        block_size: int
        num_kv_heads: int
        head_size: int
        dtype: str
        page_size_padded: int | None = None

    @dataclass(frozen=True, kw_only=True)
    class FakeIndexerKVSpec(FakeAttentionSpec):
        pass

    namespace = {
        "AttentionSpec": FakeAttentionSpec,
        "Self": object,
        "fields": fields,
    }
    exec(compile(contract_module, str(path), "exec"), namespace)
    return namespace["merge"], FakeAttentionSpec, FakeIndexerKVSpec


def _load_sfa_prefill_layer_resolver_contract():
    path = REPO_ROOT / "vllm_ascend/attention/sfa_v1.py"
    module = ast.parse(path.read_text(encoding="utf-8"))
    resolver_node = next(
        node
        for node in module.body
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "_resolve_sfa_prefill_layer_names"
        )
    )
    resolver_node.decorator_list = []
    contract_module = ast.Module(
        body=[resolver_node],
        type_ignores=[],
    )
    ast.fix_missing_locations(contract_module)

    class FakeAttentionSpec:
        pass

    class FakeIndexerKVSpec(FakeAttentionSpec):
        pass

    namespace = {
        "AttentionSpec": FakeAttentionSpec,
        "IndexerKVSpec": FakeIndexerKVSpec,
        "VllmConfig": object,
    }
    exec(compile(contract_module, str(path), "exec"), namespace)
    return namespace["_resolve_sfa_prefill_layer_names"], FakeIndexerKVSpec


class TestOperatorABI(unittest.TestCase):
    def test_acl_operator_names_are_unchanged(self):
        ksc = _read(
            "csrc/attention/kvcache_scatter_copy/"
            "op_host/kvcache_scatter_copy_def.cpp"
        )
        lidu = _read(
            "csrc/attention/lightning_indexer_decode_update/"
            "op_host/lightning_indexer_decode_update_def.cpp"
        )
        self.assertIn("class KvcacheScatterCopy", ksc)
        self.assertIn("OP_ADD(KvcacheScatterCopy)", ksc)
        self.assertIn("class LightningIndexerDecodeUpdate", lidu)
        self.assertIn("OP_ADD(LightningIndexerDecodeUpdate)", lidu)

    def test_torch_method_names_and_argument_order_are_stable(self):
        cpp = re.sub(
            r'[\s"]+',
            "",
            _read("csrc/torch_binding.cpp"),
        )
        py = re.sub(
            r'[\s"]+',
            "",
            _read("vllm_ascend/dsa_sparse/dsa_ascend_ops_backend.py"),
        )
        lidu_schema = (
            "npu_lightning_indexer_decode_update_out("
            "Tensorquery,Tensorkey,Tensorweights,Tensorreq_pool_entries,"
            "Tensor(a!)cache_slots,Tensorrow_modes,"
            "Tensoractual_seq_lengths_key,Tensorblock_table,"
            "Tensor(b!)topk_index_out,Tensor(c!)topk_slots_out,"
            "Tensor(d!)miss_count_out,Tensor(e!)tail_info_out)->()"
        )
        ksc_schema = (
            "npu_kvcache_scatter_copy(Tensor(a!)hbm_k_rope,"
            "Tensor(b!)hbm_kv_cache,Tensordram_k_rope,"
            "Tensordram_kv_cache,Tensorhbm_block_table,"
            "Tensordram_block_table,Tensorsrc_token_ids,"
            "Tensordst_slots,Tensorcopy_counts)->()"
        )
        for source in (cpp, py):
            self.assertIn(lidu_schema, source)
            self.assertIn(ksc_schema, source)

    def test_a2_a3_build_fused_ops_and_a5_uses_fallback(self):
        build_script = _read("csrc/build_aclnn.sh")
        a2 = build_script[
            build_script.index('^ascend910b'):
            build_script.index('^ascend910_93')
        ]
        a3 = build_script[
            build_script.index('^ascend910_93'):
            build_script.index('^ascend950')
        ]
        a5 = build_script[build_script.index('^ascend950'):]
        operators = (
            "lightning_indexer_decode_update",
            "kvcache_scatter_copy",
            "sparse_flash_attention_for_offload",
            "kv_cache_full_block_dump",
        )
        for operator in operators:
            self.assertIn(f'"{operator}"', a2)
            self.assertIn(f'"{operator}"', a3)
            self.assertNotIn(f'"{operator}"', a5)

        cmake = _read("CMakeLists.txt")
        self.assertIn("VLLM_ASCEND_DSA_A5_FALLBACK=1", cmake)

    def test_a5_data_plane_has_no_host_item_read(self):
        relative_path = (
            "vllm_ascend/dsa_sparse/dsa_ascend_ops_backend.py"
        )
        for function_name in (
            "_lightning_indexer_decode_update_a5",
            "_kvcache_scatter_copy_a5",
            "lightning_indexer_decode_update",
            "kvcache_scatter_copy",
        ):
            function = _function_node(relative_path, function_name)
            item_calls = [
                node for node in ast.walk(function)
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "item"
                )
            ]
            self.assertEqual(
                item_calls,
                [],
                f"{function_name} must not read device scalars on Host",
            )

    def test_a5_empty_slot_does_not_index_resident_metadata(self):
        source = _read(
            "vllm_ascend/dsa_sparse/dsa_ascend_ops_backend.py"
        )
        self.assertIn("valid_evictions = evicted_tokens >= 0", source)
        self.assertIn("safe_evicted_tokens = torch.where(", source)
        self.assertNotIn("resident_row[evicted_tokens] = -1", source)


class TestOperatorCMakeRegistration(unittest.TestCase):
    def test_custom_package_uses_v023_module_targets(self):
        operators = {
            "kv_cache_full_block_dump": ("op_host_aclnn", "aclnn"),
            "kvcache_scatter_copy": ("op_host_aclnn", "aclnn"),
            "lightning_indexer_decode_update": (
                "op_host_aclnnInner",
                "aclnn_inner",
            ),
            "sparse_flash_attention_for_offload": (
                "op_host_aclnnInner",
                "aclnn_inner",
            ),
        }
        obsolete_targets = ("opsproto", "opapi", "optiling")

        for operator, (opdef_target, aclnn_type) in operators.items():
            source = _read(
                f"csrc/attention/{operator}/op_host/CMakeLists.txt"
            )
            with self.subTest(operator=operator):
                self.assertIn("if (BUILD_OPEN_PROJECT)", source)
                self.assertRegex(
                    source,
                    rf"target_sources\(\s*{opdef_target}\s+PRIVATE",
                )
                self.assertIn("if (NOT BUILD_OPS_RTY_KERNEL)", source)
                self.assertRegex(
                    source,
                    rf"add_modules_sources\(\s*"
                    rf"OPTYPE\s+{operator}\s+"
                    rf"ACLNNTYPE\s+{aclnn_type}\s*\)",
                )
                for target in obsolete_targets:
                    self.assertNotRegex(
                        source,
                        rf"target_(?:sources|include_directories)"
                        rf"\(\s*{target}\b",
                    )


class TestBuildResourceControl(unittest.TestCase):
    def test_automatic_jobs_are_cpu_and_memory_bounded(self):
        module = _load_build_jobs_module()
        gib = 1024**3

        self.assertEqual(module.default_build_jobs(128, 64 * gib), 8)
        self.assertEqual(module.default_build_jobs(128, 16 * gib), 3)
        self.assertEqual(module.default_build_jobs(128, 4 * gib), 1)
        self.assertEqual(module.default_build_jobs(2, 64 * gib), 2)

    def test_explicit_max_jobs_is_preserved_and_validated(self):
        module = _load_build_jobs_module()

        plan = module.resolve_build_jobs("2")
        self.assertEqual(plan.num_jobs, 2)
        self.assertEqual(plan.source, "MAX_JOBS")
        for invalid in ("", "0", "-1", "many"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    module.resolve_build_jobs(invalid)

    def test_job_limit_is_exported_before_aclnn_build(self):
        source = _read("setup.py")
        start = source.index("class cmake_build_ext(")
        end = source.index("class custom_install(", start)
        command = source[start:end]

        self.assertLess(
            command.index('os.environ["MAX_JOBS"]'),
            command.index('self.run_command("build_aclnn")'),
        )
        self.assertEqual(
            command.count("subprocess.check_call(cmake_args"),
            1,
            "the main extension must be configured exactly once",
        )


class TestKVCacheGroupingContract(unittest.TestCase):
    def test_mixed_indexer_and_mla_specs_signal_non_uniform(self):
        merge, attention_cls, indexer_cls = _load_indexer_merge_contract()
        indexer_spec = indexer_cls(
            block_size=128,
            num_kv_heads=1,
            head_size=128,
            dtype="bf16",
        )
        mla_spec = attention_cls(
            block_size=128,
            num_kv_heads=1,
            head_size=512,
            dtype="bf16",
        )

        def is_uniform(specs):
            try:
                merge(indexer_cls, specs)
            except AssertionError:
                return False
            return True

        self.assertFalse(is_uniform([indexer_spec, mla_spec]))

    def test_split_indexer_uses_parent_mla_prefill_backend(self):
        resolver, indexer_cls = (
            _load_sfa_prefill_layer_resolver_contract()
        )
        prefill_backend = object()
        attention_prefix = "model.layers.0.self_attn"
        mla_layer_name = f"{attention_prefix}.attn"
        indexer_layer_name = f"{attention_prefix}.indexer.k_cache"
        static_forward_context = {
            mla_layer_name: types.SimpleNamespace(
                prefill_backend=prefill_backend,
            ),
            indexer_layer_name: types.SimpleNamespace(),
        }
        vllm_config = types.SimpleNamespace(
            compilation_config=types.SimpleNamespace(
                static_forward_context=static_forward_context,
            ),
        )

        resolved = resolver(
            indexer_cls(),
            [indexer_layer_name],
            vllm_config,
        )

        self.assertEqual(resolved, [mla_layer_name])
        self.assertIs(
            static_forward_context[resolved[0]].prefill_backend,
            prefill_backend,
        )

        source = ast.parse(_read("vllm_ascend/attention/sfa_v1.py"))
        builder = next(
            node
            for node in source.body
            if (
                isinstance(node, ast.ClassDef)
                and node.name == "AscendSFAMetadataBuilder"
            )
        )
        init = next(
            node
            for node in builder.body
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == "__init__"
            )
        )
        self.assertTrue(
            any(
                isinstance(call.func, ast.Name)
                and call.func.id == "_resolve_sfa_prefill_layer_names"
                for call in (
                    node
                    for node in ast.walk(init)
                    if isinstance(node, ast.Call)
                )
            )
        )
        super_init_calls = [
            call
            for call in (
                node
                for node in ast.walk(init)
                if isinstance(node, ast.Call)
            )
            if (
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "__init__"
                and isinstance(call.func.value, ast.Call)
                and isinstance(call.func.value.func, ast.Name)
                and call.func.value.func.id == "super"
            )
        ]
        self.assertEqual(len(super_init_calls), 1)
        self.assertEqual(
            ast.unparse(super_init_calls[0].args[1]),
            "prefill_layer_names",
        )


class TestV023LifecycleContract(unittest.TestCase):
    def test_engine_child_composes_dsa_outside_late_platform_patch(self):
        for balance_enabled in (False, True):
            with self.subTest(balance_enabled=balance_enabled):
                calls = []

                def upstream_run_engine_core(*args, **kwargs):
                    calls.append("upstream")
                    return "upstream"

                class FakeEngineCoreProc:
                    run_engine_core = upstream_run_engine_core

                entrypoint = _load_engine_process_entrypoint_contract(
                    FakeEngineCoreProc
                )
                entrypoint["ensure_dsa_engine_core_entrypoint"]()

                runtime_module = types.ModuleType(
                    "vllm_ascend.patch.dsa_sparse.patch_runtime"
                )
                platform_patch_installed = False

                def install_dsa_runtime_patches():
                    nonlocal platform_patch_installed
                    if platform_patch_installed:
                        return
                    platform_patch_installed = True

                    captured_entrypoint = FakeEngineCoreProc.run_engine_core
                    self.assertIs(
                        captured_entrypoint,
                        upstream_run_engine_core,
                        "late platform imports must not capture the DSA wrapper",
                    )

                    def balance_run_engine_core(*args, **kwargs):
                        calls.append("balance")
                        config = kwargs["vllm_config"]
                        if config.balance_enabled:
                            calls.append("balance_custom")
                            return "balance"
                        return captured_entrypoint(*args, **kwargs)

                    FakeEngineCoreProc.run_engine_core = (
                        balance_run_engine_core
                    )

                runtime_module.install_dsa_runtime_patches = (
                    install_dsa_runtime_patches
                )
                fake_packages = {}
                for module_name in (
                    "vllm_ascend",
                    "vllm_ascend.patch",
                    "vllm_ascend.patch.dsa_sparse",
                ):
                    package = types.ModuleType(module_name)
                    package.__path__ = []
                    fake_packages[module_name] = package
                fake_packages[runtime_module.__name__] = runtime_module

                def verify_entrypoint():
                    self.assertTrue(
                        entrypoint["is_dsa_run_engine_core_wrapper"](
                            FakeEngineCoreProc.run_engine_core
                        )
                    )

                entrypoint["verify_dsa_runtime_patches_installed"] = (
                    verify_entrypoint
                )
                config = types.SimpleNamespace(
                    balance_enabled=balance_enabled
                )
                with mock.patch.dict(sys.modules, fake_packages):
                    result = FakeEngineCoreProc.run_engine_core(
                        vllm_config=config
                    )

                self.assertTrue(
                    entrypoint["is_dsa_run_engine_core_wrapper"](
                        FakeEngineCoreProc.run_engine_core
                    )
                )
                if balance_enabled:
                    self.assertEqual(result, "balance")
                    self.assertEqual(calls, ["balance", "balance_custom"])
                else:
                    self.assertEqual(result, "upstream")
                    self.assertEqual(calls, ["balance", "upstream"])

    def test_allocate_slots_signature_tracks_v023(self):
        function = _function_node(
            "vllm_ascend/patch/dsa_sparse/"
            "patch_kv_cache_decoupling.py",
            "_allocate_slots",
        )
        argument_names = [arg.arg for arg in function.args.args]
        self.assertEqual(
            argument_names,
            [
                "self",
                "request",
                "num_new_tokens",
                "num_new_computed_tokens",
                "new_computed_blocks",
                "num_lookahead_tokens",
                "num_external_computed_tokens",
                "delay_cache_blocks",
                "num_encoder_tokens",
                "full_sequence_must_fit",
                "reserved_blocks",
            ],
        )

    def test_indexer_spec_uses_logical_model_abi(self):
        source = _read(
            "vllm_ascend/patch/dsa_sparse/patch_deepseek_v2.py"
        )
        self.assertIn('"index_head_dim"', source)
        self.assertIn("dtype=vllm_config.model_config.dtype", source)
        self.assertNotIn("dtype=self.dtype", source)

    def test_final_prefill_dumps_every_complete_block(self):
        source = _read(
            "vllm_ascend/dsa_sparse/dsa_forward_batch_builder.py"
        )
        self.assertIn("num_full_blocks = len(block_hashes)", source)
        self.assertIn("dump_hashes = list(block_hashes)", source)
        self.assertIn(
            "logical_block_indices = list(range(num_full_blocks))",
            source,
        )

    def test_mtp_boundary_budget(self):
        module = _load_lightweight_types_module()
        safe = module.max_safe_mtp_drafts_before_block_boundary
        self.assertEqual(safe(0, 1, 128), 126)
        self.assertEqual(safe(124, 1, 128), 2)
        self.assertEqual(safe(126, 1, 128), 0)
        self.assertEqual(safe(127, 1, 128), 0)
        with self.assertRaises(RuntimeError):
            safe(127, 2, 128)

    def test_mtp_executes_selection_before_next_round(self):
        source = _read("vllm_ascend/attention/sfa_v1.py")
        start = source.index("def _execute_dsa_offload_rounds(")
        end = source.index(
            "def _record_dcp_query_gather_context(",
            start,
        )
        function = source[start:end]
        self.assertIn("for round_index in range(max_rounds):", function)
        self.assertIn("current_key_lens = current_key_lens - remaining", function)
        self.assertIn("row_indices=active_rows", function)
        self.assertLess(
            function.index("execute_decode_selection_pipeline("),
            function.index("sparse_attention_for_offload("),
        )

    def test_supported_runtime_envelope_fails_closed(self):
        source = _read("vllm_ascend/dsa_sparse/dsa_config.py")
        for required_contract in (
            'architecture != "GlmMoeDsaForCausalLM"',
            "supports DP+TP but not PP/DCP/PCP",
            "block_size != 128",
            "cannot use sparse C8 cache modes",
            "source FP16/BF16 ABI",
            "requires non-chunked prefill",
            "supports MTP speculative decoding only",
            "enforce_eager = True",
        ):
            self.assertIn(required_contract, source)

    def test_only_mla_specs_are_resident_cache_planes(self):
        source = _read(
            "vllm_ascend/dsa_sparse/dsa_spec_utils.py"
        )
        function = source[
            source.index("def is_dsa_mla_resident_spec("):
        ]
        self.assertIn("MLAAttentionSpec", function)
        self.assertNotIn(
            "isinstance(spec, FullAttentionSpec)",
            function,
        )
        self.assertNotIn(
            '_isinstance_live(spec, "FullAttentionSpec")',
            function,
        )

    def test_non_dsa_engine_child_does_not_install_runtime_patches(self):
        source = _read(
            "vllm_ascend/patch/dsa_sparse/patch_engine_process.py"
        )
        start = source.index("def _dsa_sparse_run_engine_core(")
        end = source.index(
            "setattr(_dsa_sparse_run_engine_core",
            start,
        )
        wrapper = source[start:end]
        enabled_guard = wrapper.index(
            "if _is_dsa_enabled_on_config("
        )
        install = wrapper.index("_install_dsa_runtime_patches()")
        self.assertLess(enabled_guard, install)

    def test_decode_barrier_preserves_preempted_waiting_requests(self):
        source = _read(
            "vllm_ascend/patch/dsa_sparse/patch_scheduler.py"
        )
        start = source.index("def _withhold_waiting_for_decode(")
        end = source.index(
            "def _populate_dsa_scheduler_output(",
            start,
        )
        function = source[start:end]
        self.assertIn("temporary_waiting = list(self.waiting)", function)
        self.assertIn(
            "temporary_skipped_waiting = list(self.skipped_waiting)",
            function,
        )
        self.assertIn(
            "old_waiting.prepend_request(request)",
            function,
        )
        self.assertIn(
            "old_skipped_waiting.prepend_request(request)",
            function,
        )

    def test_worker_cache_initialization_fails_fast(self):
        source = _read(
            "vllm_ascend/dsa_sparse/dsa_ascend_hot_kv_store.py"
        )
        string_constants = {
            node.value
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
        }
        self.assertIn(
            "DSA split-cache initialization did not receive a dense "
            "Indexer KV tensor",
            string_constants,
        )
        self.assertIn("missing_mla_layers", source)
        self.assertIn("self.freeze_capacity()", source)


if __name__ == "__main__":
    unittest.main()
