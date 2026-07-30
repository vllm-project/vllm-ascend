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


def _load_full_sequence_fit_contract():
    path = (
        REPO_ROOT
        / "vllm_ascend/patch/dsa_sparse/patch_kv_cache_decoupling.py"
    )
    module = ast.parse(path.read_text(encoding="utf-8"))
    function_names = {
        "_can_allocate_by_group",
        "_can_fit_full_sequence",
    }
    selected_nodes = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name in function_names
    ]
    if {node.name for node in selected_nodes} != function_names:
        raise AssertionError(
            "DSA full-sequence admission helpers are incomplete"
        )

    class FakeMultiBlockPool:
        def __init__(self, can_allocate):
            self._can_allocate = can_allocate

        def can_allocate(self, needed, *, reserved_blocks=0):
            return self._can_allocate(needed, reserved_blocks)

    contract_module = ast.Module(body=selected_nodes, type_ignores=[])
    ast.fix_missing_locations(contract_module)
    namespace = {
        "Any": object,
        "KVCacheBlock": object,
        "MultiBlockPool": FakeMultiBlockPool,
        "Sequence": list,
    }
    exec(compile(contract_module, str(path), "exec"), namespace)
    return namespace["_can_fit_full_sequence"], FakeMultiBlockPool


def _load_dsa_config_switch_contract():
    path = REPO_ROOT / "vllm_ascend/dsa_sparse/dsa_config.py"
    module = ast.parse(path.read_text(encoding="utf-8"))
    function_names = {
        "_normalize_positive_int_sequence",
        "_normalize_dsa_trace_points_config",
        "_normalize_dsa_sparse_config",
        "is_dsa_sparse_config_enabled",
    }
    selected_nodes = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name in function_names
    ]
    contract_module = ast.Module(body=selected_nodes, type_ignores=[])
    ast.fix_missing_locations(contract_module)
    mappings = (
        ("enabled", "enable_dsa_sparse_cache"),
        ("split_indexer_cache", "enable_dsa_split_indexer_cache"),
        ("indexer_mla_block_ratio", "dsa_indexer_mla_block_ratio"),
        ("max_active_reqs", "dsa_max_active_reqs"),
        ("hot_cpu_block_multiple", "dsa_hot_cpu_block_multiple"),
    )
    graph_key = "enable_row_mode_decode_graph"
    public_keys = frozenset(
        {public for public, _ in mappings}
        | {
            "sparse_activation_tokens",
            "prompt_budget_thresholds",
            "resident_budget_tokens",
            graph_key,
            "trace_points",
        }
    )
    namespace = {
        "Any": object,
        "Sequence": (list, tuple),
        "DSA_SPARSE_ADDITIONAL_CONFIG_KEY": "dsa_sparse_config",
        "_DSA_GRAPH_PUBLIC_CONFIG_KEY": graph_key,
        "_DSA_SPARSE_CONFIG_FIELD_MAPPINGS": mappings,
        "_DSA_SPARSE_ACTIVATION_CONFIG_KEY": "sparse_activation_tokens",
        "_DSA_PROMPT_BUDGET_THRESHOLDS_CONFIG_KEY": "prompt_budget_thresholds",
        "_DSA_RESIDENT_BUDGET_TOKENS_CONFIG_KEY":
        "resident_budget_tokens",
        "_DSA_SPARSE_DEFAULT_CACHE_ATTRS": {
            "enable_dsa_sparse_cache": False,
            "enable_dsa_split_indexer_cache": False,
            "dsa_indexer_mla_block_ratio": 3,
            "dsa_sparse_activation_tokens": 6144,
            "dsa_prompt_budget_thresholds": (32768, 65536),
            "dsa_resident_budget_tokens": (6144, 10240, 12288),
            "dsa_hbm_sparse_budget": 12288,
            "dsa_max_active_reqs": 256,
            "dsa_hot_cpu_block_multiple": 3,
        },
        "_DSA_SPARSE_PUBLIC_KEYS": public_keys,
        "DSA_TRACE_PUBLIC_KEYS": frozenset({"enabled", "points", "ranks"}),
        "DSA_TRACE_DEFAULT_POINTS": ("first_sample",),
        "DSA_TRACE_DEFAULT_RANKS": (0,),
        "DSA_ROW_MODE_DECODE_GRAPH_CONFIG_KEY":
        "enable_dsa_row_mode_decode_graph",
        "DSA_TRACE_CONFIG_KEY": "dsa_sparse_trace_points",
        "DSA_SFA_COMPUTE_TOPK": 2048,
        "DSA_LIDU_OUTPUT_CAPACITY": 16384,
        "DSA_LIDU_SUPPORTED_RESIDENT_BUDGETS":
        (6144, 8192, 10240, 12288),
    }
    exec(compile(contract_module, str(path), "exec"), namespace)
    return (
        namespace["_normalize_dsa_sparse_config"],
        namespace["is_dsa_sparse_config_enabled"],
    )


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

    def test_a2_a3_install_verifies_dsa_binary_registry(self):
        build_script = _read("csrc/build_aclnn.sh")
        required_op_types = (
            "LightningIndexerDecodeUpdate",
            "KvcacheScatterCopy",
            "SparseFlashAttentionForOffload",
            "KvCacheFullBlockDump",
        )

        self.assertIn(
            'ascend910b|ascend910_93)',
            build_script,
        )
        self.assertIn("binary_info_config.json", build_script)
        for op_type in required_op_types:
            self.assertIn(f'"{op_type}"', build_script)

        verification_call = (
            'verify_installed_dsa_ops '
            '"${custom_ops_install_dir}" "${SOC_ARG}"'
        )
        self.assertIn(verification_call, build_script)
        self.assertGreater(
            build_script.index(verification_call),
            build_script.index('"${installer_candidates[0]}" '
                               '--install-path="${custom_ops_install_dir}"'),
        )

    def test_a2_a3_build_discards_stale_aclnn_tree(self):
        build_script = _read("csrc/build_aclnn.sh")
        build_section = build_script[build_script.index("log_selected_ops"):]
        clean_case = build_section[
            build_section.index('case "${SOC_ARG}" in'):
            build_section.index(
                'bash build.sh --pkg --ops="${CUSTOM_OPS}" '
                '--soc="${SOC_ARG}"'
            )
        ]

        self.assertIn("ascend910b|ascend910_93)", clean_case)
        self.assertIn("rm -rf -- build output build_out", clean_case)
        self.assertIn("rm -rf -- output build_out", clean_case)
        self.assertLess(
            build_section.index("rm -rf -- build output build_out"),
            build_section.index(
                'bash build.sh --pkg --ops="${CUSTOM_OPS}" '
                '--soc="${SOC_ARG}"'
            ),
        )

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
    def test_worker_installs_indexer_spec_patch_before_model_init(self):
        source = _read("vllm_ascend/worker/worker.py")
        worker_start = source.index("class NPUWorker(")
        init_start = source.index("    def __init__(", worker_start)
        init_end = source.index("\n    def ", init_start + 1)
        worker_init = source[init_start:init_end]

        adapt_patch = worker_init.index("adapt_patch()")
        enabled_guard = worker_init.index(
            "if is_dsa_sparse_runtime_enabled(vllm_config):"
        )
        indexer_patch = worker_init.index(
            "patch_deepseek_v2_indexer_cache_spec()"
        )
        worker_base_init = worker_init.index("super().__init__(")

        self.assertLess(adapt_patch, enabled_guard)
        self.assertLess(enabled_guard, indexer_patch)
        self.assertLess(indexer_patch, worker_base_init)

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

    def test_full_sequence_fit_uses_v023_group_admission_semantics(self):
        can_fit_full_sequence, multi_block_pool_cls = (
            _load_full_sequence_fit_contract()
        )
        coordinator_calls = []
        allocation_calls = []

        class FakeCoordinator:
            def get_num_blocks_to_allocate_by_group(self, **kwargs):
                coordinator_calls.append(kwargs)
                return [3, 5]

        pool = multi_block_pool_cls(
            lambda needed, reserved: (
                allocation_calls.append((needed, reserved))
                or needed == [3, 5]
                and reserved == 2
            )
        )
        manager = types.SimpleNamespace(
            block_pool=pool,
            coordinator=FakeCoordinator(),
            empty_kv_cache_blocks=types.SimpleNamespace(
                blocks=((), ()),
            ),
            max_model_len=512,
        )
        request = types.SimpleNamespace(
            request_id="request-0",
            num_computed_tokens=128,
            num_tokens=768,
        )

        self.assertTrue(
            can_fit_full_sequence(
                manager,
                request,
                num_new_computed_tokens=64,
                num_external_computed_tokens=32,
                reserved_blocks=2,
            )
        )
        self.assertEqual(allocation_calls, [([3, 5], 2)])
        self.assertEqual(
            coordinator_calls,
            [
                {
                    "request_id": "request-0",
                    "num_tokens": 512,
                    "new_computed_blocks": ((), ()),
                    "num_encoder_tokens": 0,
                    "total_computed_tokens": 224,
                    "num_tokens_main_model": 512,
                    "apply_admission_cap": True,
                }
            ],
        )

    def test_full_sequence_fit_is_installed_on_kv_cache_manager(self):
        installer = _function_node(
            "vllm_ascend/patch/dsa_sparse/"
            "patch_kv_cache_decoupling.py",
            "install_dsa_kv_cache_decoupling_patch",
        )
        bindings = {
            (
                ast.unparse(node.targets[0]),
                ast.unparse(node.value),
            )
            for node in installer.body
            if isinstance(node, ast.Assign) and len(node.targets) == 1
        }
        self.assertIn(
            (
                "manager_mod.KVCacheManager.can_fit_full_sequence",
                "_can_fit_full_sequence",
            ),
            bindings,
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

    def test_draft_boundary_helper_is_conservative(self):
        module = _load_lightweight_types_module()
        safe = module.max_safe_mtp_drafts_before_block_boundary
        self.assertEqual(safe(0, 1, 128), 126)
        self.assertEqual(safe(124, 1, 128), 2)
        self.assertEqual(safe(126, 1, 128), 0)
        self.assertEqual(safe(127, 1, 128), 0)
        with self.assertRaises(RuntimeError):
            safe(127, 2, 128)

    def test_multi_token_rounds_select_before_attention(self):
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
            "data_parallel_size=1 only",
            "supports TP but not PP/DCP/PCP",
            "cannot be combined with SFA DSA-CP",
            "block_size != 128",
            "cannot use sparse C8 cache modes",
            "source FP16/BF16 ABI",
            "requires non-chunked prefill",
            "does not support speculative/MTP decoding",
            "enforce_eager = True",
        ):
            self.assertIn(required_contract, source)

    def test_dsa_dp_guard_and_example_default_are_accuracy_safe(self):
        source = _read("vllm_ascend/dsa_sparse/dsa_config.py")
        self.assertIn("if data_parallel_size != 1:", source)
        self.assertIn("DP>1 can silently corrupt ", source)
        self.assertIn('"token accuracy."', source)

        example = _read("examples/glm51_dsa_sparse_mtp.sh")
        self.assertIn('DP_SIZE="${DP_SIZE:-1}"', example)
        self.assertNotIn('DP_SIZE="${DP_SIZE:-2}"', example)
        self.assertIn('\\"enable_dsa_cp\\":false', example)

    def test_example_enables_first_sample_trace_with_explicit_tp_rank(self):
        example = _read("examples/glm51_dsa_sparse_mtp.sh")
        self.assertIn(
            'DSA_TRACE_ENABLED="${DSA_TRACE_ENABLED:-${DSA_ENABLED}}"',
            example,
        )
        self.assertIn('DSA_TRACE_RANK="${DSA_TRACE_RANK:-0}"', example)
        self.assertIn(
            '\\"trace_points\\":{\\"enabled\\":${DSA_TRACE_ENABLED},'
            '\\"points\\":[\\"first_sample\\"],'
            '\\"ranks\\":[${DSA_TRACE_RANK}]}',
            example,
        )

    def test_sparse_offload_rejects_independent_dsa_cp_switch(self):
        source = _read("vllm_ascend/dsa_sparse/dsa_config.py")
        self.assertIn(
            'dsa_cp_enabled = additional_config.get("enable_dsa_cp", False)',
            source,
        )
        self.assertIn(
            "additional_config['enable_dsa_cp'] must be a bool",
            source,
        )
        self.assertIn(
            "different token sharding, slot mappings, and Indexer/SFA tensor",
            source,
        )

    def test_public_enabled_false_is_a_true_off_switch(self):
        source = _read("vllm_ascend/dsa_sparse/dsa_config.py")
        enabled = ast.unparse(
            _function_node(
                "vllm_ascend/dsa_sparse/dsa_config.py",
                "is_dsa_sparse_config_enabled",
            )
        )
        self.assertLess(
            enabled.index("'enabled' in dsa_config"),
            enabled.index("cache_config = getattr"),
        )
        self.assertIn(
            "split_indexer_cache']=True is only valid when",
            source,
        )
        self.assertIn("must be a bool", source)

        example = _read("examples/glm51_dsa_sparse_mtp.sh")
        self.assertIn('DSA_ENABLED="${DSA_ENABLED:-true}"', example)
        self.assertIn('\\"enabled\\":${DSA_ENABLED}', example)
        self.assertNotIn('\\"split_indexer_cache\\":true', example)

        normalize, is_enabled = _load_dsa_config_switch_contract()
        enabled_attrs, enabled_updates = normalize({"enabled": True})
        self.assertTrue(enabled_attrs["enable_dsa_sparse_cache"])
        self.assertTrue(enabled_attrs["enable_dsa_split_indexer_cache"])
        self.assertEqual(
            enabled_updates["dsa_sparse_trace_points"],
            {
                "enabled": True,
                "points": ["first_sample"],
                "ranks": [0],
            },
        )
        disabled_attrs, disabled_updates = normalize({"enabled": False})
        self.assertFalse(disabled_attrs["enable_dsa_sparse_cache"])
        self.assertFalse(disabled_attrs["enable_dsa_split_indexer_cache"])
        self.assertEqual(
            disabled_updates["dsa_sparse_trace_points"],
            {
                "enabled": False,
                "points": ["first_sample"],
                "ranks": [0],
            },
        )
        _, explicit_trace_updates = normalize(
            {
                "enabled": True,
                "trace_points": False,
            }
        )
        self.assertEqual(
            explicit_trace_updates["dsa_sparse_trace_points"],
            {"enabled": False},
        )
        with self.assertRaises(TypeError):
            normalize({"enabled": "false"})
        with self.assertRaises(ValueError):
            normalize({
                "enabled": False,
                "split_indexer_cache": True,
            })

        stale_cache_config = types.SimpleNamespace(
            enable_dsa_sparse_cache=True)
        public_off_config = types.SimpleNamespace(
            additional_config={
                "dsa_sparse_config": {
                    "enabled": False,
                },
            },
            cache_config=stale_cache_config,
        )
        self.assertFalse(is_enabled(public_off_config))

    def test_dense_decode_keeps_native_attention_until_sparse_activation(self):
        batch_source = _read(
            "vllm_ascend/dsa_sparse/dsa_forward_batch.py"
        )
        runtime_source = _read(
            "vllm_ascend/dsa_sparse/dsa_row_mode_runtime.py"
        )
        sfa_forward = ast.unparse(
            _function_node(
                "vllm_ascend/attention/sfa_v1.py",
                "forward",
            )
        )
        self.assertIn("uses_sparse_offload: bool", batch_source)
        self.assertIn(
            "uses_sparse_offload = bool(np.any(sparse_mask))",
            runtime_source,
        )
        self.assertIn(
            "attn_metadata.dsa_row_mode_batch.uses_sparse_offload",
            sfa_forward,
        )

    def test_speculative_decode_is_rejected_and_example_disables_it(self):
        source = _read("vllm_ascend/dsa_sparse/dsa_config.py")
        start = source.index(
            'speculative_config = getattr(vllm_config, "speculative_config"'
        )
        end = source.index("raw_dsa_config =", start)
        speculative_guard = source[start:end]

        self.assertIn(
            "if speculative_config is not None:",
            speculative_guard,
        )
        self.assertIn(
            "does not support speculative/MTP decoding",
            speculative_guard,
        )
        self.assertIn("--speculative-config", speculative_guard)
        self.assertNotIn("_is_mtp_config", source)
        self.assertNotIn("num_speculative_tokens", speculative_guard)

        example = _read("examples/glm51_dsa_sparse_mtp.sh")
        self.assertNotIn("--speculative-config", example)
        self.assertNotIn("NUM_SPECULATIVE_TOKENS", example)
        self.assertIn(
            'QUANTIZATION="${QUANTIZATION-ascend}"',
            example,
        )
        self.assertIn('--quantization "${QUANTIZATION}"', example)

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


class TestDSATraceContract(unittest.TestCase):
    def _model_runner_method(self, method_name: str) -> ast.FunctionDef:
        module = ast.parse(
            _read("vllm_ascend/worker/model_runner_v1.py")
        )
        runner = next(
            node
            for node in module.body
            if (
                isinstance(node, ast.ClassDef)
                and node.name == "NPUModelRunner"
            )
        )
        return next(
            node
            for node in runner.body
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == method_name
            )
        )

    def test_model_runner_configures_and_emits_first_sample_trace(self):
        init_source = ast.unparse(self._model_runner_method("__init__"))
        sample_source = ast.unparse(
            self._model_runner_method("sample_tokens")
        )
        update_source = ast.unparse(
            self._model_runner_method("_update_states")
        )
        trace_source = ast.unparse(
            self._model_runner_method(
                "_trace_dsa_first_sample_boundary"
            )
        )

        self.assertIn(
            "self._dsa_first_sample_traced_req_ids: set[str] = set()",
            init_source,
        )
        self.assertIn(
            "configure_dsa_trace_from_additional_config("
            "additional_config if "
            "is_dsa_sparse_config_enabled(vllm_config) else None)",
            init_source,
        )
        self.assertIn(
            "self._dsa_trace_tp_rank = "
            "int(get_tp_group().rank_in_group)",
            init_source,
        )
        self.assertIn(
            "self._dsa_first_sample_trace_enabled = dsa_trace_enabled("
            "DSA_TRACE_POINT_FIRST_SAMPLE, "
            "tp_rank=self._dsa_trace_tp_rank)",
            init_source,
        )
        self.assertIn(
            "[DSA trace worker state] global_rank=%s tp_rank=%s",
            init_source,
        )
        self.assertIn(
            "trace_first_sample = "
            "self._dsa_first_sample_trace_enabled and any(",
            sample_source,
        )
        self.assertIn(
            "pre_sample_top1 = logits.detach().argmax(dim=-1)",
            sample_source,
        )
        self.assertIn(
            "self._dsa_first_sample_traced_req_ids.difference_update("
            "scheduler_output.finished_req_ids)",
            update_source,
        )
        self.assertLess(
            sample_source.index(
                "self._dsa_first_sample_trace_enabled"
            ),
            sample_source.index(
                "sampler_output = self._sample("
            ),
        )
        self.assertLess(
            sample_source.index("self._bookkeeping_sync("),
            sample_source.index(
                "self._trace_dsa_first_sample_boundary("
            ),
        )
        self.assertIn("logger.warning(", trace_source)
        self.assertNotIn("logger.info(", trace_source)
        self.assertIn(
            "[DSA first-sample boundary] point=%s tp_rank=%s",
            trace_source,
        )

    def test_first_sample_trace_logs_each_request_once(self):
        method = self._model_runner_method(
            "_trace_dsa_first_sample_boundary"
        )
        method.decorator_list = []
        contract_module = ast.Module(body=[method], type_ignores=[])
        ast.fix_missing_locations(contract_module)

        warning_calls = []

        class FakeLogger:
            def warning(self, *args):
                warning_calls.append(args)

        class FakeTensor:
            def __init__(self, values):
                self.values = values

            def detach(self):
                return self

            def reshape(self, *shape):
                return self

            def to(self, **kwargs):
                return self

            def tolist(self):
                return self.values

        namespace = {
            "BatchDescriptor": object,
            "DSA_TRACE_POINT_FIRST_SAMPLE": "first_sample",
            "logger": FakeLogger(),
            "torch": types.SimpleNamespace(Tensor=object),
        }
        exec(
            compile(
                contract_module,
                "vllm_ascend/worker/model_runner_v1.py",
                "exec",
            ),
            namespace,
        )
        trace = namespace["_trace_dsa_first_sample_boundary"]
        runner = types.SimpleNamespace(
            _dsa_first_sample_traced_req_ids=set(),
            _dsa_trace_tp_rank=0,
            input_batch=types.SimpleNamespace(
                req_id_to_index={"request-0": 0},
                num_prompt_tokens=[16],
                num_computed_tokens_cpu=[16],
            ),
        )
        scheduler_output = types.SimpleNamespace(
            req_dsa_stage={"request-0": 1},
            num_scheduled_tokens={"request-0": 16},
        )
        batch_desc = types.SimpleNamespace(
            num_tokens=16,
            num_reqs=1,
        )

        kwargs = {
            "scheduler_output": scheduler_output,
            "batch_desc": batch_desc,
            "req_ids": ["request-0"],
            "sampled_token_ids": [[42]],
            "pre_sample_top1": FakeTensor([41]),
        }
        trace(runner, **kwargs)
        trace(runner, **kwargs)

        self.assertEqual(len(warning_calls), 1)
        self.assertEqual(
            runner._dsa_first_sample_traced_req_ids,
            {"request-0"},
        )
        self.assertEqual(warning_calls[0][1], "first_sample")
        self.assertEqual(
            warning_calls[0][-1],
            [
                {
                    "row": 0,
                    "input_row": 0,
                    "req_id": "request-0",
                    "is_prefill": False,
                    "prompt_tokens": 16,
                    "computed_tokens": 16,
                    "scheduled_tokens": 16,
                    "dsa_stage": "1",
                    "top1_before_processors": 41,
                    "sampled": [42],
                }
            ],
        )

    def test_trace_config_enables_selected_tp_ranks(self):
        warning_calls = []

        class FakeLogger:
            def warning(self, *args):
                warning_calls.append(args)

        fake_vllm = types.ModuleType("vllm")
        fake_vllm.__path__ = []
        fake_logger_module = types.ModuleType("vllm.logger")
        fake_logger_module.logger = FakeLogger()

        path = REPO_ROOT / "vllm_ascend/dsa_sparse/dsa_trace.py"
        module_name = "_dsa_trace_contract"
        spec = importlib.util.spec_from_file_location(module_name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(
            sys.modules,
            {
                "vllm": fake_vllm,
                "vllm.logger": fake_logger_module,
                module_name: module,
            },
        ):
            spec.loader.exec_module(module)
            default_config = (
                module.configure_dsa_trace_from_additional_config(
                    {
                        "dsa_sparse_config": {
                            "enabled": True,
                        }
                    }
                )
            )
            self.assertTrue(default_config.enabled)
            self.assertEqual(
                default_config.points,
                frozenset({"first_sample"}),
            )
            self.assertEqual(default_config.ranks, frozenset({0}))
            module.configure_dsa_trace(None)

            config = (
                module.configure_dsa_trace_from_additional_config(
                    {
                        "dsa_sparse_config": {
                            "trace_points": {
                                "enabled": True,
                                "points": ["first_sample"],
                                "ranks": [0],
                            }
                        }
                    }
                )
            )

            self.assertTrue(config.enabled)
            self.assertTrue(
                module.dsa_trace_enabled(
                    "first_sample",
                    tp_rank=0,
                )
            )
            self.assertFalse(
                module.dsa_trace_enabled(
                    "first_sample",
                    tp_rank=1,
                )
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "requires an explicit TP rank",
            ):
                module.dsa_trace_enabled("first_sample")
            self.assertEqual(len(warning_calls), 2)
            module.configure_dsa_trace(None)
            self.assertFalse(
                module.dsa_trace_enabled(
                    "first_sample",
                    tp_rank=0,
                )
            )
            disabled_config = (
                module.configure_dsa_trace_from_additional_config(
                    {
                        "dsa_sparse_config": {
                            "enabled": False,
                        }
                    }
                )
            )
            self.assertFalse(disabled_config.enabled)


class TestIndexerRopePrecisionContract(unittest.TestCase):
    def test_indexer_k_q_and_cp_share_full_head_rope_helper(self):
        expected_calls = {
            "vllm_ascend/attention/sfa_v1.py": (
                (
                    "indexer_select_pre_process",
                    "self._apply_indexer_rope(k_li, cos, sin)",
                ),
                (
                    "_prepare_indexer_query_and_weights",
                    "self._apply_indexer_rope(q_li, cos, sin)",
                ),
            ),
            "vllm_ascend/attention/context_parallel/sfa_cp.py": (
                (
                    "indexer_select_post_process",
                    "self._apply_indexer_rope(q_li, cos, sin)",
                ),
            ),
        }
        for relative_path, contracts in expected_calls.items():
            for function_name, expected_call in contracts:
                function = ast.unparse(_function_node(relative_path, function_name))
                with self.subTest(
                    relative_path=relative_path,
                    function_name=function_name,
                ):
                    self.assertIn(expected_call, function)

    def test_non_triton_indexer_rope_honors_operator_contract(self):
        function = ast.unparse(
            _function_node(
                "vllm_ascend/attention/sfa_v1.py",
                "_apply_indexer_rope",
            )
        )
        self.assertIn("if x.shape[-1] != self.head_dim", function)
        self.assertIn(
            "cos.reshape(-1, 1, 1, self.qk_rope_head_dim).contiguous()",
            function,
        )
        self.assertIn(
            "sin.reshape(-1, 1, 1, self.qk_rope_head_dim).contiguous()",
            function,
        )
        self.assertIn(
            "x_pe = torch_npu.npu_interleave_rope(x_pe, cos, sin)",
            function,
        )
        self.assertIn(
            "x_pe = _restore_npu_interleave_rope_layout(x_pe)",
            function,
        )

    def test_interleave_layout_restore_matches_gptj_reference(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        cos_pair = [0.8, 0.6, 0.4, 0.2]
        sin_pair = [0.2, 0.4, 0.6, 0.8]
        cos_half = cos_pair + cos_pair
        sin_half = sin_pair + sin_pair

        even = x[::2]
        odd = x[1::2]
        half_layout = even + odd
        rotated_half = [
            half_layout[index] * cos_half[index] - half_layout[index + 4] * sin_half[index] for index in range(4)
        ] + [
            half_layout[index + 4] * cos_half[index + 4] + half_layout[index] * sin_half[index + 4]
            for index in range(4)
        ]
        restored = [value for pair in zip(rotated_half[:4], rotated_half[4:]) for value in pair]
        reference = [
            value
            for index in range(4)
            for value in (
                even[index] * cos_pair[index] - odd[index] * sin_pair[index],
                odd[index] * cos_pair[index] + even[index] * sin_pair[index],
            )
        ]
        self.assertEqual(restored, reference)

    def test_glm_indexer_rope_matches_sfa_interleave_style(self):
        function = ast.unparse(
            _function_node(
                "vllm_ascend/patch/worker/patch_deepseek_v2.py",
                "_deepseek_v2_mla_attention_init",
            )
        )
        self.assertIn(
            "indexer_is_neox_style = config.model_type != "
            "'glm_moe_dsa' and (not getattr(config, "
            "'indexer_rope_interleave', False))",
            function,
        )
        self.assertIn(
            "is_neox_style=indexer_is_neox_style",
            function,
        )

        sfa_source = _read("vllm_ascend/attention/sfa_v1.py")
        self.assertIn(
            'if self.vllm_config.model_config.hf_config.model_type '
            'in ["glm_moe_dsa"]:',
            sfa_source,
        )
        self.assertIn(
            "self.is_rope_neox_style = False",
            sfa_source,
        )


if __name__ == "__main__":
    unittest.main()
