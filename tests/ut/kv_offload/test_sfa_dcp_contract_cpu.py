# SPDX-License-Identifier: Apache-2.0
"""SFA DCP layout contracts executable with the Python standard library."""

import ast
import itertools
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3] / "vllm_ascend"


def load_definitions(path, names, scope, class_name=None):
    tree = ast.parse((ROOT / path).read_text())
    body = tree.body
    if class_name:
        body = next(node.body for node in body if isinstance(node, ast.ClassDef) and node.name == class_name)
    nodes = [node for node in body if isinstance(node, ast.FunctionDef) and node.name in names]
    for node in nodes:
        node.decorator_list = []
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(ROOT / path), "exec"), scope)
    return scope


class FullAttentionSpec(SimpleNamespace):
    pass


class MLAAttentionSpec(FullAttentionSpec):
    pass


class IndexerSpec(MLAAttentionSpec):
    pass


class TestSFADCPContracts(unittest.TestCase):
    def test_model_independent_enablement_excludes_kpool_and_compressed_indexers(self):
        scope = load_definitions(
            "utils.py",
            {"model_uses_kpool_indexer", "model_uses_sfa_sparse", "enable_sfa_dcp_sharded_indexer"},
            {},
        )
        for model, dcp, kind in itertools.product(
            ("kimi_k31", "glm_moe_dsa", "deepseek_v32"), (1, 2, 4), ("lightning", "kpool", "compressed", "dense")
        ):
            with self.subTest(model=model, dcp=dcp, kind=kind):
                text = SimpleNamespace(model_type=model)
                if kind != "dense":
                    text.index_topk = 2048
                if kind == "kpool":
                    text.index_kpool = 8
                if kind == "compressed":
                    text.compress_ratios = [4]
                config = SimpleNamespace(
                    model_config=SimpleNamespace(hf_text_config=text, hf_config=text),
                    parallel_config=SimpleNamespace(decode_context_parallel_size=dcp),
                )
                self.assertEqual(scope["enable_sfa_dcp_sharded_indexer"](config), dcp > 1 and kind == "lightning")

    def test_indexer_page_bytes_include_one_local_key_and_scale_region(self):
        scope = load_definitions(
            "core/kv_cache_interface.py",
            {"state_content_size_bytes", "real_page_size_bytes"},
            {"get_dtype_size": {"int8": 1, "fp8": 1, "fp16": 2, "fp32": 4}.__getitem__},
            "AscendSFAIndexerCacheSpec",
        )
        for dtype, scale_dtype, block_size in itertools.product(("int8", "fp8"), ("fp16", "fp32"), (128, 768)):
            spec = SimpleNamespace(head_size=128, dtype=dtype, scale_dim=1, scale_dtype=scale_dtype)
            spec.state_content_size_bytes = scope["state_content_size_bytes"](spec)
            spec.block_size, spec.num_kv_heads = block_size, 1
            expected = block_size * (128 + {"fp16": 2, "fp32": 4}[scale_dtype])
            self.assertEqual(scope["real_page_size_bytes"](spec), expected)

    def test_push_indexer_and_mla_use_each_peer_sharding_flag(self):
        scope = load_definitions(
            "distributed/kv_transfer/kv_p2p/mooncake/push_worker.py",
            {"_get_layer_dcp_sizes"},
            {
                "AscendSFAIndexerCacheSpec": IndexerSpec,
                "is_circular_kv_cache_spec": lambda spec: spec.is_circular,
            },
            "MooncakePushConnectorWorker",
        )
        for local_dcp, remote_dcp, local_sharded, remote_sharded, circular, cls in itertools.product(
            (1, 2, 4), (1, 2, 4), (False, True), (False, True), (False, True), (MLAAttentionSpec, IndexerSpec)
        ):
            with self.subTest(
                local_dcp=local_dcp,
                remote_dcp=remote_dcp,
                local_sharded=local_sharded,
                remote_sharded=remote_sharded,
                circular=circular,
                cache_type=cls.__name__,
            ):
                worker = SimpleNamespace(
                    dcp_size=local_dcp,
                    kv_cache_specs=[cls(dcp_sharded=local_sharded, is_circular=circular)],
                    layer_name_to_spec_index={"cache": 0},
                )
                remote = SimpleNamespace(
                    dcp_size=remote_dcp,
                    metadata_by_pp_rank={0: SimpleNamespace(layer_names=["cache"], dcp_sharded=[remote_sharded])},
                )
                self.assertEqual(
                    scope["_get_layer_dcp_sizes"](worker, "cache", remote, (0, 0, 0)),
                    (
                        local_dcp if local_sharded and not circular else 1,
                        remote_dcp if remote_sharded and not circular else 1,
                    ),
                )

    def test_pd_indexer_and_mla_follow_identical_token_ownership(self):
        scope = load_definitions(
            "distributed/kv_transfer/kv_p2p/mooncake/pull_worker.py",
            {"_get_attention_dcp_sizes", "_compute_group_block_ids", "_expand_block_ids", "_select_remote_tp_rank"},
            {
                "FullAttentionSpec": FullAttentionSpec,
                "SlidingWindowSpec": type("SlidingWindowSpec", (), {}),
                "is_circular_kv_cache_spec": lambda spec: False,
            },
            "MooncakePullRecvingThread",
        )
        worker_cls = type(
            "Worker",
            (),
            {
                name: staticmethod(value) if name in {"_expand_block_ids", "_select_remote_tp_rank"} else value
                for name, value in scope.items()
                if name.startswith("_") and callable(value)
            },
        )
        # Different P/D page sizes and interleaves, partial tails and cached prefixes.
        for local_dcp, remote_dcp, local_parts, remote_parts, prefix in itertools.product(
            (1, 2, 4), (1, 2, 4), (1, 2), (1, 2), (0, 17)
        ):
            if local_dcp == remote_dcp == 1:
                continue
            for rank in range(local_dcp):
                with self.subTest(
                    local_dcp=local_dcp,
                    remote_dcp=remote_dcp,
                    rank=rank,
                    local_parts=local_parts,
                    remote_parts=remote_parts,
                    prefix=prefix,
                ):
                    worker = worker_cls()
                    worker.dcp_size, worker.dcp_rank = local_dcp, rank
                    local_page, remote_page, kernel = 16 * local_parts, 16 * remote_parts, 16
                    worker.vllm_config = SimpleNamespace(
                        parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=local_page)
                    )
                    remote = SimpleNamespace(dcp_sharded=[True])
                    local_ids, remote_ids = list(range(100, 120)), list(range(200, 220))
                    results = []
                    for cls in (MLAAttentionSpec, IndexerSpec):
                        spec = cls(dcp_sharded=True)
                        sizes = worker._get_attention_dcp_sizes(spec, 0, remote, remote_dcp)
                        self.assertEqual(sizes, (local_dcp, remote_dcp))
                        results.append(
                            worker._compute_group_block_ids(
                                "request",
                                [list(range(remote_dcp))],
                                remote_dcp,
                                0,
                                local_page,
                                remote_page,
                                local_ids,
                                local_ids,
                                remote_ids,
                                102,
                                101,
                                prefix,
                                local_parts,
                                remote_parts,
                                spec,
                                0,
                                remote_page,
                            )
                        )
                    self.assertEqual(results[0], results[1])
                    actual = {(owner, dst, src) for owner, dsts, srcs in results[1] for dst, src in zip(dsts, srcs)}
                    expected = set()
                    for token in range(prefix // kernel * kernel, 101, kernel):
                        if token // local_page % local_dcp != rank:
                            continue
                        local = token // (local_page * local_dcp) * local_page + token % local_page
                        remote_token = token // (remote_page * remote_dcp) * remote_page + token % remote_page
                        expected.add(
                            (
                                token // remote_page % remote_dcp,
                                local_ids[local // local_page] * local_parts + local % local_page // kernel,
                                remote_ids[remote_token // remote_page] * remote_parts
                                + remote_token % remote_page // kernel,
                            )
                        )
                    self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
