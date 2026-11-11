# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of the production attention hooks and standard connector methods."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402

# isort: split
from tests.ut.attention.test_mla_layerwise_load_cpu import load
from tests.ut.distributed.ascend_store.test_cross_layer_gva import make_cross_worker, metadata
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import AscendStoreConnector

UTILS = "vllm_ascend/attention/utils.py"
ATTENTION = "vllm_ascend/attention/dsa_v41.py"


def connector(*, shared=False, bound=True, children=()):
    return SimpleNamespace(
        use_cross_layer_kv=shared,
        sub_connectors=children,
        has_connector_metadata=Mock(return_value=bound),
        wait_for_layer_load=Mock(),
        save_kv_layer=Mock(),
    )


def hook_scope(root, *, composite=True):
    attn = SimpleNamespace(supports_cross_layer_kv_transfer_hooks=composite)
    context = SimpleNamespace(attn_metadata=object(), no_compile_layers={"layer": attn})
    scope = {
        "get_forward_context": lambda: context,
        "get_kv_transfer_group": lambda: root,
        "has_kv_transfer_group": Mock(return_value=True),
        "is_v1_kv_transfer_group": Mock(return_value=True),
    }
    for name in (
        "_kv_layer_transfer_connectors",
        "wait_for_kv_layer_from_connector",
        "maybe_save_kv_layer_to_connector",
    ):
        load(UTILS, name, scope)
    return scope, context, attn


class TestStandardConnectorHooks(unittest.TestCase):
    def test_nested_pd_multi_connector_selects_bound_shared_children(self):
        pd, ordinary, shared, unbound = (
            connector(),
            connector(),
            connector(shared=True),
            connector(shared=True, bound=False),
        )
        nested = connector(children=[ordinary, shared, unbound])
        root = connector(children=[pd, nested])
        scope, context, _ = hook_scope(root)
        caches = [torch.ones(1)]
        scope["wait_for_kv_layer_from_connector"]("layer")
        scope["maybe_save_kv_layer_to_connector"]("layer", caches)
        shared.wait_for_layer_load.assert_called_once_with("layer")
        shared.save_kv_layer.assert_called_once_with("layer", caches, context.attn_metadata)
        for sibling in (root, nested, pd, ordinary, unbound):
            sibling.wait_for_layer_load.assert_not_called()
            sibling.save_kv_layer.assert_not_called()

    def test_ordinary_owner_keeps_root_standard_forwarding(self):
        root = connector(children=[connector(), connector(shared=True)])
        scope, context, _ = hook_scope(root, composite=False)
        caches = [torch.ones(1)]
        scope["wait_for_kv_layer_from_connector"]("layer")
        scope["maybe_save_kv_layer_to_connector"]("layer", caches)
        root.wait_for_layer_load.assert_called_once_with("layer")
        root.save_kv_layer.assert_called_once_with("layer", caches, context.attn_metadata)
        for child in root.sub_connectors:
            child.has_connector_metadata.assert_not_called()

    def test_no_transfer_or_metadata_does_not_dispatch(self):
        for case in ("disabled", "other_interface", "profiling", "unbound"):
            with self.subTest(case=case):
                root = connector(shared=True, bound=case != "unbound")
                scope, context, _ = hook_scope(root)
                scope["has_kv_transfer_group"].return_value = case != "disabled"
                scope["is_v1_kv_transfer_group"].return_value = case != "other_interface"
                if case == "profiling":
                    context.attn_metadata = None
                scope["wait_for_kv_layer_from_connector"]("layer")
                scope["maybe_save_kv_layer_to_connector"]("layer", [])
                root.wait_for_layer_load.assert_not_called()
                root.save_kv_layer.assert_not_called()

    def test_composite_owner_without_shared_connector_does_not_advance_other_paths(self):
        for name in ("bulk", "ordinary_gva", "mooncake_block_key", "decode_pd"):
            with self.subTest(name=name):
                root = connector()
                scope, _, _ = hook_scope(root)
                scope["wait_for_kv_layer_from_connector"]("layer")
                scope["maybe_save_kv_layer_to_connector"]("layer", [])
                root.wait_for_layer_load.assert_not_called()
                root.save_kv_layer.assert_not_called()

    def test_attention_waits_then_computes_then_saves(self):
        root = connector(shared=True)
        scope, context, attn = hook_scope(root)
        trace = []
        root.wait_for_layer_load.side_effect = lambda *_: trace.append("wait")
        root.save_kv_layer.side_effect = lambda *_: trace.append("save")
        attn.v41_impl = SimpleNamespace(forward=Mock(side_effect=lambda *_: trace.append("compute")))
        attn.dsa_attn = SimpleNamespace(swa_cache_layer=SimpleNamespace(kv_cache=[torch.ones(1)]))
        hidden, output = torch.ones(3, 2), torch.empty(3, 2)
        load(ATTENTION, "dsa_v41_forward", scope)(hidden, output, "layer")
        self.assertEqual(trace, ["wait", "compute", "save"])
        attn.v41_impl.forward.assert_called_once_with(attn, None, hidden, output)
        root.save_kv_layer.assert_called_once_with(
            "layer", attn.dsa_attn.swa_cache_layer.kv_cache, context.attn_metadata
        )

    def test_cache_only_write_waits_without_early_save(self):
        root = connector(shared=True)
        scope, _, attn = hook_scope(root)
        trace = []
        root.wait_for_layer_load.side_effect = lambda *_: trace.append("wait")
        attn.v41_impl = SimpleNamespace(
            _get_layer_metadata=lambda _: SimpleNamespace(swa=SimpleNamespace(num_actual_tokens=2)),
            _update_caches=Mock(side_effect=lambda *_: trace.append("write")),
        )
        load(ATTENTION, "dsa_v41_write_kv", scope)(torch.ones(4, 2), torch.empty(4, 2), "layer")
        self.assertEqual(trace, ["wait", "write"])
        self.assertEqual(attn.v41_impl._update_caches.call_args.args[1].shape[0], 2)
        root.save_kv_layer.assert_not_called()

    def test_load_failure_prevents_attention_and_save(self):
        root = connector(shared=True)
        scope, _, attn = hook_scope(root)
        attn.v41_impl = Mock()
        root.wait_for_layer_load.side_effect = RuntimeError("copy failed")
        with self.assertRaisesRegex(RuntimeError, "copy failed"):
            load(ATTENTION, "dsa_v41_forward", scope)(torch.ones(1), torch.ones(1), "layer")
        attn.v41_impl.forward.assert_not_called()
        root.save_kv_layer.assert_not_called()

    def test_standard_methods_keep_real_gva_save_and_load_fences(self):
        for method in (None, "dspark"):
            with self.subTest(method=method):
                worker, store, caches = make_cross_worker(self, bounded=True, method=method)
                pool = object.__new__(AscendStoreConnector)
                pool.use_layerwise = pool.use_cross_layer_kv = pool.can_put = True
                pool.connector_worker = worker
                pool.has_connector_metadata = lambda pool=pool: getattr(pool, "_connector_metadata", None) is not None
                root = connector(children=[connector(), connector(children=[pool])])
                scope, context, attn = hook_scope(root)
                save = metadata(worker)
                pool.bind_connector_metadata(save)
                for layer in range(12):
                    name = f"model.layers.{layer}.self_attn.v41_attn"
                    context.no_compile_layers[name] = attn
                    scope["maybe_save_kv_layer_to_connector"](name, list(caches.values()))
                pool.finish_forward()
                self.assertFalse(store.published)
                pool.wait_for_save()
                self.assertTrue(store.published)
                for cache in caches.values():
                    cache.zero_()
                restore = metadata(worker, load=True, save=False)
                pool.bind_connector_metadata(restore)
                pool.start_load_kv(context)
                for layer in range(12):
                    scope["wait_for_kv_layer_from_connector"](f"model.layers.{layer}.self_attn.v41_attn")
                pool.finish_forward()
                self.assertFalse(store.leases)
                self.assertTrue(all(int(cache[1].sum()) > 0 for cache in caches.values()))
                for child in root.sub_connectors:
                    child.wait_for_layer_load.assert_not_called()
                    child.save_kv_layer.assert_not_called()
