# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of the production attention hooks and standard connector methods."""

import unittest
from types import MethodType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import torch

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402

# isort: split
from tests.ut.attention.test_mla_layerwise_load_cpu import load
from tests.ut.distributed.ascend_store.test_shared_source_layerwise import make_shared_worker, metadata
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import AscendStoreConnector

UTILS = "vllm_ascend/attention/utils.py"
ATTENTION = "vllm_ascend/attention/dsa_v41.py"


def connector(*, shared=False, bound=True, children=()):
    return SimpleNamespace(
        supports_composite_kv_loads=shared,
        supports_composite_kv_saves=shared,
        sub_connectors=children,
        has_connector_metadata=Mock(return_value=bound),
        wait_for_layer_load=Mock(),
        save_kv_layer=Mock(),
    )


def hook_scope(root, *, composite=True):
    attn = SimpleNamespace(kv_transfer_cache_names=("cache",)) if composite else SimpleNamespace()
    context = SimpleNamespace(attn_metadata=object(), no_compile_layers={"layer": attn})
    scope = {
        "get_forward_context": lambda: context,
        "get_kv_transfer_group": lambda: root,
        "has_kv_transfer_group": Mock(return_value=True),
        "is_v1_kv_transfer_group": Mock(return_value=True),
        "record_attention_prefetch_start": Mock(),
    }
    for name in (
        "_kv_layer_transfer_connectors",
        "_supports_attention_prefetch_gate",
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

    def test_cut_cache_only_then_normal_forward_saves_once(self):
        events = []

        def prepare_inputs_and_caches(*args):
            events.append("final-write")
            return None, None

        layer_metadata = SimpleNamespace(
            swa=SimpleNamespace(num_actual_tokens=1),
            positions=torch.zeros(1),
            rope=lambda *args: (None, None),
        )
        attn = SimpleNamespace(
            v41_layer_name="cut",
            n_local_heads=1,
            head_dim=1,
            rotary_emb=SimpleNamespace(layername="rope"),
            dsa_attn=SimpleNamespace(
                swa_cache_layer=SimpleNamespace(kv_cache=[]),
                dsa_attn=SimpleNamespace(impl=SimpleNamespace()),
            ),
            v41_impl=SimpleNamespace(
                _get_layer_metadata=lambda _: layer_metadata,
                _update_caches=lambda *args: events.append("cache-only-write"),
                _prepare_inputs_and_caches=prepare_inputs_and_caches,
                _select_sparse_indices=Mock(),
                _forward_attention=Mock(),
                _project_output=lambda *args, **kwargs: events.append("project"),
            ),
        )
        ctx = SimpleNamespace(attn_metadata={}, no_compile_layers={"cut": attn})
        scope: dict[str, Any] = dict(
            torch=torch,
            AscendDSAV41Impl=SimpleNamespace(_apply_rotary=Mock()),
            oproj_tp_enable=lambda: False,
            get_forward_context=lambda: ctx,
            wait_for_kv_layer_from_connector=Mock(side_effect=lambda _: events.append("wait")),
            maybe_save_kv_layer_to_connector=Mock(side_effect=lambda *args: events.append("save")),
        )
        attn.v41_impl.forward = MethodType(load(ATTENTION, "forward", scope, "AscendDSAV41Impl"), attn.v41_impl)
        load(ATTENTION, "dsa_v41_write_kv", scope)
        load(ATTENTION, "dsa_v41_forward", scope)
        hidden = torch.ones(1)
        scope["dsa_v41_write_kv"](hidden, hidden, "cut")
        scope["dsa_v41_forward"](hidden, hidden, "cut")
        self.assertEqual(events, ["wait", "cache-only-write", "wait", "final-write", "project", "save"])
        scope["maybe_save_kv_layer_to_connector"].assert_called_once_with("cut", attn.dsa_attn.swa_cache_layer.kv_cache)

        for num_tokens in (1, 0):
            with self.subTest(num_tokens=num_tokens):
                events.clear()
                layer_metadata.swa.num_actual_tokens = num_tokens
                attn.v41_impl.forward(attn, None, hidden, hidden)
                self.assertEqual(events, ["wait", "final-write", "project", "save"])
                scope["wait_for_kv_layer_from_connector"].assert_called_with("cut")

        events.clear()
        ctx.attn_metadata = None
        scope["dsa_v41_forward"](hidden, hidden, "cut")
        self.assertEqual(events, [])

    def test_standard_connector_drives_full_sparse_sequence(self):
        worker, store, _ = make_shared_worker(self, bounded=True, draft=True)
        meta = metadata(worker)
        conn = AscendStoreConnector.__new__(AscendStoreConnector)
        conn.use_layerwise = conn.can_put = True
        conn.connector_worker = worker
        conn._mamba_state = None
        conn._current_step_has_real_forward = False
        conn.bind_connector_metadata(meta)
        conn.start_load_kv(SimpleNamespace())
        for layer in range(worker.num_layers):
            name = f"model.layers.{layer}.v41_attn" if layer < 12 else f"model.mtp.layers.{layer - 12}.v41_attn"
            conn.wait_for_layer_load(name)
            conn.save_kv_layer(name, [], None)
        self.assertTrue(store.published)  # the final layer hook completed the save
        conn.finish_forward()
        conn.wait_for_save()
        conn.get_finished(set())
        self.assertTrue(store.published)
        self.assertFalse(store.leases)

    def test_composite_save_preserves_configured_order(self):
        events = []
        provider, pd = connector(shared=True), connector(shared=True)
        provider.save_kv_layer.side_effect = lambda *args: events.append("pool")
        pd.save_kv_layer.side_effect = lambda *args: events.append("pd")
        root = connector(children=[pd, provider])
        scope, _, _ = hook_scope(root)
        scope["maybe_save_kv_layer_to_connector"]("layer", [])
        self.assertEqual(events, ["pd", "pool"])
