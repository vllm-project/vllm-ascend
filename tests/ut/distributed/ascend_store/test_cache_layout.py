# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import unittest
from types import SimpleNamespace

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401

# isort: split
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.cache_layout import describe_cache_bindings


class FakeTensor:
    dtype = "float16"

    def __init__(self, shape=(4, 2, 8, 3), strides=(48, 24, 3, 1), address=1000, storage_size=384):
        self.shape = shape
        self._strides = strides
        self._address = address
        self._storage_size = storage_size

    def stride(self):
        return self._strides

    def element_size(self):
        return 2

    def data_ptr(self):
        return self._address

    def untyped_storage(self):
        return SimpleNamespace(data_ptr=lambda: 1000, nbytes=lambda: self._storage_size)


class TestCacheLayout(unittest.TestCase):
    def config(self, layers=("layer.0", "draft.0")):
        return SimpleNamespace(
            num_blocks=2,
            kv_cache_groups=[
                SimpleNamespace(layer_names=[layer], kv_cache_spec=SimpleNamespace(block_size=8)) for layer in layers
            ],
        )

    def test_preserves_real_components_draft_and_group_membership(self):
        tensors = {"layer.0": (FakeTensor(), FakeTensor()), "draft.0": FakeTensor()}
        bindings = describe_cache_bindings(self.config(), tensors)
        self.assertEqual(len(bindings["layer.0"]), 2)
        self.assertEqual(bindings["draft.0"][0].group_id, 1)
        self.assertEqual(bindings["layer.0"][0].block_size_scale, 2)
        self.assertEqual(bindings["layer.0"][0].page_span, 96)
        self.assertNotIn("address", bindings["layer.0"][0].to_wire())

    def test_hnd_page_coordinates_include_physical_block_and_head_origin(self):
        binding = describe_cache_bindings(self.config(("layer.0",)), {"layer.0": FakeTensor()})["layer.0"][0]
        page = binding.token_page(2, token_start=32, head_start=4, token_axis=2, head_axis=1, format_id="k-fp16")
        self.assertEqual(page.address, 1000 + 2 * 96)
        self.assertEqual((page.cache.token_start, page.cache.token_end), (32, 40))
        self.assertEqual((page.cache.head_start, page.cache.head_end), (4, 6))
        self.assertEqual(page.cache.vector_bytes, 6)
        self.assertEqual((page.cache.token_stride, page.cache.head_stride), (6, 48))

    def test_mla_component_keeps_its_real_offset_and_stride(self):
        # View occupies a subregion of each padded page, not a full allocation.
        tensor = FakeTensor(shape=(4, 8, 3), strides=(32, 3, 1), address=1008, storage_size=256)
        binding = describe_cache_bindings(self.config(("layer.0",)), {"layer.0": tensor})["layer.0"][0]
        page = binding.token_page(1, token_start=8, head_start=0, token_axis=1, head_axis=None, format_id="latent-fp16")
        self.assertEqual(page.address, 1072)
        self.assertEqual(page.cache.vector_bytes, 6)
        self.assertEqual(binding.byte_strides[0], 64)
        self.assertEqual(binding.page_span, 48)

    def test_owner_selection_does_not_export_scratch_layers(self):
        bindings = describe_cache_bindings(
            self.config(), {"layer.0": FakeTensor(), "draft.0": FakeTensor()}, owned_layers={"draft.0"}
        )
        self.assertEqual(set(bindings), {"draft.0"})

    def test_kvpp_can_register_only_its_owned_layers_but_cannot_omit_an_owner(self):
        bindings = describe_cache_bindings(self.config(), {"draft.0": FakeTensor()}, owned_layers={"draft.0"})
        self.assertEqual(set(bindings), {"draft.0"})
        with self.assertRaisesRegex(ValueError, "Registered cache layers"):
            describe_cache_bindings(self.config(), {"draft.0": FakeTensor()}, owned_layers={"layer.0"})

    def test_missing_layer_or_duplicate_group_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Registered cache layers"):
            describe_cache_bindings(self.config(), {"layer.0": FakeTensor()})
        with self.assertRaisesRegex(ValueError, "multiple groups"):
            describe_cache_bindings(self.config(("layer.0", "layer.0")), {"layer.0": FakeTensor()})

    def test_invalid_physical_count_or_out_of_bounds_view_is_rejected(self):
        for tensor in (FakeTensor(shape=(3, 2, 8, 3)), FakeTensor(address=1010), FakeTensor(strides=(48, 24, 3, -1))):
            with self.subTest(tensor=tensor), self.assertRaises(ValueError):
                describe_cache_bindings(self.config(("layer.0",)), {"layer.0": tensor})

    def test_axes_and_block_bounds_are_checked(self):
        binding = describe_cache_bindings(self.config(("layer.0",)), {"layer.0": FakeTensor()})["layer.0"][0]
        for block, token_axis, head_axis in ((4, 2, 1), (0, 0, 1), (0, 2, 2)):
            with self.subTest(block=block, token_axis=token_axis, head_axis=head_axis), self.assertRaises(ValueError):
                binding.token_page(
                    block, token_start=0, head_start=0, token_axis=token_axis, head_axis=head_axis, format_id="fp16"
                )
