# from unittest.mock import MagicMock, patch

# import torch
# from torch import nn
# from vllm.distributed.parallel_state import GroupCoordinator
# from vllm.model_executor.layers.linear import LinearBase

from tests.ut.base import TestBase
from vllm_ascend.torchair.torchair_mla import (
    AscendSFATorchairBackend, AscendSFATorchairImpl, AscendSFATorchairMetadata,
    AscendSFATorchairMetadataBuilder)


class TestAscendSFATorchairBackend(TestBase):

    def test_get_name(self):
        self.assertEqual(AscendSFATorchairBackend.get_name(),
                         "ASCEND_SFA_TORCHAIR")

    def test_get_metadata_cls(self):
        self.assertEqual(AscendSFATorchairBackend.get_metadata_cls(),
                         AscendSFATorchairMetadata)

    def test_get_builder_cls(self):
        self.assertEqual(AscendSFATorchairBackend.get_builder_cls(),
                         AscendSFATorchairMetadataBuilder)

    def test_get_kv_cache_shape(self):
        result = AscendSFATorchairBackend.get_kv_cache_shape(2, 4, 8, 128)
        self.assertEqual(result, (2, 4, 8, 128))

    def test_get_impl_cls(self):
        result = AscendSFATorchairBackend.get_impl_cls()
        self.assertEqual(result, AscendSFATorchairImpl)
