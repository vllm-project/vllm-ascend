import os
import unittest
from unittest import mock

from vllm_ascend.ascend_forward_context import get_dispatcher_name


class TestGetDispatcherName(unittest.TestCase):

    def test_get_dispatcher_name(self):
        result = get_dispatcher_name(1, False)
        assert result == "TokenDispatcherWithAllGather"
        result = get_dispatcher_name(4, False)
        assert result == "TokenDispatcherWithAll2AllV"
        result = get_dispatcher_name(16, True)
        assert result == "TokenDispatcherWithAll2AllV"
        result = get_dispatcher_name(16, False)
        assert result == "TokenDispatcherWithMC2"
        with mock.patch.dict(os.environ,
                             {"VLLM_ENABLE_FUSED_EXPERTS_ALLGATHER_EP": "1"}):
            result = get_dispatcher_name(16, False)
            assert result == "TokenDispatcherWithAllGather"
