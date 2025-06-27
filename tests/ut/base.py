import unittest

from vllm_ascend.utils import adapt_patch


class TestBase(unittest.TestCase):

    def setUp(self):
        # adapt patch by default.
        adapt_patch(True)
        adapt_patch()
        super().setUp()
