from tests.ut.base import TestBase
from vllm_ascend.quantization.quant_type import QuantType


class TestQuantType(TestBase):
    def test_enum_values(self):
        self.assertEqual(QuantType.NONE.value, 0)
        self.assertEqual(QuantType.W8A8.value, 1)
        self.assertEqual(QuantType.W4A8.value, 2)
        self.assertEqual(QuantType.MXFP8.value, 3)
        self.assertEqual(QuantType.W4A16.value, 4)
        self.assertEqual(QuantType.MXFP4.value, 5)

    def test_enum_members_count(self):
        members = list(QuantType)
        self.assertEqual(len(members), 6)

    def test_enum_identity(self):
        self.assertIs(QuantType.W8A8, QuantType.W8A8)
        self.assertIsNot(QuantType.W8A8, QuantType.W4A8)

    def test_enum_by_value(self):
        self.assertEqual(QuantType(1), QuantType.W8A8)
        self.assertEqual(QuantType(3), QuantType.MXFP8)

    def test_enum_by_name(self):
        self.assertEqual(QuantType["NONE"], QuantType.NONE)
        self.assertEqual(QuantType["W4A16"], QuantType.W4A16)
