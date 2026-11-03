# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import unittest

from reference import decode_e4m3fn, decode_e8m0


class ReferenceTest(unittest.TestCase):
    def test_e4m3_known_encodings(self):
        for code, expected in ((0x00, 0.0), (0x01, 2**-9), (0x08, 2**-6), (0x38, 1.0), (0x7E, 448.0)):
            self.assertEqual(decode_e4m3fn(code), expected)
            self.assertEqual(decode_e4m3fn(code | 0x80), -expected)
        self.assertEqual(math.copysign(1.0, decode_e4m3fn(0x80)), -1.0)
        self.assertTrue(math.isnan(decode_e4m3fn(0x7F)))
        self.assertTrue(math.isnan(decode_e4m3fn(0xFF)))

    def test_e4m3_all_finite_positive_codes_are_ordered(self):
        values = [decode_e4m3fn(code) for code in range(0x7F)]
        self.assertTrue(all(left < right for left, right in zip(values, values[1:])))

    def test_e8m0_has_no_zero_encoding(self):
        for code in range(0xFF):
            self.assertEqual(decode_e8m0(code), 2.0 ** (code - 127))
        self.assertEqual(decode_e8m0(0), 2.0**-127)
        self.assertEqual(decode_e8m0(127), 1.0)
        self.assertTrue(math.isnan(decode_e8m0(255)))

    def test_invalid_bytes(self):
        for decoder in (decode_e4m3fn, decode_e8m0):
            for code in (-1, 256):
                with self.assertRaises(ValueError):
                    decoder(code)


if __name__ == "__main__":
    unittest.main()
