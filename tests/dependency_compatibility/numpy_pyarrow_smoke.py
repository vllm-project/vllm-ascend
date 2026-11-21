# SPDX-License-Identifier: Apache-2.0
"""Opt-in CPU smoke tests for a resolved dependency candidate environment."""

import io
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


class NumpyPyArrowCompatibilityTest(unittest.TestCase):
    def test_numpy_arrow_roundtrip(self):
        for dtype in (np.int64, np.float32):
            with self.subTest(dtype=dtype):
                values = np.arange(8, dtype=dtype)
                np.testing.assert_array_equal(pa.array(values).to_numpy(), values)

    def test_parquet_roundtrip(self):
        values = np.arange(8, dtype=np.int64)
        table = pa.table({"values": pa.array(values)})
        stream = io.BytesIO()
        pq.write_table(table, stream)
        # Use the single-file reader; this probe does not exercise datasets.
        # Keeping ownership scoped also avoids background scanner teardown.
        with pq.ParquetFile(io.BytesIO(stream.getvalue())) as parquet:
            restored = parquet.read()
        self.assertTrue(restored.equals(table))


if __name__ == "__main__":
    unittest.main()
