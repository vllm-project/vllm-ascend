import json
import sqlite3
import struct
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from typing import Any

from tools.ttft_diagnostic import analyze_round, decode_time_points


class TestTTFTDiagnostic(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.path = self.root / "gsm8k_details.jsonl"
        self.row: dict[str, Any] = {
            "id": 0,
            "success": True,
            "input_tokens": 3500,
            "output_tokens": 1500,
            "input": "[{'role': 'user', 'content': 'test prompt'}]",
            "time_points": [10.0, 10.5, 11.0, 12.0],
        }

    def analyze(self, rows):
        self.path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        return analyze_round(self.path, ["test prompt"])

    def test_stream_metrics(self):
        result = self.analyze([self.row])
        self.assertEqual(result["ttft_ms"]["mean"], 500)
        self.assertAlmostEqual(result["tpot_ms"]["mean"], 1500 / 1499)
        self.assertEqual(result["itl_ms"]["mean"], 750)

    def test_sqlite_timestamps(self):
        header = b"{'descr': '<f8', 'fortran_order': False, 'shape': (4,), }\n"
        blob = b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header
        blob += struct.pack("<4d", *self.row["time_points"])
        self.assertEqual(decode_time_points(blob), self.row["time_points"])
        (self.root / "db_data").mkdir()
        with closing(sqlite3.connect(self.root / "db_data/test.db")) as connection:
            connection.execute("CREATE TABLE numpy_store (id INTEGER, arr_blob BLOB)")
            connection.execute("INSERT INTO numpy_store VALUES (?, ?)", (1, blob))
            connection.commit()
        self.row.update(time_points={"__db_ref__": 1}, db_name="test.db")
        self.assertEqual(self.analyze([self.row])["ttft_ms"]["mean"], 500)

    def test_invalid_requests_fail_instead_of_producing_comparison(self):
        invalid = (
            {"input_tokens": 3501},
            {"output_tokens": 1499},
            {"success": False},
            {"input": "[{'role': 'user', 'content': 'different'}]"},
            {"time_points": [10, 11, 10.5]},
        )
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.analyze([{**self.row, **changes}])
        for rows in ([], [self.row, self.row]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.analyze(rows)


if __name__ == "__main__":
    unittest.main()
