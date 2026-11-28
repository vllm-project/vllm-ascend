# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sqlite3
import unittest
from types import SimpleNamespace

import benchmark
import profile_qsfa


class WorkloadTest(unittest.TestCase):
    def test_packed_rope_reference_workload(self):
        storage = benchmark.prepare_storage(1, "cpu", 93211, 64)
        inputs, metadata = benchmark.make_inputs(1, 128, 64, 16, 93211, storage, "token")
        rows, golden = benchmark.selected_golden(inputs, metadata, 1, benchmark.DEFAULT_SCALE)
        self.assertEqual(rows, slice(0, 6))
        self.assertEqual(tuple(golden.shape), (6, 64, 512))
        self.assertTrue(golden.isfinite().all())
        record = benchmark.workload_counts(inputs, metadata, 16, "token")
        self.assertEqual(record, {"valid_sparse_pairs": 96, "useful_flops": 13_369_344, "logical_io_bytes": 898_956})
        record["native_us"] = 100
        benchmark.add_utilization(record, 280, 1600)
        self.assertAlmostEqual(record["native_useful_mfu_pct"], 100 * 13_369_344 / 28_000_000_000)
        self.assertAlmostEqual(record["native_logical_mbu_pct"], 100 * 898_956 / 160_000_000)


class ProfilerTest(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(":memory:")
        self.addCleanup(self.db.close)
        self.db.execute("CREATE TABLE STRING_IDS (id INTEGER, value TEXT)")

    def test_pipe_matches_reused_graph_task_by_replay_end(self):
        self.db.executescript("""
            CREATE TABLE TASK (startNs, endNs, deviceId, modelId, streamId, globalTaskId, connectionId, taskType);
            CREATE TABLE CANN_API (startNs, connectionId, name);
            CREATE TABLE COMPUTE_TASK_INFO (globalTaskId, name);
            CREATE TABLE MSTX_EVENTS (startNs, endNs, message);
            CREATE TABLE TASK_PMU_INFO (timestampNs, name, value, globalTaskId);
            INSERT INTO STRING_IDS VALUES (1,'MODEL_EXECUTE'), (2,'MODEL_WAIT_COMPLETE'), (3,'KERNEL_AICORE'),
                (4,'aclmdlRIExecuteAsync'), (5,'QSFA_PIPE_B1_native_0'), (6,'QSFA_PIPE_B1_candidate_0'),
                (7,'aic_total_time'), (8,'aic_mac_time'), (9,'kernel');
            INSERT INTO CANN_API VALUES (1500,11,4), (11500,12,4);
            INSERT INTO MSTX_EVENTS VALUES (1000,2000,5), (11000,12000,6);
            INSERT INTO COMPUTE_TASK_INFO VALUES (99,9);
            INSERT INTO TASK VALUES (14000,15000,0,3,1,101,11,1), (19000,20000,0,3,1,102,11,2),
                (15500,17500,0,3,2,99,501,3),
                (24000,25000,0,3,1,101,12,1), (29000,30000,0,3,1,102,12,2),
                (25500,27500,0,3,2,99,502,3);
            INSERT INTO TASK_PMU_INFO VALUES (17512,7,1000,99), (17512,8,200,99),
                (27502,7,1200,99), (27502,8,900,99);
        """)
        manifest = [
            {"batch": 1, "tokens": 6, "implementation": name, "label": f"QSFA_PIPE_B1_{name}_0"}
            for name in ("native", "candidate")
        ]
        native, candidate = profile_qsfa.pipe_samples(self.db, manifest)
        self.assertEqual(native["profile_device_span_us"], 2)
        self.assertEqual(native["aic_mac_pct"], 20)
        self.assertEqual(candidate["aic_mac_pct"], 75)
        self.assertEqual(candidate["kernels"][0]["pmu"]["match_error_ns"], 2)
        self.assertEqual(candidate["profile_tasks_with_aiv"], 0)
        self.assertNotIn("aiv_vec_pct", candidate)

    def test_hbm_sums_four_units_and_converts_units(self):
        self.db.execute("CREATE TABLE HBM (deviceId, timestampNs, bandwidth, hbmId, type)")
        self.db.executemany("INSERT INTO STRING_IDS VALUES (?,?)", [(1, "hbmReadBw"), (2, "hbmWriteBw")])
        self.db.executemany(
            "INSERT INTO HBM VALUES (?,?,?,?,?)",
            (
                (7, timestamp, bandwidth, unit, direction)
                for timestamp in range(0, 1_000_000_000, 20_000_000)
                for unit in range(4)
                for direction, bandwidth in ((1, 100e9), (2, 50e9))
            ),
        )
        window = {
            "batch": 1,
            "tokens": 6,
            "implementation": "native",
            "label": "QSFA_HBM_B1_native_0",
            "start_monotonic_ns": 200_000_000,
            "end_monotonic_ns": 600_000_000,
            "start_realtime_ns": 1_000_200_000_000,
            "end_realtime_ns": 1_000_600_000_000,
            "forwards": 40,
        }
        options = SimpleNamespace(min_hbm_samples=10, trim_ms=50, peak_hbm_gb_s=1600, hbm_device_id=None)
        (sample,) = profile_qsfa.hbm_samples(self.db, [window], options)
        self.assertEqual(sample["device_id"], 7)
        self.assertEqual(sample["clock"], "monotonic")
        self.assertAlmostEqual(sample["hbm_read_gb_s"], 400)
        self.assertAlmostEqual(sample["hbm_write_gb_s"], 200)
        self.assertAlmostEqual(sample["physical_mbu_pct"], 37.5)
        self.assertAlmostEqual(sample["hbm_bytes_per_forward_est"], 6e9, delta=1e-3)


if __name__ == "__main__":
    unittest.main()
