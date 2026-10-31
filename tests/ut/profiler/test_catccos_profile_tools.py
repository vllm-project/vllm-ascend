# SPDX-License-Identifier: Apache-2.0

import csv
import importlib.util
from pathlib import Path


def load_extract_module():
    script = Path(__file__).parents[3] / "tools/catccos_profiling/extract_ranges.py"
    spec = importlib.util.spec_from_file_location("extract_ranges", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extract_ranges_preserves_shape_and_supports_binding_ranges(tmp_path):
    module = load_extract_module()
    csv_path = tmp_path / "operator_details.csv"
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "Name",
                "Host Total Duration(us)",
                "Device Total Duration(us)",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "Name": "vllm_ascend.moe.catccos.host_moe_body[M=2,H=2048,topK=8]",
                    "Host Total Duration(us)": "10",
                    "Device Total Duration(us)": "0",
                },
                {
                    "Name": "catccos.a5.binding.rank_barrier_host[M=2,H=2048,N=1536,topK=8]",
                    "Host Total Duration(us)": "4",
                    "Device Total Duration(us)": "0",
                },
                {
                    "Name": "catccos.a5.binding.rank_barrier_host[M=2,H=2048,N=1536,topK=8]",
                    "Host Total Duration(us)": "6",
                    "Device Total Duration(us)": "0",
                },
                {
                    "Name": "unrelated",
                    "Host Total Duration(us)": "100",
                    "Device Total Duration(us)": "0",
                },
            ]
        )

    rows = module.summarize(module.read_samples(csv_path))

    assert len(rows) == 2
    barrier = next(row for row in rows if "rank_barrier_host" in row["name"])
    assert barrier["count"] == 2
    assert barrier["host_p50_us"] == 5
    assert "M=2,H=2048,N=1536,topK=8" in barrier["name"]
