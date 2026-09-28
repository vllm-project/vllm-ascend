# SPDX-License-Identifier: Apache-2.0
"""Print selected per-vector-core msprof PipeUtilization ranges."""

import argparse
import csv
import json
import statistics
from pathlib import Path

COMPONENTS = (
    "aiv_time(us)",
    "aiv_vec_time(us)",
    "aiv_mte2_time(us)",
    "aiv_mte3_time(us)",
    "aiv_scalar_time(us)",
    "aiv_scalar_wait_time(us)",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile_dirs", nargs="+", type=Path)
    args = parser.parse_args()
    for profile_dir in args.profile_dirs:
        paths = list(profile_dir.glob("OPPROF_*/PipeUtilization.csv"))
        if len(paths) != 1:
            raise ValueError(f"expected one PipeUtilization.csv in {profile_dir}, got {len(paths)}")
        with paths[0].open(newline="") as file:
            rows = list(csv.DictReader(file))
        vector_rows = [row for row in rows if row.get("aiv_time(us)", "NA") != "NA"]
        result = {"profile": str(profile_dir), "vector_cores": len(vector_rows)}
        for field in COMPONENTS:
            values = [float(row[field]) for row in vector_rows if row.get(field, "NA") != "NA"]
            if values:
                result[field] = {
                    "min": round(min(values), 3),
                    "median": round(statistics.median(values), 3),
                    "max": round(max(values), 3),
                }
            else:
                result[field] = None
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
