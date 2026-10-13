#!/usr/bin/env python3
"""Build/run independent main-op and metadata Y boundary checks via ACL."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cann", type=Path, default=Path(os.environ.get(
        "ASCEND_HOME_PATH", "/home/lch/CANN/950_0903/cann-9.2.0")))
    args = parser.parse_args()
    opp = Path(os.environ["ASCEND_CUSTOM_OPP_PATH"])
    source = Path(__file__).with_name("test_blocky_range_acl.cpp")
    with tempfile.TemporaryDirectory(prefix="gbsa-y-range-") as folder:
        executable = Path(folder) / "y_range"
        subprocess.run(["g++", "-std=c++17", "-O2", str(source), "-o", str(executable),
                        "-I" + str(opp / "op_api/include"), "-I" + str(args.cann / "include"),
                        "-L" + str(opp / "op_api/lib"), "-L" + str(args.cann / "lib64"),
                        "-lcust_opapi", "-lascendcl", "-lnnopbase"], check=True)
        subprocess.run([str(executable), str(args.device)], check=True)


if __name__ == "__main__":
    main()
