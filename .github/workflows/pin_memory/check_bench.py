#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Gate on the h2d_d2h bench output (bench.log).

Emits ::error:: / ::notice:: workflow annotations and exits non-zero when the
bench results contradict the expected pin-memory behavior, so a degraded run
(e.g. the ~12 GB/s recognition-chain-broken state) fails the CI step instead
of passing green.

Usage: check_bench.py <log> [--expect-registered]
"""

import sys

ARMS = {"pageable", "torch", "native-unregistered", "native-registered"}


def parse(log_path):
    rows = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("arm,"):
                continue
            parts = line.split(",")
            # RESULT= json lines and anything else are skipped via the arm filter.
            if len(parts) == 6 and parts[0] in ARMS:
                rows.append(dict(zip(("arm", "size_mib", "h2d", "d2h", "torch_pinned", "acc_pinned"), parts)))
    return rows


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    expect_registered = "--expect-registered" in sys.argv[1:]
    if len(args) != 1:
        print("usage: check_bench.py <log> [--expect-registered]")
        sys.exit(2)
    rows = parse(args[0])
    arms = {r["arm"] for r in rows}
    expected = ARMS if expect_registered else ARMS - {"native-registered"}
    missing = expected - arms
    if missing:
        print(f"::error::bench output is missing arms: {sorted(missing)}")
        sys.exit(1)

    problems = 0
    for r in rows:
        if r["arm"] == "native-registered":
            h2d = float(r["h2d"])
            if r["torch_pinned"] != "True":
                # The dual signature: registered + is_pinned=True is the fast
                # path switch; False means the recognition chain is broken.
                print(
                    f"::error::native-registered {r['size_mib']}MiB has torch_is_pinned={r['torch_pinned']}: "
                    f"recognition chain broken (check /usr/local/Ascend/driver/version.info and the torch_npu patch)"
                )
                problems += 1
            if r["size_mib"] == "64":
                if h2d < 12.0:
                    print(
                        f"::error::native-registered 64MiB H2D {h2d:.2f} GB/s is below the 12 GB/s floor "
                        f"(expected ~23 GB/s fast path)"
                    )
                    problems += 1
                elif h2d < 20.0:
                    print(
                        f"::notice::native-registered 64MiB H2D {h2d:.2f} GB/s is in the slow-window range "
                        f"(fast path is >= 20)"
                    )
    if problems:
        print(f"::error::bench gate FAILED with {problems} problem(s)")
        sys.exit(1)
    print(f"bench gate PASSED ({len(rows)} rows, registered={'expected' if expect_registered else 'skipped'})")


if __name__ == "__main__":
    main()
