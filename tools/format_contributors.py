# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
# Adapted from https://github.com/vllm-project/vllm/tree/main/tools
#
import argparse
import re
import sys
from datetime import datetime

p = re.compile(r"@(?P<user>[A-Za-z0-9-_]+)[^\`]*\`(?P<sha>[0-9a-fA-F]+)\`\s*[-–—]\s*(?P<date>.+)$")


def parse_lines(lines):
    items = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        m = p.search(ln)
        if not m:
            continue
        user = m.group("user")
        sha = m.group("sha")
        datestr = m.group("date").strip()
        try:
            dt = datetime.fromisoformat(datestr)
        except Exception:
            # fallback: try to parse common formats
            try:
                dt = datetime.strptime(datestr, "%Y/%m/%d")
            except Exception:
                continue
        items.append((dt, user, sha, datestr))
    return items


def main():
    ap = argparse.ArgumentParser(
        description="Format and sort contributor lines by date (newest first). Outputs markdown table by default."
    )
    ap.add_argument(
        "file", nargs="?", help="input file (default stdin), output from collect_user_first_contribution.sh"
    )
    ap.add_argument("--start", type=int, default=1, help="minimum number for table (oldest row will have this number)")
    ap.add_argument("--repo", default="vllm-project/vllm-ascend", help="repo used for commit links")
    args = ap.parse_args()

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            lines = f.readlines()
    else:
        lines = sys.stdin.readlines()

    items = parse_lines(lines)
    # sort newest first
    items.sort(key=lambda x: x[0], reverse=True)

    # Outputs markdown table (sorted by date), the minimum number is args.start
    count = len(items)
    if count == 0:
        return
    n = args.start + count - 1
    for dt, user, sha, datestr in items:
        short = sha[:7]
        date_short = dt.strftime("%Y/%m/%d")
        user_url = f"https://github.com/{user}"
        commit_url = f"https://github.com/{args.repo}/commit/{sha}"
        print(f"| {n} | [@{user}]({user_url}) | {date_short} | [{short}]({commit_url}) |")
        n -= 1


if __name__ == "__main__":
    main()
