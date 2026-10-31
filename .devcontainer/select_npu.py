#!/usr/bin/env python3
# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# -------------------------------------------------------------------------
"""探测当前「没有进程运行」且健康状态为 OK 的空闲 NPU 卡。

在容器内（post-create 阶段）执行。容器 --privileged 且全设备挂载，
npu-smi info 能看到全系统（含其他容器）的 NPU 进程，判定准确。

用法：
    NPU_REQUEST_COUNT=2 python3 select_npu.py

行为：
    - stdout 只输出选中的卡号，逗号分隔（例如 "0,2"），供调用方直接使用。
    - 诊断日志一律输出到 stderr，不污染 stdout。
    - 退出码：0 表示成功（含「没有空闲卡」这一正常情况）；非 0 表示出错
      （未发现 /dev/davinci<N>、npu-smi 执行失败等）。
"""

import glob
import os
import subprocess
import sys

import regex as re


def detect_free_cards(count):
    """返回按卡号升序的空闲卡列表，最多 count 张。"""
    # 1. 从 /dev/davinci<N> 枚举容器内真实挂载的卡，不做固定数量假设。
    cards = set()
    for path in glob.glob("/dev/davinci[0-9]*"):
        m = re.fullmatch(r"/dev/davinci(\d+)", path)
        if m:
            cards.add(int(m.group(1)))
    cards = sorted(cards)
    if not cards:
        print("no /dev/davinci<N> device found", file=sys.stderr)
        sys.exit(1)

    # 2. 查询 npu-smi info。
    try:
        out = subprocess.run(
            ["npu-smi", "info"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except Exception as e:  # noqa: BLE001 - 此处需上报任何探测失败原因
        print(f"npu-smi info failed: {e}", file=sys.stderr)
        sys.exit(1)

    # 3. 解析板卡表头行的 Health，非 OK 的卡不参与分配。
    #    板卡表头形如：| 0     910B3    | OK    | ...（名称列非空，能命中）；
    #    Chip 行形如：| 0              | 0000:C1:00.0 | ...（名称列为空，不会命中）。
    bad_health = {
        int(m.group(1)) for m in re.finditer(r"^\|\s*(\d+)\s+\S+\s+\|\s*(\S+)", out, re.MULTILINE) if m.group(2) != "OK"
    }

    # 4. 空闲 = 进程区出现「No running processes found in NPU N」且健康 OK。
    free_set = {int(m.group(1)) for m in re.finditer(r"No running processes found in NPU\s+(\d+)", out)}
    free = [c for c in cards if c in free_set and c not in bad_health]
    return free[:count]


def main():
    count = int(os.environ.get("NPU_REQUEST_COUNT", "1"))
    chosen = detect_free_cards(count)

    # stdout 只输出卡号，供 shell 捕获。
    print(",".join(map(str, chosen)))

    if len(chosen) < count:
        print(
            f"only {len(chosen)} free card(s), requested {count}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
