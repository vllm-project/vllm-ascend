#!/usr/bin/env bash
# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# -------------------------------------------------------------------------
# activate_npu.sh - 探测空闲 NPU 卡并激活 ASCEND_RT_VISIBLE_DEVICES
#
# 独立脚本，可在任意目录（当前工作目录）下调用：依赖脚本自身位置定位同级
# select_npu.py，而非硬编码容器路径，因此不限定执行目录。
#
# 职责：
#   调用 select_npu.py 探测「没有进程运行」且健康状态为 OK 的空闲卡，
#   把卡号写入 ASCEND_RT_VISIBLE_DEVICES 并注入当前用户的 shell 启动文件
#   ($HOME/.bashrc 与 $HOME/.bash_profile)，使 CANN 运行时只枚举这些卡。
#
# 用法：
#   bash activate_npu.sh            # 默认探测 1 张空闲卡
#   NPU_REQUEST_COUNT=2 bash activate_npu.sh
#
# 说明：
#   - 脚本为降级语义：探测失败 / 无空闲卡时仅告警，退出码仍为 0，便于
#     嵌入 post-create 等不阻塞主流程的上下文。
#   - 写入启动文件是「下次新 shell」生效；如需当前终端立即生效，可执行
#     `export ASCEND_RT_VISIBLE_DEVICES=<脚本打印的卡号>`。
# -------------------------------------------------------------------------

set -uo pipefail

# 脚本自身所在目录，与 select_npu.py 同级，用于跨目录定位。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DETECTOR="$SCRIPT_DIR/select_npu.py"
COUNT="${NPU_REQUEST_COUNT:-1}"

log()  { printf '[activate-npu] %s\n' "$*"; }
warn() { printf '[activate-npu] warning: %s\n' "$*" >&2; }

if ! command -v npu-smi >/dev/null 2>&1 || ! command -v python3 >/dev/null 2>&1; then
    warn "npu-smi or python3 not available; skipping NPU device filter"
    exit 0
fi

if [ ! -f "$DETECTOR" ]; then
    warn "detector script not found: $DETECTOR"
    exit 0
fi

ids="$(NPU_REQUEST_COUNT="$COUNT" python3 "$DETECTOR")" || {
    warn "NPU free-card detection failed; ASCEND_RT_VISIBLE_DEVICES will not be set"
    exit 0
}

if [ -z "$ids" ]; then
    warn "no free NPU card found; ASCEND_RT_VISIBLE_DEVICES will not be set"
    exit 0
fi

MARKER_BEGIN="# >>> mindstudio devcontainer npu-free >>>"
MARKER_END="# <<< mindstudio devcontainer npu-free <<<"
LINE="export ASCEND_RT_VISIBLE_DEVICES=\"$ids\""

for rc in "$HOME/.bashrc" "$HOME/.bash_profile"; do
    touch "$rc"
    # 先删除旧块再追加，避免重复执行后新旧卡号并存。
    sed -i "/${MARKER_BEGIN}/,/${MARKER_END}/d" "$rc" 2>/dev/null || true
    {
        printf '\n%s\n' "$MARKER_BEGIN"
        printf '%s\n' "$LINE"
        printf '%s\n' "$MARKER_END"
    } >> "$rc"
done

log "NPU device filter activated: ASCEND_RT_VISIBLE_DEVICES=$ids"