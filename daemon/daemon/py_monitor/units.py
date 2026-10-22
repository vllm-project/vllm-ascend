from __future__ import annotations

GB = 1024**3
MB = 1024**2
KB = 1024


def fmt_bytes(num: int) -> str:
    if num < MB:
        return f"{num / KB:.0f}KB"
    if num < GB:
        return f"{num / MB:.0f}MB"
    return f"{num / GB:.2f}GB"
