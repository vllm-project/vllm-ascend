#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
"""Deterministic GitCode URL parser. Extracts owner/repo/type/number from PR or Issue URLs.

Usage:
    python parse_gitcode_url.py "https://gitcode.com/cann/ops-math/pulls/123"
    # stdout → {"owner":"cann","repo":"ops-math","type":"pr","number":123}

    python parse_gitcode_url.py "https://gitcode.com/cann/ops-math/merge_requests/789"
    # stdout → {"owner":"cann","repo":"ops-math","type":"pr","number":789}

Exit codes: 0=success, 1=parse error.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from _logging import init_output_logger, init_error_logger

ALLOWED_DOMAIN = "gitcode.com"

URL_PATTERN = re.compile(
    r"https?://"
    r"(?P<domain>[^/]+)/"
    r"(?P<owner>[^/]+)/"
    r"(?P<repo>[^/]+)/"
    r"(?P<type>pull|pulls|merge_requests|issues)/(?P<number>\d+)",
    re.IGNORECASE,
)

TYPE_MAP = {
    "pull": "pr",
    "pulls": "pr",
    "merge_requests": "pr",
    "issues": "issue",
}


def parse_url(url: str) -> dict:
    """Parse a GitCode URL and return structured components.

    Args:
        url: Full GitCode PR or Issue URL.

    Returns:
        dict with keys: owner, repo, type ("pr"|"issue"), number

    Raises:
        ValueError: URL format invalid or domain mismatch.
    """
    url = url.strip().rstrip("/")

    match = URL_PATTERN.search(url)
    if not match:
        raise ValueError(
            f"Cannot parse URL: '{url}'. "
            f"Expected format: https://{ALLOWED_DOMAIN}/<owner>/<repo>/<pulls|issues>/<n>"
        )

    domain = match.group("domain")
    if domain != ALLOWED_DOMAIN:
        raise ValueError(
            f"Unsupported domain: {domain}. Only {ALLOWED_DOMAIN} is supported."
        )

    return {
        "owner": match.group("owner"),
        "repo": match.group("repo"),
        "type": TYPE_MAP[match.group("type").lower()],
        "number": int(match.group("number")),
    }


def main(argv: list[str] | None = None) -> int:
    OUTPUT_LOGGER = init_output_logger()
    ERROR_LOGGER = init_error_logger()

    parser = argparse.ArgumentParser(
        description="Parse GitCode PR/Issue URL into structured JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s https://gitcode.com/cann/ops-math/pulls/123
  %(prog)s https://gitcode.com/cann/ops-math/issues/456
  %(prog)s https://gitcode.com/cann/ops-math/merge_requests/789""",
    )
    parser.add_argument("url", help="GitCode PR or Issue URL")
    args = parser.parse_args(argv)

    try:
        result = parse_url(args.url)
    except ValueError as e:
        ERROR_LOGGER.error("ERROR: %s", e)
        return 1

    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
