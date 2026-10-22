#!/usr/bin/env python3
"""Check HTTP links in Markdown files.

Extracts all URLs from markdown files and verifies they return a valid
HTTP status code. Skips URLs in fenced code blocks and inline code.

Usage:
    python check_md_links.py <file_or_dir> [--exclude <pattern>]
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import regex as re
import requests

URL_RE = re.compile(r'https?://[^\s\)"\'`\]>]+')
FENCE_RE = re.compile(r"^(`{3,}|~{3,})", re.MULTILINE)
INLINE_CODE_RE = re.compile(r"`[^`]+`")

TIMEOUT = 20
RETRY_COUNT = 3
RETRY_DELAY = 5
ALIVE_STATUS_CODES = {200, 206, 301, 302, 307, 308, 401, 403, 405}
IGNORE_PATTERNS = ["localhost", "127.0.0.1"]


def extract_urls(content: str) -> set[str]:
    """Extract URLs from markdown content, excluding code blocks."""
    segments = []
    fence_matches = list(FENCE_RE.finditer(content))
    if not fence_matches:
        segments.append((content, False))
    else:
        pos = 0
        in_code = False
        fence_marker = None
        for match in fence_matches:
            marker = match.group(1)
            if not in_code:
                if match.start() > pos:
                    segments.append((content[pos : match.start()], False))
                pos = match.end()
                in_code = True
                fence_marker = marker[0]
            elif marker[0] == fence_marker:
                segments.append((content[pos : match.end()], True))
                pos = match.end()
                in_code = False
                fence_marker = None
        if pos < len(content):
            segments.append((content[pos:], in_code))

    urls = set()
    for seg_text, is_code in segments:
        if is_code:
            continue
        seg_text = INLINE_CODE_RE.sub("", seg_text)
        for match in URL_RE.finditer(seg_text):
            url = match.group().rstrip(".,;!")
            if not any(p in url for p in IGNORE_PATTERNS):
                urls.add(url)
    return urls


def check_url(url: str) -> tuple[str, bool, str]:
    """Check a single URL. Returns (url, is_alive, message)."""
    for attempt in range(RETRY_COUNT):
        try:
            resp = requests.head(
                url,
                timeout=TIMEOUT,
                allow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0 link-checker"},
            )
            if resp.status_code in ALIVE_STATUS_CODES:
                return (url, True, f"OK ({resp.status_code})")
            if resp.status_code == 405:
                try:
                    resp = requests.get(
                        url,
                        timeout=TIMEOUT,
                        allow_redirects=True,
                        headers={"User-Agent": "Mozilla/5.0 link-checker"},
                        stream=True,
                    )
                    if resp.status_code in ALIVE_STATUS_CODES:
                        return (url, True, f"OK ({resp.status_code})")
                    resp.close()
                except requests.RequestException:
                    pass
        except requests.RequestException as exc:
            if attempt == RETRY_COUNT - 1:
                return (url, False, str(exc))
    return (url, False, f"HTTP {resp.status_code}")


def collect_files(paths: list[str], excludes: list[str]) -> list[Path]:
    """Collect markdown files from given paths."""
    files = []
    for path_str in paths:
        path = Path(path_str)
        if path.is_file() and path.suffix == ".md":
            if not any(e in str(path) for e in excludes):
                files.append(path)
        elif path.is_dir():
            for md in path.rglob("*.md"):
                if not any(e in str(md) for e in excludes):
                    files.append(md)
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description="Check HTTP links in Markdown files.")
    parser.add_argument("paths", nargs="+", help="Markdown files or directories to check")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Path patterns to exclude",
    )
    args = parser.parse_args()

    files = collect_files(args.paths, args.exclude)
    if not files:
        print("No markdown files found.")
        return 0

    print(f"Checking {len(files)} markdown file(s)...")

    all_urls = {}
    for md_file in files:
        content = md_file.read_text(encoding="utf-8")
        urls = extract_urls(content)
        for url in urls:
            if url not in all_urls:
                all_urls[url] = md_file

    print(f"Found {len(all_urls)} unique URL(s) to check.\n")

    failed = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(check_url, url): url for url in all_urls}
        for future in as_completed(futures):
            url, is_alive, message = future.result()
            status = "OK" if is_alive else "FAIL"
            md_file = all_urls[url]
            print(f"  [{status}] {url} - {message}")
            if not is_alive:
                failed.append((url, md_file, message))

    if failed:
        print(f"\n{'=' * 60}")
        print(f"Failed links ({len(failed)}):")
        for url, md_file, message in failed:
            print(f"  {md_file}: {url} - {message}")
        print(f"{'=' * 60}")
        return 1

    print(f"\nAll {len(all_urls)} links passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
