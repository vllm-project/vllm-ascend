#!/usr/bin/env python3
"""Check rendered Markdown links with concurrent checks and a late GET retry sweep.

No vLLM or accelerator imports are needed. See docs/requirements-linkcheck.txt
and the contribution testing guide for standalone usage.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import ipaddress
import json
import math
import time
from collections import Counter
from contextlib import suppress
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html import escape
from html.parser import HTMLParser
from itertools import zip_longest
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

import httpx
import markdown
import regex as re

MARKDOWN_EXTENSIONS = (
    "admonition",
    "attr_list",
    "def_list",
    "footnotes",
    "md_in_html",
    "tables",
    "pymdownx.details",
    "pymdownx.superfences",
    "pymdownx.tabbed",
)
REDIRECT_CODES = frozenset({301, 302, 303, 307, 308})
GET_FALLBACK_CODES = frozenset({401, 403, 404, 405, 501})
TRANSIENT_CODES = frozenset({408, 425, 500, 502, 503, 504})
MAX_REDIRECTS = 5
UNREACHABLE_EXAMPLE_IP = "192.0.0.1"
TEMPLATE_RE = re.compile(r"\$\{|\{\{|[<>]")
USER_AGENT = "Mozilla/5.0 (compatible; vllm-ascend-linkcheck; +https://github.com/vllm-project/vllm-ascend)"

# Exact hostnames only; subdomains must be listed separately.
SKIP_DOMAINS = {
    "deepwiki.com": "Vercel browser challenge blocks automated HTTP checks; verify the page manually.",
}


@dataclass(frozen=True)
class Settings:
    global_concurrency: int = 10
    initial_attempts: int = 3
    recheck_attempts: int = 4
    request_timeout: float = 15
    connect_timeout: float = 5
    total_timeout: float = 1200
    retry_delay: float = 5
    excludes: tuple[str, ...] = ("docs/source/zh",)


class LinkParser(HTMLParser):
    """Read only clickable/rendered targets; code and comments are not links."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.urls: set[str] = set()
        self.code_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"pre", "code", "script", "style"}:
            self.code_depth += 1
        if self.code_depth:
            return
        key = {"a": "href", "img": "src"}.get(tag)
        value = dict(attrs).get(key)
        if value and value.lower().startswith(("http:", "https:")):
            # Keep malformed targets for an explicit broken-link result.
            with suppress(httpx.InvalidURL, ValueError):
                value = str(httpx.URL(value).copy_with(fragment=None))
            self.urls.add(value)

    def handle_endtag(self, tag):
        if tag in {"pre", "code", "script", "style"} and self.code_depth:
            self.code_depth -= 1

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)


def extract_urls(content: str) -> set[str]:
    parser = LinkParser()
    # These syntax extensions mirror the block/link syntax in mkdocs.yml.
    # Do not execute MkDocs hooks, macros, or custom fence renderers.
    parser.feed(markdown.markdown(content, extensions=MARKDOWN_EXTENSIONS))
    return parser.urls


def collect_links(paths: list[str], settings: Settings) -> dict[str, list[str]]:
    files: set[Path] = set()
    for name in paths:
        path = Path(name)
        if not path.exists():
            raise ValueError(f"input does not exist: {name}")
        files.update(path.rglob("*.md") if path.is_dir() else [path])
    links: dict[str, list[str]] = {}
    for path in sorted(files):
        source = path.as_posix()
        if path.suffix != ".md" or any(f"/{exclude.strip('/')}/" in f"/{source}/" for exclude in settings.excludes):
            continue
        for url in sorted(extract_urls(path.read_text(encoding="utf-8"))):
            links.setdefault(url, []).append(source)
    # Spread PR/issue/commit pages throughout a host's scan rather than making
    # long lexical runs against one route. The order is repeatable across runs.
    return dict(sorted(links.items(), key=lambda item: hashlib.sha256(item[0].encode()).digest()))


def example_reason(url: str) -> str:
    decoded = unquote(url)
    if TEMPLATE_RE.search(decoded):
        return "template/example URL"
    host = urlsplit(url).hostname
    if host and (host.lower() == "localhost" or host.lower().endswith(".localhost")):
        return "local service example"
    try:
        address = ipaddress.ip_address(host or "")
    except ValueError:
        return ""
    if address.is_private or address.is_loopback or address.is_unspecified or str(address) == UNREACHABLE_EXAMPLE_IP:
        return "local service example"
    return ""


def is_contributor_profile(url: str, sources: list[str]) -> bool:
    # Historical attribution tolerates account renames/deletions, but a reference
    # from any other document must still be checked.
    return bool(re.fullmatch(r"https://github\.com/[^/?#]+/?", url)) and all(
        f"/{source}".endswith("/docs/source/community/contributors.md") for source in sources
    )


def retry_after(value: str | None, now: datetime | None = None) -> float:
    if not value:
        return 0
    try:
        seconds = float(value)
    except ValueError:
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            seconds = (date - (now or datetime.now(timezone.utc))).total_seconds()
        except (ValueError, TypeError, OverflowError):
            return 0
    return max(0, seconds) if math.isfinite(seconds) else 0


@dataclass
class Link:
    url: str
    sources: list[str]
    status: str = "pending"
    message: str = ""
    accepted: bool = False
    attempts: int = 0
    history: list[dict] = field(default_factory=list)

    @property
    def retryable(self) -> bool:
        return self.status in {"network_error", "rate_limited"}


def interleave_links(links: list[Link]) -> list[Link]:
    """Spread hosts across a batch without maintaining a scheduler queue."""
    groups: dict[str, list[Link]] = {}
    for link in links:
        groups.setdefault(httpx.URL(link.url).host, []).append(link)
    return [link for group in zip_longest(*groups.values()) for link in group if link is not None]


class LinkChecker:
    def __init__(self, settings: Settings, client: httpx.AsyncClient):
        self.settings = settings
        self.client = client
        self.semaphore = asyncio.Semaphore(settings.global_concurrency)
        self.blocked_until: dict[str, float] = {}

    def finish(self, link: Link, status: str, message: str, accepted: bool = False) -> Link:
        link.status, link.message, link.accepted = status, message, accepted
        print(f"[{status}] {link.url}: {message}", flush=True)
        return link

    def prepare(self, urls: dict[str, list[str]]) -> list[Link]:
        links = []
        for url, sources in urls.items():
            link = Link(url, sources)
            links.append(link)
            try:
                reason = example_reason(url)
                if not reason:
                    reason = SKIP_DOMAINS.get(httpx.URL(url).host.rstrip("."), "")
                if not reason and is_contributor_profile(url, sources):
                    reason = "Historical contributor attribution; account renames/deletions are allowed."
                if reason:
                    self.finish(link, "skipped", reason, accepted=True)
                    continue
                parsed = httpx.URL(url)
                if parsed.scheme not in {"http", "https"} or not parsed.host or parsed.username:
                    raise ValueError("expected public HTTP(S) URL without credentials")
            except (httpx.InvalidURL, ValueError) as exc:
                self.finish(link, "broken", str(exc))
        return links

    def server_delay(self, code: int, headers: httpx.Headers) -> float:
        if code not in {403, 429, 503}:
            return 0
        delay = retry_after(headers.get("retry-after"))
        if headers.get("x-ratelimit-remaining") == "0":
            with suppress(ValueError, KeyError):
                reset = float(headers["x-ratelimit-reset"]) - time.time()
                if math.isfinite(reset):
                    delay = max(delay, reset)
        return delay

    async def request(self, url: str, method: str) -> tuple[int, httpx.Headers]:
        host = httpx.URL(url).host
        while True:
            delay = self.blocked_until.get(host, 0) - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)  # Never hold an HTTP slot while waiting.
            async with self.semaphore:
                if self.blocked_until.get(host, 0) > time.monotonic():
                    continue  # Another response set a delay while we waited for a slot.
                async with asyncio.timeout(self.settings.request_timeout):
                    async with self.client.stream(method, url, follow_redirects=False) as response:
                        code, headers = response.status_code, response.headers
                delay = self.server_delay(code, headers)
                if delay > 0:
                    self.blocked_until[host] = max(self.blocked_until.get(host, 0), time.monotonic() + delay)
                    print(f"[wait] {host}: server requested {delay:.1f}s", flush=True)
                return code, headers

    async def check_url(self, link: Link, *, method: str = "HEAD", phase: str = "initial") -> Link:
        """One logical check: follow redirects, confirm HEAD failures, classify."""
        link.attempts += 1
        target = link.url
        redirects = 0
        while True:
            event = {"url": target, "method": method, "attempt": link.attempts, "phase": phase}
            try:
                code, headers = await self.request(target, method)
                event["code"] = code
            except (httpx.TransportError, TimeoutError) as exc:
                event["error"] = f"{type(exc).__name__}: {exc}"
                link.history.append(event)
                return self.finish(link, "network_error", event["error"])
            link.history.append(event)
            if code in REDIRECT_CODES:
                if not headers.get("location") or redirects >= MAX_REDIRECTS:
                    return self.finish(link, "broken", "missing Location or redirect limit exceeded")
                try:
                    parsed = httpx.URL(urljoin(target, headers["location"]))
                    if (
                        parsed.scheme not in {"http", "https"}
                        or not parsed.host
                        or parsed.username
                        or example_reason(str(parsed))
                    ):
                        raise ValueError("redirect is not a public HTTP(S) target")
                except (httpx.InvalidURL, ValueError) as exc:
                    return self.finish(link, "broken", str(exc))
                reason = SKIP_DOMAINS.get(parsed.host.rstrip("."))
                if reason:
                    return self.finish(link, "skipped", f"redirect to skipped domain {parsed.host}: {reason}", True)
                target = str(parsed.copy_with(fragment=None))
                redirects += 1
                continue
            if 200 <= code < 300:
                return self.finish(link, "passed", f"HTTP {code}", accepted=True)
            if code == 429 or (
                code == 403 and ("retry-after" in headers or headers.get("x-ratelimit-remaining") == "0")
            ):
                return self.finish(link, "rate_limited", f"HTTP {code}")
            if method == "HEAD" and code in GET_FALLBACK_CODES:
                method = "GET"
                continue
            if code in TRANSIENT_CODES or 500 <= code < 600:
                return self.finish(link, "network_error", f"HTTP {code}")
            if code in {401, 403}:
                return self.finish(link, "restricted", f"HTTP {code}")
            return self.finish(link, "broken", f"HTTP {code}")

    async def check_batch(self, links: list[Link]) -> None:
        """Concurrent first pass; only temporary failures get another attempt."""

        async def check(link: Link):
            for _ in range(self.settings.initial_attempts):
                await self.check_url(link)
                # Rate limits go straight to late GET rechecks; request()
                # enforces explicit server waits without a default pause.
                if not link.retryable or link.status == "rate_limited":
                    break

        # A programming error propagates with its traceback and cancels peers.
        async with asyncio.TaskGroup() as tasks:
            for link in interleave_links(links):
                tasks.create_task(check(link))

    async def run(self, urls: dict[str, list[str]]) -> list[Link]:
        links = self.prepare(urls)
        try:
            async with asyncio.timeout(self.settings.total_timeout):
                await self.check_batch([link for link in links if link.status == "pending"])
                # Keep the proven late, serial GET sweep over temporary failures.
                for link in interleave_links([link for link in links if link.retryable]):
                    for attempt in range(self.settings.recheck_attempts):
                        if attempt:
                            await asyncio.sleep(self.settings.retry_delay)
                        await self.check_url(link, method="GET", phase="recheck")
                        if not link.retryable:
                            break
        except TimeoutError:
            for link in links:
                if link.status == "pending" or link.retryable:
                    status = "rate_limited" if link.status == "rate_limited" else "unverified"
                    self.finish(link, status, f"overall check deadline exceeded; last result: {link.message}")
        return links


def write_reports(directory: Path, links: list[Link], elapsed: float):
    directory.mkdir(parents=True, exist_ok=True)
    counts = Counter(link.status for link in links)
    report = {"elapsed": elapsed, "counts": dict(counts), "links": [asdict(link) for link in links]}
    (directory / "output.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = [
        "## Documentation link check",
        "",
        f"Checked {len(links)} unique URLs in {elapsed:.1f}s.",
        "",
        "| Result | URLs |",
        "| --- | ---: |",
    ]
    summary.extend(f"| {status} | {count} |" for status, count in sorted(counts.items()))
    summary.extend(["", "Unaccepted results fail the check.", ""])
    for link in links:
        if not link.accepted:
            summary.append(
                f"- {link.status}: <code>{escape(link.url)}</code> — <code>{escape(link.message)}</code> "
                f"(<code>{escape(', '.join(link.sources))}</code>)"
            )
    (directory / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="Markdown files or directories")
    parser.add_argument("--files-from", type=Path, help="JSON array of file paths (deleted files are ignored)")
    parser.add_argument("--report-dir", type=Path, default=Path("docs/_build/linkcheck"))
    parser.add_argument("--timeout", type=float, help="Overall check deadline in seconds")
    parser.add_argument("--extract-only", action="store_true", help="List parsed targets without network requests")
    args = parser.parse_args()
    if args.timeout is not None and not 0 < args.timeout < math.inf:
        parser.error("--timeout must be finite and positive")
    return args


async def main() -> int:
    args = parse_args()
    settings = Settings()
    paths = args.paths
    if args.files_from:
        values = json.loads(args.files_from.read_text(encoding="utf-8"))
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise ValueError("--files-from must contain a JSON array of strings")
        paths.extend(value for value in values if Path(value).is_file())
    if not paths and not args.files_from:
        raise ValueError("provide Markdown paths or --files-from")
    if args.timeout is not None:
        settings = replace(settings, total_timeout=args.timeout)
    urls = collect_links(paths, settings)
    if args.extract_only:
        print(json.dumps(urls, indent=2, ensure_ascii=False))
        return 0
    started = time.monotonic()
    limits = httpx.Limits(
        max_connections=settings.global_concurrency, max_keepalive_connections=settings.global_concurrency
    )
    timeout = httpx.Timeout(settings.request_timeout, connect=settings.connect_timeout)
    async with httpx.AsyncClient(limits=limits, timeout=timeout, headers={"User-Agent": USER_AGENT}) as client:
        checker = LinkChecker(settings, client)
        links = await checker.run(urls)
    write_reports(args.report_dir, links, time.monotonic() - started)
    return int(any(not link.accepted for link in links))


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
