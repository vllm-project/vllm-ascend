#!/usr/bin/env python3
"""Check rendered Markdown links with adaptive, per-host HTTP scheduling.

No vLLM or accelerator imports are needed. See docs/requirements-linkcheck.txt
and the contribution testing guide for standalone usage.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import heapq
import ipaddress
import json
import math
import time
import traceback
from collections import Counter
from contextlib import suppress
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html import escape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

import httpx
import markdown
import regex as re
import tomllib

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / ".github/workflows/configs/doc_linkcheck.toml"
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
MAX_REQUESTS = 24
RATE_REDUCTION = 0.5
GROWTH_STEP = 2
STABLE_RESPONSES = 32
RECHECK_RESERVE_FRACTION = 0.25
UNREACHABLE_EXAMPLE_IP = "192.0.0.1"
TEMPLATE_RE = re.compile(r"\$\{|\{\{|[<>]")
USER_AGENT = "Mozilla/5.0 (compatible; vllm-ascend-linkcheck; +https://github.com/vllm-project/vllm-ascend)"


@dataclass(frozen=True)
class HostPolicy:
    concurrency: int = 4
    max_concurrency: int = 8
    requests_per_second: float = 4
    max_requests_per_second: float = 8

    def __post_init__(self):
        for name in ("concurrency", "max_concurrency"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if not 1 <= self.concurrency <= self.max_concurrency:
            raise ValueError("require 1 <= concurrency <= max_concurrency")
        if not 0 < self.requests_per_second <= self.max_requests_per_second < math.inf:
            raise ValueError("require finite 0 < requests_per_second <= max_requests_per_second")


@dataclass(frozen=True)
class ExceptionRule:
    pattern: str
    reason: str
    action: str = "skip"
    files: str = ".*"

    def __post_init__(self):
        re.compile(self.pattern)
        re.compile(self.files)
        if not self.reason.strip() or self.action not in {"skip", "allow_restricted"}:
            raise ValueError("exceptions require a reason and skip/allow_restricted action")

    def matches(self, url: str, sources: list[str]) -> bool:
        # An exception scoped to one file must not hide a failure in another.
        return bool(re.search(self.pattern, url)) and all(re.search(self.files, source) for source in sources)


@dataclass(frozen=True)
class Settings:
    global_concurrency: int = 32
    max_rounds: int = 3
    request_timeout: float = 20
    connect_timeout: float = 5
    total_timeout: float = 1200
    retry_delay: float = 5
    cooldown: float = 60
    growth_window: float = 30
    excludes: tuple[str, ...] = ("docs/source/zh",)
    default_host: HostPolicy = field(default_factory=HostPolicy)
    hosts: dict[str, HostPolicy] = field(default_factory=dict)
    exceptions: tuple[ExceptionRule, ...] = ()

    def __post_init__(self):
        for name in ("global_concurrency", "max_rounds"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("request_timeout", "connect_timeout", "total_timeout", "retry_delay", "cooldown", "growth_window"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not 0 < value < math.inf:
                raise ValueError(f"{name} must be finite and positive")


def load_settings(path: Path) -> Settings:
    with path.open("rb") as stream:
        data = tomllib.load(stream)
    allowed = {"check", "default_host", "hosts", "exceptions"}
    if data.keys() - allowed:
        raise ValueError(f"unknown configuration sections: {data.keys() - allowed}")
    options = data.get("check", {})
    if options.keys() - {item.name for item in fields(Settings) if item.name not in allowed}:
        raise ValueError("unknown check setting")
    if "excludes" in options:
        options["excludes"] = tuple(options["excludes"])
    return Settings(
        **options,
        default_host=HostPolicy(**data.get("default_host", {})),
        hosts={host.lower(): HostPolicy(**policy) for host, policy in data.get("hosts", {}).items()},
        exceptions=tuple(ExceptionRule(**rule) for rule in data.get("exceptions", [])),
    )


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


def skip_reason(url: str) -> str:
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
    round: int = 1
    target: str = ""
    method: str = "HEAD"
    redirects: int = 0
    ready_at: float = 0
    history: list[dict] = field(default_factory=list)


@dataclass
class HostState:
    policy: HostPolicy
    concurrency: int = field(init=False)
    rate: float = field(init=False)
    active: int = 0
    next_start: float = 0
    blocked_until: float = 0
    cooldowns: int = 0
    probing: bool = False
    stable: int = 0
    window_started: float = field(default_factory=time.monotonic)
    requests: int = 0
    elapsed: float = 0
    changes: list[dict] = field(default_factory=list)

    def __post_init__(self):
        self.concurrency = self.policy.concurrency
        self.rate = self.policy.requests_per_second

    @property
    def limit(self) -> int:
        return 1 if self.probing else self.concurrency

    def limited(self, delay: float):
        now = time.monotonic()
        # Multiple in-flight 429s belong to one cooldown wave.
        if now >= self.blocked_until:
            self.cooldowns += 1
            self.concurrency = max(1, int(self.concurrency * RATE_REDUCTION))
            self.rate = max(min(1, self.policy.requests_per_second), self.rate * RATE_REDUCTION)
            self.stable = 0
            self.changes.append(
                {"event": "cooldown", "concurrency": self.concurrency, "rate": self.rate, "delay": delay}
            )
        self.blocked_until = max(self.blocked_until, now + delay)
        self.window_started = self.blocked_until
        self.probing = True

    def recovered(self):
        if time.monotonic() >= self.blocked_until:
            # Only consecutive unsuccessful recovery probes trip the circuit.
            self.cooldowns = 0
            self.probing = False

    def succeeded(self, settings: Settings):
        now = time.monotonic()
        if now < self.blocked_until:
            return
        self.stable += 1
        if self.stable >= STABLE_RESPONSES and now - self.window_started >= settings.growth_window:
            self.concurrency = min(self.policy.max_concurrency, self.concurrency + GROWTH_STEP)
            self.rate = min(self.policy.max_requests_per_second, self.rate + GROWTH_STEP)
            self.changes.append({"event": "growth", "concurrency": self.concurrency, "rate": self.rate})
            self.cooldowns = 0
            self.stable = 0
            self.window_started = now


class LinkChecker:
    def __init__(self, settings: Settings, client: httpx.AsyncClient):
        self.settings = settings
        self.client = client
        self.hosts: dict[str, HostState] = {}
        self.pending: dict[str, list[tuple[float, int, Link]]] = {}
        self.deferred: list[Link] = []
        self.sequence = 0

    def enqueue(self, link: Link, *, priority: bool = False):
        host = httpx.URL(link.target).host
        self.hosts.setdefault(host, HostState(self.settings.hosts.get(host, self.settings.default_host)))
        self.sequence += 1
        order = (0, -self.sequence) if priority else (link.ready_at, self.sequence)
        heapq.heappush(self.pending.setdefault(host, []), (*order, link))

    def finish(self, link: Link, status: str, message: str, accepted: bool = False):
        link.status, link.message, link.accepted = status, message, accepted
        print(f"[{status}] {link.url}: {message}", flush=True)

    def defer(self, link: Link, status: str, message: str, delay: float):
        link.status, link.message = status, message
        if link.round >= self.settings.max_rounds or len(link.history) >= MAX_REQUESTS:
            self.finish(link, status, message)
            return
        link.ready_at = time.monotonic() + delay
        self.deferred.append(link)
        print(f"[retry] {link.url}: {message}; queued for round {link.round + 1}", flush=True)

    async def request(self, link: Link) -> dict:
        started = time.monotonic()
        result = {"url": link.target, "method": link.method, "round": link.round}
        try:
            async with asyncio.timeout(self.settings.request_timeout):
                async with self.client.stream(link.method, link.target, follow_redirects=False) as response:
                    result.update(code=response.status_code, headers=dict(response.headers))
        except (httpx.TransportError, TimeoutError) as exc:
            result.update(error="network_error", message=f"{type(exc).__name__}: {exc}")
        except Exception:
            result.update(error="checker_error", message=traceback.format_exc())
        result["elapsed"] = time.monotonic() - started
        return result

    def complete(self, link: Link, result: dict):
        link.history.append(result)
        if result.get("error") == "checker_error":
            self.finish(link, "checker_error", result["message"])
            return
        if result.get("error"):
            self.defer(link, "network_error", result["message"], self.settings.retry_delay * 2 ** (link.round - 1))
            return
        code, headers = result["code"], result["headers"]
        host = self.hosts[httpx.URL(link.target).host]
        limited = code == 429 or (
            code == 403 and ("retry-after" in headers or headers.get("x-ratelimit-remaining") == "0")
        )
        if not limited and 200 <= code < 500:
            host.recovered()
        if limited:
            wave = host.cooldowns if time.monotonic() >= host.blocked_until else max(0, host.cooldowns - 1)
            delay = max(self.settings.cooldown * 2**wave, retry_after(headers.get("retry-after")))
            if headers.get("x-ratelimit-remaining") == "0":
                with suppress(ValueError, KeyError):
                    reset = float(headers["x-ratelimit-reset"]) - time.time()
                    if math.isfinite(reset):
                        delay = max(delay, reset)
            previous = host.cooldowns
            host.limited(delay)
            if host.cooldowns != previous:
                print(
                    f"[cooldown] {httpx.URL(link.target).host}: {delay:.1f}s; "
                    f"concurrency={host.concurrency}, requests/s={host.rate:g}",
                    flush=True,
                )
            self.defer(link, "rate_limited", f"HTTP {code}; host cooling down for {delay:.1f}s", delay)
        elif code in REDIRECT_CODES:
            self.redirect(link, headers.get("location"))
        elif 200 <= code < 300:
            host.succeeded(self.settings)
            self.finish(link, "passed", f"HTTP {code}", accepted=True)
        elif link.method == "HEAD" and code in GET_FALLBACK_CODES:
            link.method = "GET"
            link.ready_at = 0
            self.enqueue(link)
        elif code in TRANSIENT_CODES or 500 <= code < 600:
            self.defer(
                link,
                "network_error",
                f"HTTP {code}",
                max(self.settings.retry_delay * 2 ** (link.round - 1), retry_after(headers.get("retry-after"))),
            )
        elif code in {401, 403}:
            rule = next(
                (
                    rule
                    for rule in self.settings.exceptions
                    if rule.action == "allow_restricted" and rule.matches(link.url, link.sources)
                ),
                None,
            )
            self.finish(
                link, "restricted", f"HTTP {code}" + (f"; {rule.reason}" if rule else ""), accepted=rule is not None
            )
        else:
            self.finish(link, "broken", f"HTTP {code}")

    def redirect(self, link: Link, location: str | None):
        if not location or link.redirects >= MAX_REDIRECTS:
            self.finish(link, "broken", "missing Location or redirect limit exceeded")
            return
        target = urljoin(link.target, location)
        try:
            parsed = httpx.URL(target)
            if parsed.scheme not in {"http", "https"} or not parsed.host or parsed.username or skip_reason(target):
                raise ValueError("redirect is not a public HTTP(S) target")
        except (httpx.InvalidURL, ValueError) as exc:
            self.finish(link, "broken", str(exc))
            return
        link.target = str(parsed.copy_with(fragment=None))
        link.redirects += 1
        link.ready_at = 0
        self.enqueue(link)

    def prepare(self, urls: dict[str, list[str]]) -> list[Link]:
        links = []
        for url, sources in urls.items():
            link = Link(url, sources, target=url)
            links.append(link)
            try:
                reason = skip_reason(url)
                rule = next(
                    (rule for rule in self.settings.exceptions if rule.action == "skip" and rule.matches(url, sources)),
                    None,
                )
                if reason or rule:
                    self.finish(link, "skipped", reason or rule.reason, accepted=True)
                    continue
                parsed = httpx.URL(url)
                if parsed.scheme not in {"http", "https"} or not parsed.host or parsed.username:
                    raise ValueError("expected public HTTP(S) URL without credentials")
                self.enqueue(link)
            except (httpx.InvalidURL, ValueError) as exc:
                self.finish(link, "broken", str(exc))
        return links

    def dispatch(self, active: dict[asyncio.Task, tuple[str, Link]]) -> float:
        now = time.monotonic()
        # One request per host on each sweep prevents a large host starving others.
        while len(active) < self.settings.global_concurrency:
            launched = False
            for name, queue in self.pending.items():
                host = self.hosts[name]
                if host.cooldowns >= self.settings.max_rounds:
                    while queue:
                        _, _, link = heapq.heappop(queue)
                        self.finish(link, "rate_limited", "host did not recover within the cooldown budget")
                if not queue or host.active >= host.limit:
                    continue
                due = max(queue[0][0], host.next_start, host.blocked_until)
                if due > now:
                    continue
                _, _, link = heapq.heappop(queue)
                if len(link.history) >= MAX_REQUESTS:
                    self.finish(link, "unverified", "HTTP request budget exceeded")
                    continue
                host.active += 1
                host.requests += 1
                host.next_start = now + 1 / host.rate
                active[asyncio.create_task(self.request(link))] = (name, link)
                launched = True
                if len(active) >= self.settings.global_concurrency:
                    break
            if not launched:
                break
        ready = [
            max(queue[0][0], self.hosts[name].next_start, self.hosts[name].blocked_until)
            for name, queue in self.pending.items()
            if queue and self.hosts[name].active < self.hosts[name].limit
        ]
        return min(ready, default=math.inf)

    async def run(self, urls: dict[str, list[str]]) -> list[Link]:
        links = self.prepare(urls)
        deadline = time.monotonic() + self.settings.total_timeout
        recheck_at = deadline - self.settings.total_timeout * RECHECK_RESERVE_FRACTION
        active: dict[asyncio.Task, tuple[str, Link]] = {}
        try:
            while any(self.pending.values()) or active or self.deferred:
                if time.monotonic() >= deadline:
                    break
                idle = not any(self.pending.values()) and not active
                if self.deferred and (idle or time.monotonic() >= recheck_at):
                    # Late rechecks usually follow the first pass. In a long
                    # scan, reserve time so they cannot starve behind new URLs.
                    deferred, self.deferred = self.deferred, []
                    for link in deferred:
                        if not idle and link.ready_at > time.monotonic():
                            self.deferred.append(link)
                            continue
                        link.round += 1
                        link.target, link.method, link.redirects = link.url, "HEAD", 0
                        self.enqueue(link, priority=not idle)
                ready_at = self.dispatch(active)
                if self.deferred:
                    ready_at = min(ready_at, max(recheck_at, min(link.ready_at for link in self.deferred)))
                remaining = deadline - time.monotonic()
                delay = max(0, min(remaining, ready_at - time.monotonic()))
                if active:
                    # With all slots occupied, wait for completion rather than
                    # spinning on an eligible queue that cannot be dispatched.
                    if len(active) >= self.settings.global_concurrency:
                        delay = max(0, remaining)
                    done, _ = await asyncio.wait(active, timeout=delay, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        name, link = active.pop(task)
                        self.hosts[name].active -= 1
                        result = task.result()
                        self.hosts[name].elapsed += result["elapsed"]
                        self.complete(link, result)
                elif any(self.pending.values()):
                    await asyncio.sleep(delay)
        finally:
            for task in active:
                task.cancel()
            await asyncio.gather(*active, return_exceptions=True)
        unfinished = {id(link) for queue in self.pending.values() for _, _, link in queue}
        unfinished.update(id(link) for _, link in active.values())
        unfinished.update(id(link) for link in self.deferred)
        for link in links:
            if id(link) in unfinished:
                host = self.hosts[httpx.URL(link.target).host]
                status = "rate_limited" if host.blocked_until > time.monotonic() else "unverified"
                self.finish(link, status, "overall check deadline exceeded")
        return links


def write_reports(directory: Path, links: list[Link], checker: LinkChecker, elapsed: float):
    directory.mkdir(parents=True, exist_ok=True)
    counts = Counter(link.status for link in links)
    hosts = {
        name: {
            "requests": host.requests,
            "mean_latency": host.elapsed / max(1, host.requests),
            "concurrency": host.concurrency,
            "requests_per_second": host.rate,
            "changes": host.changes,
        }
        for name, host in sorted(checker.hosts.items())
    }
    report = {"elapsed": elapsed, "counts": dict(counts), "hosts": hosts, "links": [asdict(link) for link in links]}
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
    summary.extend(["", "Unaccepted results fail the check. Restricted exceptions are explicitly configured.", ""])
    for link in links:
        if not link.accepted:
            summary.append(
                f"- {link.status}: <code>{escape(link.url)}</code> (<code>{escape(', '.join(link.sources))}</code>)"
            )
    (directory / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="Markdown files or directories")
    parser.add_argument("--files-from", type=Path, help="JSON array of file paths (deleted files are ignored)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report-dir", type=Path, default=Path("docs/_build/linkcheck"))
    parser.add_argument("--timeout", type=float, help="Overall check deadline in seconds")
    parser.add_argument("--extract-only", action="store_true", help="List parsed targets without network requests")
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    settings = load_settings(args.config)
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
    write_reports(args.report_dir, links, checker, time.monotonic() - started)
    return int(any(not link.accepted for link in links))


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except (ValueError, OSError, TypeError) as error:
        raise SystemExit(f"Link check configuration/input error: {error}") from error
