"""CPU-only contract tests for the standalone documentation link checker."""

import asyncio
import importlib.util
import json
import sys
import threading
import time
from collections import Counter
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from textwrap import dedent

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def checker_module():
    name = "_standalone_doc_linkcheck"
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / ".github/workflows/scripts/check_md_links.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(name)


@pytest.fixture
def settings(checker_module):
    return checker_module.Settings(
        global_concurrency=4,
        default_host=checker_module.HostPolicy(2, 4, 1000, 2000),
        cooldown=0.02,
        retry_delay=0.002,
        growth_window=0.01,
        total_timeout=2,
    )


def run_check(module, settings, urls, handler=None, transport=None):
    async def run():
        async with httpx.AsyncClient(transport=transport or httpx.MockTransport(handler)) as client:
            checker = module.LinkChecker(settings, client)
            results = await checker.run({url: ["docs/source/test.md"] for url in urls})
            return results, checker

    return asyncio.run(run())


def test_extract_markdown_semantics(checker_module):
    content = dedent("""
        [inline](https://EXAMPLE.com/a#one)
        [reference][target]
        [target]: https://example.com/a#two
        ![image](https://images.example/a.png)
        <https://example.com/auto>
        <a href="https://example.com/raw?q=1&amp;x=2">raw</a>
        <!-- <a href="https://example.com/comment">comment</a> -->
        `https://example.com/inline-code`

        === "Serving"

            [visible](https://example.com/tab)

            ```bash
            curl http://<SERVICE_IP>:8000
            # https://example.com/hidden-code
            ```

        !!! note

            [nested](https://example.com/note)

            ~~~~text
            [hidden](https://example.com/hidden)
            ~~~~
    """)
    assert checker_module.extract_urls(content) == {
        "https://example.com/a",
        "https://images.example/a.png",
        "https://example.com/auto",
        "https://example.com/raw?q=1&x=2",
        "https://example.com/tab",
        "https://example.com/note",
    }


def test_collect_preserves_all_sources_and_excludes_translation(checker_module, settings, tmp_path):
    first = tmp_path / "a.md"
    second = tmp_path / "b with spaces.md"
    translated = tmp_path / "zh"
    translated.mkdir()
    first.write_text("[a](https://example.com/a#first)")
    second.write_text("[a](https://example.com/a#second)")
    (translated / "c.md").write_text("[c](https://example.com/zh)")
    links = checker_module.collect_links([str(tmp_path)], replace(settings, excludes=("zh",)))
    assert links == {"https://example.com/a": [str(first), str(second)]}
    with pytest.raises(ValueError, match="does not exist"):
        checker_module.collect_links([str(tmp_path / "missing.md")], settings)


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8000/v1",
        "http://127.0.0.1:8000",
        "http://0.0.0.0:8000",
        "http://192.0.0.1:8000",
        "http://[::1]:8000",
        "http://<SERVICE_IP>:8000",
        "https://files.example/${PYTHON_TAG}.whl",
        "https://example.com/{{version}}/",
    ],
)
def test_examples_never_make_requests(checker_module, settings, url):
    def handler(request):
        pytest.fail(f"unexpected request: {request.url}")

    results, _ = run_check(checker_module, settings, [url], handler)
    assert results[0].status == "skipped" and results[0].accepted


def test_localhost_substrings_do_not_skip_public_targets(checker_module):
    assert not checker_module.skip_reason("https://example.com/docs/localhost")
    assert not checker_module.skip_reason("https://localhost.example.com/a")


def test_head_fallback_and_real_broken_links(checker_module, settings):
    calls = []

    def handler(request):
        calls.append((request.method, request.url.path))
        if request.url.path == "/head-unsupported" and request.method == "HEAD":
            return httpx.Response(405)
        if request.url.path == "/head-404" and request.method == "HEAD":
            return httpx.Response(404)
        if request.url.path == "/gone":
            return httpx.Response(404)
        return httpx.Response(200)

    results, _ = run_check(
        checker_module,
        settings,
        [f"https://a.example/{path}" for path in ("head-unsupported", "head-404", "gone")],
        handler,
    )
    assert {result.url.rsplit("/", 1)[1]: result.status for result in results} == {
        "head-unsupported": "passed",
        "head-404": "passed",
        "gone": "broken",
    }
    assert calls.count(("HEAD", "/gone")) == calls.count(("GET", "/gone")) == 1


def test_redirect_uses_destination_policy_and_detects_loops(checker_module, settings):
    calls = []

    def handler(request):
        calls.append(str(request.url))
        if request.url.host == "a.example":
            return httpx.Response(302, headers={"location": "https://b.example/final"})
        if request.url.path == "/loop":
            return httpx.Response(301, headers={"location": "/loop"})
        return httpx.Response(200)

    results, checker = run_check(
        checker_module, settings, ["https://a.example/start", "https://b.example/loop"], handler
    )
    assert results[0].status == "passed" and results[0].history[-1]["url"] == "https://b.example/final"
    assert results[1].status == "broken"
    assert checker.hosts["b.example"].requests == len(calls) - 1


def test_failures_rechecked_after_first_pass(checker_module, settings):
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(503 if request.url.path == "/flaky" and calls.count("/flaky") == 1 else 200)

    results, _ = run_check(checker_module, settings, ["https://a.example/flaky", "https://a.example/healthy"], handler)
    assert calls == ["/flaky", "/healthy", "/flaky"]
    assert all(result.accepted for result in results)
    assert [event["round"] for event in results[0].history] == [1, 2]


def test_rate_limit_cools_whole_host_without_blocking_other_hosts(checker_module, settings):
    calls = []

    def handler(request):
        calls.append((request.url.host, request.url.path, time.monotonic()))
        if request.url.path == "/limited" and sum(path == "/limited" for _, path, _ in calls) == 1:
            return httpx.Response(429, headers={"retry-after": "0"})
        return httpx.Response(200)

    results, checker = run_check(
        checker_module,
        settings,
        [
            "https://a.example/limited",
            "https://a.example/other",
            "https://b.example/healthy",
        ],
        handler,
    )
    first_a = next(stamp for host, _, stamp in calls if host == "a.example")
    second_a = [stamp for host, _, stamp in calls if host == "a.example"][1]
    b_time = next(stamp for host, _, stamp in calls if host == "b.example")
    assert second_a - first_a >= settings.cooldown * 0.9
    assert b_time < second_a
    assert all(result.accepted for result in results)
    assert checker.hosts["a.example"].concurrency == 1


def test_unrecoverable_host_stops_bulk_requests(checker_module, settings):
    calls = Counter()

    def handler(request):
        calls[request.url.host] += 1
        return httpx.Response(429 if request.url.host == "a.example" else 200)

    results, _ = run_check(
        checker_module,
        settings,
        [*[f"https://a.example/{index}" for index in range(100)], "https://b.example/ok"],
        handler,
    )
    assert calls["a.example"] < 10
    assert calls["b.example"] == 1
    assert all(result.status == "rate_limited" for result in results if "a.example" in result.url)


def test_retry_after_and_deadline_are_not_shortened(checker_module, settings):
    results, _ = run_check(
        checker_module,
        replace(settings, total_timeout=0.025),
        ["https://a.example/a"],
        lambda request: httpx.Response(429, headers={"retry-after": "3600"}),
    )
    assert results[0].status == "rate_limited" and not results[0].accepted
    assert len(results[0].history) == 1
    now = datetime(2026, 10, 29, tzinfo=timezone.utc)
    assert checker_module.retry_after(format_datetime(now + timedelta(seconds=90)), now) == 90
    for value in (None, "garbage", "NaN", "inf", "-10"):
        assert checker_module.retry_after(value, now) == 0


def test_inflight_requests_cancel_at_deadline(checker_module, settings):
    cancelled = []

    async def handler(request):
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.append(str(request.url))

    results, _ = run_check(checker_module, replace(settings, total_timeout=0.02), ["https://a.example/a"], handler)
    assert cancelled == ["https://a.example/a"]
    assert results[0].status == "unverified"


def test_network_errors_exhaust_round_budget(checker_module, settings):
    def handler(request):
        raise httpx.ReadTimeout("read timed out", request=request)

    results, _ = run_check(checker_module, settings, ["https://a.example/timeout"], handler)
    assert results[0].status == "network_error"
    assert len(results[0].history) == settings.max_rounds


def test_unexpected_error_is_reported_without_losing_other_results(checker_module, settings):
    def handler(request):
        if request.url.path == "/crash":
            raise AssertionError("bad host")
        return httpx.Response(200)

    results, _ = run_check(checker_module, settings, ["https://a.example/crash", "https://a.example/good"], handler)
    assert [result.status for result in results] == ["checker_error", "passed"]
    assert "AssertionError" in results[0].message


def test_concurrency_and_request_spacing(checker_module, settings):
    running = Counter()
    peak = Counter()
    starts = []

    async def handler(request):
        host = request.url.host
        starts.append((host, time.monotonic()))
        running[host] += 1
        running["total"] += 1
        peak[host] = max(peak[host], running[host])
        peak["total"] = max(peak["total"], running["total"])
        await asyncio.sleep(0.015)
        running[host] -= 1
        running["total"] -= 1
        return httpx.Response(200)

    policy = checker_module.HostPolicy(3, 3, 500, 500)
    settings = replace(settings, global_concurrency=5, default_host=policy)
    results, _ = run_check(
        checker_module,
        settings,
        [f"https://{host}.example/{index}" for host in ("a", "b") for index in range(10)],
        handler,
    )
    assert all(result.accepted for result in results)
    assert peak["a.example"] == peak["b.example"] == 3
    assert peak["total"] == 5
    for host in ("a.example", "b.example"):
        timestamps = [stamp for name, stamp in starts if name == host]
        assert all(right - left >= 0.0015 for left, right in zip(timestamps, timestamps[1:]))


def test_adaptive_growth_and_single_wave_reduction(checker_module, settings):
    host = checker_module.HostState(checker_module.HostPolicy(12, 24, 8, 16))
    host.window_started -= 100
    for _ in range(checker_module.STABLE_RESPONSES):
        host.succeeded(settings)
    assert (host.concurrency, host.rate) == (14, 10)
    host.limited(10)
    host.limited(10)
    assert (host.concurrency, host.rate, host.cooldowns) == (7, 5, 1)


def test_restricted_exceptions_are_scoped_and_visible(checker_module, settings):
    rule = checker_module.ExceptionRule(r"^https://a\.example/", "CI anti-bot restriction", "allow_restricted")
    settings = replace(settings, exceptions=(rule,))
    results, _ = run_check(
        checker_module,
        settings,
        ["https://a.example/help", "https://b.example/help"],
        lambda request: httpx.Response(403),
    )
    assert [result.status for result in results] == ["restricted", "restricted"]
    assert [result.accepted for result in results] == [True, False]
    assert not checker_module.ExceptionRule(".*", "attribution", files="contributors.md$").matches(
        "https://a.example/",
        ["contributors.md", "guide.md"],
    )


def test_config_validation_and_repository_defaults(checker_module, tmp_path):
    settings = checker_module.load_settings(REPO_ROOT / ".github/workflows/configs/doc_linkcheck.toml")
    assert settings.hosts["github.com"].concurrency == 12
    assert settings.hosts["github.com"].max_concurrency == 24
    bad = tmp_path / "bad.toml"
    bad.write_text("[check]\nglobal_concurrency = 0\n")
    with pytest.raises(ValueError):
        checker_module.load_settings(bad)
    bad.write_text("[check]\nmisspelled = 2\n")
    with pytest.raises(ValueError):
        checker_module.load_settings(bad)


def test_reports_include_failures_sources_and_host_metrics(checker_module, settings, tmp_path):
    results, checker = run_check(
        checker_module,
        settings,
        ["https://a.example/a", "https://a.example/gone"],
        lambda request: httpx.Response(404 if request.url.path == "/gone" else 200),
    )
    checker_module.write_reports(tmp_path, results, checker, 0.5)
    report = json.loads((tmp_path / "output.json").read_text())
    assert report["counts"] == {"passed": 1, "broken": 1}
    assert report["hosts"]["a.example"]["requests"] == 3
    assert report["links"][1]["sources"] == ["docs/source/test.md"]
    summary = (tmp_path / "summary.md").read_text()
    assert "https://a.example/gone" in summary and "broken" in summary


def test_real_http_head_get_redirect_and_retry(checker_module, settings, monkeypatch):
    calls = Counter()

    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self):
            self.respond()

        def do_GET(self):
            self.respond()

        def respond(self):
            calls[(self.command, self.path)] += 1
            code = 200
            headers = {}
            if self.path == "/redirect":
                code, headers = 302, {"Location": "/head"}
            elif self.path == "/head" and self.command == "HEAD":
                code = 405
            elif self.path == "/flaky" and calls[(self.command, self.path)] == 1:
                code = 503
            self.send_response(code)
            for name, value in headers.items():
                self.send_header(name, value)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    # Local endpoints are intentional only in this isolated transport test.
    monkeypatch.setattr(checker_module, "skip_reason", lambda url: "")
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        results, _ = run_check(
            checker_module, settings, [f"{base}/redirect", f"{base}/flaky"], transport=httpx.AsyncHTTPTransport()
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert all(result.accepted for result in results)
    assert calls[("HEAD", "/flaky")] == 2
    assert calls[("GET", "/head")] == 1


def test_malformed_target_does_not_abort_healthy_links(checker_module, settings):
    results, _ = run_check(
        checker_module, settings, ["http://[invalid", "https://a.example/good"], lambda request: httpx.Response(200)
    )
    assert [result.status for result in results] == ["broken", "passed"]


def test_explicit_403_rate_limit_is_not_accepted_as_restricted(checker_module, settings):
    rule = checker_module.ExceptionRule(".*", "known restriction", "allow_restricted")
    results, _ = run_check(
        checker_module,
        replace(settings, total_timeout=0.025, exceptions=(rule,)),
        ["https://a.example/limit"],
        lambda request: httpx.Response(403, headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": "9999999999"}),
    )
    assert results[0].status == "rate_limited" and not results[0].accepted
    assert len(results[0].history) == 1


@pytest.mark.parametrize("location", [None, "file:///etc/passwd", "http://127.0.0.1:8000/"])
def test_invalid_redirects_fail(checker_module, settings, location):
    results, _ = run_check(
        checker_module,
        settings,
        ["https://a.example/redirect"],
        lambda request: httpx.Response(302, headers={"location": location} if location else {}),
    )
    assert results[0].status == "broken"
    assert len(results[0].history) == 1


def test_request_budget_is_unverified_not_a_false_broken_link(checker_module, settings, monkeypatch):
    monkeypatch.setattr(checker_module, "MAX_REQUESTS", 1)
    results, _ = run_check(checker_module, settings, ["https://a.example/head"], lambda request: httpx.Response(405))
    assert results[0].status == "unverified"
    assert len(results[0].history) == 1


def test_streaming_get_closes_without_downloading_body(checker_module, settings):
    closed = []

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            pytest.fail("checker must not download response bodies")
            yield b"unreachable"

        async def aclose(self):
            closed.append(True)

    def handler(request):
        return httpx.Response(405 if request.method == "HEAD" else 200, stream=Body())

    results, _ = run_check(checker_module, settings, ["https://a.example/download"], handler)
    assert results[0].accepted
    assert len(closed) == 2


def test_deleted_files_input_and_empty_report(checker_module, tmp_path, monkeypatch):
    paths = tmp_path / "files.json"
    paths.write_text(json.dumps([str(tmp_path / "deleted.md")]))
    report = tmp_path / "report"
    monkeypatch.setattr(sys, "argv", ["check_md_links.py", "--files-from", str(paths), "--report-dir", str(report)])
    assert asyncio.run(checker_module.main()) == 0
    assert json.loads((report / "output.json").read_text())["links"] == []


def test_large_host_uses_and_grows_high_concurrency(checker_module, settings):
    active = 0
    peak = 0

    async def handler(request):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.04)
        active -= 1
        return httpx.Response(200)

    policy = checker_module.HostPolicy(12, 24, 2000, 2000)
    settings = replace(settings, global_concurrency=32, default_host=policy, growth_window=0.005, total_timeout=5)
    results, checker = run_check(
        checker_module, settings, [f"https://a.example/{index}" for index in range(400)], handler
    )
    assert all(result.accepted for result in results)
    assert peak == 24
    assert checker.hosts["a.example"].concurrency == 24


def test_unknown_server_error_is_not_mislabeled_as_broken(checker_module, settings):
    results, _ = run_check(checker_module, settings, ["https://a.example/help"], lambda request: httpx.Response(567))
    assert results[0].status == "network_error"
    assert len(results[0].history) == settings.max_rounds


@pytest.mark.parametrize("status", [401, 403, 404, 405, 501])
def test_head_method_failure_confirmed_with_get(checker_module, settings, status):
    results, _ = run_check(
        checker_module,
        settings,
        ["https://a.example/help"],
        lambda request: httpx.Response(status if request.method == "HEAD" else 200),
    )
    assert results[0].accepted
    assert [event["method"] for event in results[0].history] == ["HEAD", "GET"]


@pytest.mark.parametrize("reset", ["inf", "NaN", "garbage"])
def test_invalid_rate_reset_cannot_poison_reports(checker_module, settings, tmp_path, reset):
    results, checker = run_check(
        checker_module,
        replace(settings, max_rounds=1),
        ["https://a.example/limit"],
        lambda request: httpx.Response(403, headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": reset}),
    )
    checker_module.write_reports(tmp_path, results, checker, 0)
    json.dumps(json.loads((tmp_path / "output.json").read_text()), allow_nan=False)


def test_intermittent_limits_do_not_abort_healthy_host(checker_module, settings):
    calls = Counter()

    def handler(request):
        calls[request.url.path] += 1
        limited = request.url.path in {"/0", "/10", "/20"} and calls[request.url.path] == 1
        return httpx.Response(429 if limited else 200)

    urls = [f"https://a.example/{index}" for index in range(30)]
    results, checker = run_check(checker_module, settings, urls, handler)
    assert all(result.accepted for result in results)
    assert sum(event["event"] == "cooldown" for event in checker.hosts["a.example"].changes) == 3


def test_recovery_probe_is_single_and_restores_reduced_budget(checker_module, settings):
    host = checker_module.HostState(checker_module.HostPolicy(12, 24, 8, 16))
    host.limited(10)
    assert host.concurrency == 6 and host.limit == 1
    host.recovered()
    assert host.limit == 1  # In-flight responses during cooldown cannot reopen it.
    host.blocked_until = time.monotonic() - 1
    host.recovered()
    assert host.limit == 6 and host.cooldowns == 0


def test_long_first_pass_reserves_time_for_failed_link_recheck(checker_module, settings):
    calls = Counter()

    async def handler(request):
        calls[request.url.path] += 1
        await asyncio.sleep(0.01)
        return httpx.Response(503 if request.url.path == "/flaky" and calls["/flaky"] == 1 else 200)

    settings = replace(settings, global_concurrency=1, total_timeout=0.3)
    results, _ = run_check(
        checker_module,
        settings,
        ["https://a.example/flaky", *[f"https://a.example/{index}" for index in range(50)]],
        handler,
    )
    assert results[0].accepted and calls["/flaky"] == 2
    assert any(result.status == "unverified" for result in results)
