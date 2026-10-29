"""CPU-only contract tests for the standalone documentation link checker."""

import asyncio
import importlib.util
import json
import sys
import threading
import time
from builtins import ExceptionGroup
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
HUAWEI_HELP_URL = "https://support.huaweicloud.com/intl/en-us/usermanual-cce/cce_10_0239.html"


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
        retry_delay=0.002,
        total_timeout=2,
    )


def run_check(module, settings, urls, handler=None, transport=None, sources=("docs/source/test.md",)):
    async def run():
        async with httpx.AsyncClient(transport=transport or httpx.MockTransport(handler)) as client:
            checker = module.LinkChecker(settings, client)
            results = await checker.run({url: list(sources) for url in urls})
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
    assert not checker_module.example_reason("https://example.com/docs/localhost")
    assert not checker_module.example_reason("https://localhost.example.com/a")


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


def test_redirect_records_destination_and_detects_loops(checker_module, settings):
    calls = []

    def handler(request):
        calls.append(str(request.url))
        if request.url.host == "a.example":
            return httpx.Response(302, headers={"location": "https://b.example/final"})
        if request.url.path == "/loop":
            return httpx.Response(301, headers={"location": "/loop"})
        return httpx.Response(200)

    results, _ = run_check(checker_module, settings, ["https://a.example/start", "https://b.example/loop"], handler)
    assert results[0].status == "passed" and results[0].history[-1]["url"] == "https://b.example/final"
    assert results[1].status == "broken"


def test_failures_rechecked_after_first_pass(checker_module, settings):
    calls = []

    async def handler(request):
        if request.url.path == "/healthy":
            await asyncio.sleep(0.01)
        calls.append((request.method, request.url.path))
        return httpx.Response(503 if request.url.path == "/flaky" and request.method == "HEAD" else 200)

    results, _ = run_check(checker_module, settings, ["https://a.example/flaky", "https://a.example/healthy"], handler)
    assert calls == [("HEAD", "/flaky")] * 3 + [("HEAD", "/healthy"), ("GET", "/flaky")]
    assert all(result.accepted for result in results)
    assert [event["attempt"] for event in results[0].history] == [1, 2, 3, 4]
    assert [event["phase"] for event in results[0].history] == ["initial"] * 3 + ["recheck"]


def test_explicit_server_wait_frees_slots_for_other_hosts(checker_module, settings):
    calls = []

    def handler(request):
        calls.append((request.url.host, request.url.path, time.monotonic()))
        if request.url.path == "/limited" and sum(path == "/limited" for _, path, _ in calls) == 1:
            return httpx.Response(429, headers={"retry-after": "0.04"})
        return httpx.Response(200)

    results, _ = run_check(
        checker_module,
        replace(settings, global_concurrency=1),
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
    assert second_a - first_a >= 0.035
    assert b_time < second_a
    assert all(result.accepted for result in results)


def test_limits_without_headers_defer_without_default_host_wait(checker_module, settings):
    calls = Counter()

    def handler(request):
        calls[request.url.host] += 1
        return httpx.Response(429 if request.url.host == "a.example" else 200)

    results, checker = run_check(
        checker_module,
        settings,
        [*[f"https://a.example/{index}" for index in range(3)], "https://b.example/ok"],
        handler,
    )
    assert calls["a.example"] == 3 * (1 + settings.recheck_attempts)
    assert calls["b.example"] == 1
    assert all(result.status == "rate_limited" for result in results if "a.example" in result.url)
    assert all(result.attempts == 1 + settings.recheck_attempts for result in results if "a.example" in result.url)
    assert checker.blocked_until == {}
    assert all([event["method"] for event in result.history] == ["HEAD"] + ["GET"] * 4 for result in results[:3])


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


def test_network_errors_exhaust_two_stage_attempt_budget(checker_module, settings):
    def handler(request):
        raise httpx.ReadTimeout("read timed out", request=request)

    results, _ = run_check(checker_module, settings, ["https://a.example/timeout"], handler)
    assert results[0].status == "network_error"
    assert results[0].attempts == len(results[0].history) == settings.initial_attempts + settings.recheck_attempts
    assert [event["method"] for event in results[0].history] == ["HEAD"] * 3 + ["GET"] * 4


def test_programming_error_propagates_and_cancels_other_requests(checker_module, settings):
    started = asyncio.Event()
    cancelled = []

    async def handler(request):
        if request.url.path == "/crash":
            await started.wait()
            raise AssertionError("bad host")
        started.set()
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.append(str(request.url))

    with pytest.raises(ExceptionGroup) as error:
        run_check(checker_module, settings, ["https://a.example/crash", "https://a.example/slow"], handler)
    assert len(error.value.exceptions) == 1
    assert isinstance(error.value.exceptions[0], AssertionError)
    assert str(error.value.exceptions[0]) == "bad host"
    assert cancelled == ["https://a.example/slow"]


def test_programming_error_in_recheck_propagates(checker_module, settings):
    def handler(request):
        if request.method == "GET":
            raise TypeError("broken recheck")
        return httpx.Response(503)

    with pytest.raises(TypeError, match="broken recheck"):
        run_check(checker_module, settings, ["https://a.example/crash"], handler)


def test_global_concurrency_spreads_hosts_without_extra_host_limits(checker_module, settings):
    running = Counter()
    peak = Counter()

    async def handler(request):
        host = request.url.host
        running[host] += 1
        running["total"] += 1
        peak[host] = max(peak[host], running[host])
        peak["total"] = max(peak["total"], running["total"])
        await asyncio.sleep(0.015)
        running[host] -= 1
        running["total"] -= 1
        return httpx.Response(200)

    settings = replace(settings, global_concurrency=5)
    results, _ = run_check(
        checker_module,
        settings,
        [f"https://{host}.example/{index}" for host in ("a", "b") for index in range(10)],
        handler,
    )
    assert all(result.accepted for result in results)
    assert peak["a.example"] == peak["b.example"] == 3
    assert peak["total"] == 5


def test_host_order_is_interleaved_without_queues(checker_module):
    links = [checker_module.Link(f"https://{host}.example/{index}", []) for host in ("a", "b") for index in range(3)]
    assert [link.url for link in checker_module.interleave_links(links)] == [
        f"https://{host}.example/{index}" for index in range(3) for host in ("a", "b")
    ]
    assert checker_module.interleave_links([]) == []


@pytest.mark.parametrize("status", [401, 403, 404])
def test_huawei_help_url_has_no_restricted_access_exception(checker_module, settings, status):
    results, _ = run_check(
        checker_module,
        settings,
        [HUAWEI_HELP_URL],
        lambda request: httpx.Response(status),
    )
    assert results[0].status == ("broken" if status == 404 else "restricted")
    assert not results[0].accepted
    assert results[0].attempts == 1
    assert [event["method"] for event in results[0].history] == ["HEAD", "GET"]


@pytest.mark.parametrize(
    "url,sources,skipped",
    [
        ("https://github.com/alice", ["docs/source/community/contributors.md"], True),
        ("https://github.com/alice/", ["/repo/docs/source/community/contributors.md"], True),
        ("https://github.com/alice", ["docs/source/community/contributors.md", "guide.md"], False),
        ("https://github.com/alice", ["docs/source/other/contributors.md"], False),
        ("https://github.com/owner/repo/pull/1", ["docs/source/community/contributors.md"], False),
    ],
)
def test_contributor_profile_skip_is_scoped_to_all_sources(checker_module, settings, url, sources, skipped):
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200)

    results, _ = run_check(checker_module, settings, [url], handler, sources=sources)
    assert results[0].status == ("skipped" if skipped else "passed")
    assert calls == ([] if skipped else [url])


def test_reports_include_failure_reasons_sources_and_attempts(checker_module, settings, tmp_path):
    results, _ = run_check(
        checker_module,
        settings,
        ["https://a.example/a", "https://a.example/gone"],
        lambda request: httpx.Response(404 if request.url.path == "/gone" else 200),
    )
    checker_module.write_reports(tmp_path, results, 0.5)
    report = json.loads((tmp_path / "output.json").read_text())
    assert report["counts"] == {"passed": 1, "broken": 1}
    assert report["links"][1]["sources"] == ["docs/source/test.md"]
    assert report["links"][1]["attempts"] == 1
    assert [event["method"] for event in report["links"][1]["history"]] == ["HEAD", "GET"]
    summary = (tmp_path / "summary.md").read_text()
    assert "https://a.example/gone" in summary and "HTTP 404" in summary


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
    monkeypatch.setattr(checker_module, "example_reason", lambda url: "")
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


def test_explicit_403_rate_limit_is_classified_separately(checker_module, settings):
    results, _ = run_check(
        checker_module,
        replace(settings, total_timeout=0.025),
        [HUAWEI_HELP_URL],
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


@pytest.mark.parametrize("status", [404, 410])
def test_permanent_failures_never_enter_compensation(checker_module, settings, status):
    def handler(request):
        return httpx.Response(status if request.url.path == "/gone" else 503)

    results, _ = run_check(checker_module, settings, ["https://a.example/gone", "https://a.example/flaky"], handler)
    assert results[0].status == "broken" and results[0].attempts == 1
    assert all(event["phase"] == "initial" for event in results[0].history)
    assert [event["method"] for event in results[0].history] == (["HEAD", "GET"] if status == 404 else ["HEAD"])
    assert results[1].attempts == settings.initial_attempts + settings.recheck_attempts


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


@pytest.mark.parametrize("value", ["guide.md", [123]])
def test_invalid_file_list_fails_instead_of_checking_nothing(checker_module, tmp_path, monkeypatch, value):
    paths = tmp_path / "files.json"
    paths.write_text(json.dumps(value))
    monkeypatch.setattr(sys, "argv", ["check_md_links.py", "--files-from", str(paths)])
    with pytest.raises(ValueError, match="JSON array of strings"):
        asyncio.run(checker_module.main())


def test_large_host_keeps_fixed_high_concurrency(checker_module, settings):
    active = 0
    peak = 0

    async def handler(request):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.04)
        active -= 1
        return httpx.Response(200)

    settings = replace(settings, global_concurrency=12, total_timeout=5)
    results, _ = run_check(checker_module, settings, [f"https://a.example/{index}" for index in range(400)], handler)
    assert all(result.accepted for result in results)
    assert peak == 12


def test_unknown_server_error_is_not_mislabeled_as_broken(checker_module, settings):
    results, _ = run_check(checker_module, settings, ["https://a.example/help"], lambda request: httpx.Response(567))
    assert results[0].status == "network_error"
    assert len(results[0].history) == settings.initial_attempts + settings.recheck_attempts


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
def test_invalid_rate_reset_does_not_block_other_requests(checker_module, settings, reset):
    def handler(request):
        if request.url.path == "/limit":
            return httpx.Response(403, headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": reset})
        return httpx.Response(200)

    results, _ = run_check(
        checker_module,
        replace(settings, global_concurrency=1, recheck_attempts=0),
        ["https://a.example/limit", "https://a.example/healthy"],
        handler,
    )
    assert [result.status for result in results] == ["rate_limited", "passed"]


def test_intermittent_limits_do_not_abort_healthy_host(checker_module, settings):
    calls = Counter()

    def handler(request):
        calls[request.url.path] += 1
        limited = request.url.path in {"/0", "/10", "/20"} and calls[request.url.path] == 1
        return httpx.Response(429 if limited else 200)

    urls = [f"https://a.example/{index}" for index in range(30)]
    results, _ = run_check(checker_module, settings, urls, handler)
    assert all(result.accepted for result in results)
    assert all(count == (2 if path in {"/0", "/10", "/20"} else 1) for path, count in calls.items())


def test_late_get_rechecks_are_serial_and_stop_on_recovery(checker_module, settings):
    active = 0
    peak = 0
    calls = Counter()

    async def handler(request):
        nonlocal active, peak
        calls[(request.method, request.url.path)] += 1
        if request.method == "HEAD":
            return httpx.Response(503)
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.005)
        active -= 1
        return httpx.Response(503 if calls[("GET", request.url.path)] == 1 else 200)

    results, _ = run_check(checker_module, settings, ["https://a.example/a", "https://b.example/b"], handler)
    assert all(result.accepted for result in results)
    assert peak == 1
    assert calls == Counter({("HEAD", "/a"): 3, ("HEAD", "/b"): 3, ("GET", "/a"): 2, ("GET", "/b"): 2})


def test_domain_skip_list_is_exact_and_does_not_request_skipped_hosts(checker_module, settings):
    requested = []

    def handler(request):
        requested.append(request.url.host)
        return httpx.Response(200)

    results, _ = run_check(
        checker_module,
        settings,
        [
            "https://DeepWiki.com/a",
            "https://deepwiki.com./b",
            "https://mcp.deepwiki.com/",
            "https://deepwiki.com.example/",
        ],
        handler,
    )
    assert [result.status for result in results] == ["skipped", "skipped", "passed", "passed"]
    assert "Vercel browser challenge" in results[0].message
    assert requested == ["mcp.deepwiki.com", "deepwiki.com.example"]


def test_redirect_to_skipped_domain_is_reported_without_following(checker_module, settings):
    requested = []

    def handler(request):
        requested.append(str(request.url))
        return httpx.Response(302, headers={"location": "https://deepwiki.com/vllm-project/vllm-ascend"})

    results, _ = run_check(
        checker_module,
        settings,
        ["https://a.example/wiki"],
        handler,
    )
    assert requested == ["https://a.example/wiki"]
    assert results[0].status == "skipped" and results[0].accepted
    assert "redirect to skipped domain" in results[0].message


def test_deadline_preserves_completed_results_and_cancels_unfinished_work(checker_module, settings):
    async def handler(request):
        if request.url.path == "/slow":
            await asyncio.sleep(10)
        return httpx.Response(404 if request.url.path == "/gone" else 200)

    settings = replace(settings, total_timeout=0.03)
    results, _ = run_check(
        checker_module,
        settings,
        ["https://a.example/gone", "https://a.example/healthy", "https://a.example/slow"],
        handler,
    )
    assert [result.status for result in results] == ["broken", "passed", "unverified"]
    assert results[0].attempts == 1


def test_deadline_during_recheck_delay_stops_without_extra_requests(checker_module, settings):
    settings = replace(settings, total_timeout=0.03, retry_delay=10)
    results, _ = run_check(checker_module, settings, ["https://a.example/slow"], lambda request: httpx.Response(503))
    assert results[0].status == "unverified"
    assert [event["method"] for event in results[0].history] == ["HEAD"] * 3 + ["GET"]
