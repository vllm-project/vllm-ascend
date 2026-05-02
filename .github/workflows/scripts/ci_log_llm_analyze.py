"""
POST failure logs to an OpenAI-compatible chat API (CI only).

Reads `VLLM_ASCEND_CI_LLM_*` from the environment (optional: `vllm_ascend.envs`
when the package on PYTHONPATH matches `vllm_ascend/envs.py`).
If ``VLLM_ASCEND_CI_LLM_ENABLED`` is unset or 0, exits immediately (no HTTP, no log read).
If enabled but API key or base URL is missing, exits 0 after a short skip note to
``$GITHUB_STEP_SUMMARY`` when set.

Extend via VLLM_ASCEND_CI_LLM_BACKEND: only ``openai_compatible`` is implemented;
other values produce a skip message so callers can add backends later.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import ci_log_filter_llm as log_filter  # noqa: E402


def _ci_llm_config_from_env() -> dict[str, str | int]:
    """Mirror defaults in ``vllm_ascend/envs.py`` for images without a fresh install."""
    return {
        "enabled": bool(int(os.getenv("VLLM_ASCEND_CI_LLM_ENABLED", "0"))),
        "api_key": os.getenv("VLLM_ASCEND_CI_LLM_API_KEY", "").strip(),
        "base_url": os.getenv("VLLM_ASCEND_CI_LLM_BASE_URL", "").strip(),
        "model": os.getenv("VLLM_ASCEND_CI_LLM_MODEL", "").strip() or "gpt-4o-mini",
        "backend": os.getenv("VLLM_ASCEND_CI_LLM_BACKEND", "openai_compatible").strip().lower(),
        "max_chars": int(os.getenv("VLLM_ASCEND_CI_LLM_MAX_INPUT_CHARS", "120000")),
        "ctx_before": int(os.getenv("VLLM_ASCEND_CI_LLM_CONTEXT_LINES_BEFORE", "40")),
        "ctx_after": int(os.getenv("VLLM_ASCEND_CI_LLM_CONTEXT_LINES_AFTER", "80")),
        "timeout_s": int(os.getenv("VLLM_ASCEND_CI_LLM_TIMEOUT_S", "120")),
    }


def _ci_llm_config() -> dict[str, str | int | bool]:
    try:
        import vllm_ascend.envs as envs_ascend

        return {
            "enabled": bool(envs_ascend.VLLM_ASCEND_CI_LLM_ENABLED),
            "api_key": (envs_ascend.VLLM_ASCEND_CI_LLM_API_KEY or "").strip(),
            "base_url": (envs_ascend.VLLM_ASCEND_CI_LLM_BASE_URL or "").strip(),
            "model": (envs_ascend.VLLM_ASCEND_CI_LLM_MODEL or "").strip() or "gpt-4o-mini",
            "backend": (envs_ascend.VLLM_ASCEND_CI_LLM_BACKEND or "openai_compatible").strip().lower(),
            "max_chars": int(envs_ascend.VLLM_ASCEND_CI_LLM_MAX_INPUT_CHARS),
            "ctx_before": int(envs_ascend.VLLM_ASCEND_CI_LLM_CONTEXT_LINES_BEFORE),
            "ctx_after": int(envs_ascend.VLLM_ASCEND_CI_LLM_CONTEXT_LINES_AFTER),
            "timeout_s": int(envs_ascend.VLLM_ASCEND_CI_LLM_TIMEOUT_S),
        }
    except Exception:
        return _ci_llm_config_from_env()


def _append_step_summary(text: str) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        print(text, file=sys.stderr)
        return
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(text.rstrip() + "\n")


def _openai_compatible_chat(
    *,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_content: str,
    timeout_s: int = 120,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.2,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError(f"Unexpected API response (no choices): {payload!r:.2000}")
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if not content:
        raise RuntimeError(f"Unexpected API response: {payload!r:.2000}")
    return str(content)


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM-backed CI failure diagnostics (optional).")
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--step-name", default="tests", help="Label for this job in the prompt.")
    parser.add_argument(
        "--structured-summary",
        type=Path,
        default=None,
        help="Optional JSON from ci_log_summary.py --format llm-json.",
    )
    args = parser.parse_args()

    cfg = _ci_llm_config()
    if not bool(cfg["enabled"]):
        return 0

    api_key = str(cfg["api_key"])
    base_url = str(cfg["base_url"])
    model = str(cfg["model"])
    backend = str(cfg["backend"])
    max_chars = int(cfg["max_chars"])
    ctx_before = int(cfg["ctx_before"])
    ctx_after = int(cfg["ctx_after"])
    timeout_s = int(cfg["timeout_s"])

    if not api_key or not base_url:
        _append_step_summary(
            "## LLM failure diagnostics\n\n"
            "_Skipped: set repository secrets `VLLM_ASCEND_CI_LLM_API_KEY` and "
            "`VLLM_ASCEND_CI_LLM_BASE_URL` (OpenAI-compatible prefix, e.g. "
            "`https://api.openai.com/v1`)._\n"
        )
        return 0

    if backend != "openai_compatible":
        _append_step_summary(
            f"## LLM failure diagnostics\n\n"
            f"_Skipped: backend `{backend}` is not implemented. "
            f"Supported: `openai_compatible` (extend `.github/workflows/scripts/ci_log_llm_analyze.py`)._\n"
        )
        return 0

    if not args.log_file.is_file() or args.log_file.stat().st_size == 0:
        _append_step_summary("## LLM failure diagnostics\n\n_Skipped: log file missing or empty._\n")
        return 0

    raw_log = args.log_file.read_text(encoding="utf-8", errors="replace")
    bundle = log_filter.build_llm_log_bundle(
        raw_log,
        context_before=ctx_before,
        context_after=ctx_after,
    )
    bundle = log_filter.clip_text(bundle, max_chars=max_chars)

    extra = ""
    if args.structured_summary and args.structured_summary.is_file():
        try:
            extra = args.structured_summary.read_text(encoding="utf-8", errors="replace")
            extra = log_filter.clip_text(extra, max_chars=min(16_000, max_chars // 4))
            extra = "### Structured summary (ci_log_summary.py)\n\n```json\n" + extra + "\n```\n\n"
        except OSError:
            extra = ""

    system_prompt = (
        "You are a senior CI engineer for the vllm-ascend project (Huawei Ascend NPU + vLLM). "
        "Given filtered CI logs, produce: (1) most likely root cause, "
        "(2) whether it looks like infra flake vs test vs product bug, with evidence, "
        "(3) concrete next checks or fixes. Reply in Chinese. Be concise; reference line prefixes "
        "L<number>: when useful."
    )
    user_content = (
        f"### Failed step\n{args.step_name}\n\n{extra}### Filtered log bundle (two-phase extraction)\n\n{bundle}"
    )

    try:
        reply = _openai_compatible_chat(
            base_url=base_url,
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            user_content=user_content,
            timeout_s=timeout_s,
        )
    except Exception as exc:
        _append_step_summary(f"## LLM failure diagnostics\n\n_Request failed ({type(exc).__name__}): {exc}_\n")
        return 0

    _append_step_summary("## LLM failure diagnostics\n\n" + reply.strip() + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
