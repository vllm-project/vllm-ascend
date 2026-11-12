#!/usr/bin/env python3
"""
post_comment.py - 评测结果发布到 PR：

  1. 总结评论（幂等）：按 MARKER 找到既有机器人评论原位更新；
  2. 行内评论（--inline）：F/S 级问题以 Review API 钉到 diff 具体行，
     仅锚定本次 diff 内的行（其余回退到总结评论）；
  3. 过期治理：发布新行内评论前，删除上一轮机器人留下的行内评论，
     保证 PR 上只保留本轮的行内评论。

用法:
  GITHUB_TOKEN=... python post_comment.py --repo <owner/name> --pr <n> \
      --comment-file comment.md [--inline inline.json] [--head-sha <sha>]

环境变量:
  GITHUB_TOKEN        GitHub API 令牌（workflow 中传 ${{ github.token }}）
  GITHUB_API_URL      默认 https://api.github.com（GHES 自适应）
  DOC_EVAL_MAX_INLINE 行内评论条数上限，默认 10
"""

import contextlib
import json
import os
import sys
import urllib.error
import urllib.request

import regex as re

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

MAX_BODY = 65000
HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")

def api(method, url, token, payload=None):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "User-Agent": "doc-quality-bot",
        },
    )
    with urllib.request.urlopen(req) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body) if body else {}


# ------------------------------------------------ 过期评论治理 ----


def delete_outdated_inline(api_base, repo, pr, token, marker):
    """删除上一轮机器人留下的行内评论，返回删除条数。

    走 REST 而非 GraphQL：GITHUB_TOKEN 访问 reviewThreads 会被拒
    ("Resource not accessible by integration")。机器人行内评论每轮全量重发，
    删除旧的比 resolve 更干净——PR 上只保留本轮的行内评论。
    """
    deleted = 0
    for page in range(1, 11):  # 上限 10 页 × 100 条
        comments = api("GET", f"{api_base}/repos/{repo}/pulls/{pr}/comments?per_page=100&page={page}", token)
        if not comments:
            break
        for c in comments:
            if marker not in (c.get("body") or ""):
                continue
            try:
                api("DELETE", f"{api_base}/repos/{repo}/pulls/comments/{c['id']}", token)
                deleted += 1
            except urllib.error.HTTPError as e:
                print(f"[post_comment] 删除过期行内评论 {c['id']} 失败: {e}", file=sys.stderr)
        if len(comments) < 100:
            break
    return deleted


# ------------------------------------------------ diff 行解析 ----


def fetch_diff_lines(api_base, repo, pr, token):
    """从 files API 的 patch 解析每个文件「新侧」可锚定行号集合。"""
    lines_map = {}
    page = 1
    while True:
        files = api("GET", f"{api_base}/repos/{repo}/pulls/{pr}/files?per_page=100&page={page}", token)
        if not files:
            break
        for f in files:
            patch = f.get("patch")
            if not patch:
                # 超过 300 行的 diff patch 被截断或二进制文件，无法定位
                lines_map[f["filename"]] = None
                continue
            hits = set()
            start = None
            for ln in patch.splitlines():
                m = HUNK_RE.match(ln)
                if m:
                    start = int(m.group(1))
                    continue
                if start is None:
                    continue
                if ln.startswith("+") or ln.startswith(" "):
                    hits.add(start)
                    start += 1
                elif ln.startswith("-") or ln.startswith("\\"):
                    continue
            lines_map[f["filename"]] = hits
        if len(files) < 100:
            break
        page += 1
    return lines_map


# ------------------------------------------------ 发布 ----


def find_existing(api_base, repo, pr, token, marker):
    page = 1
    while True:
        data = api("GET", f"{api_base}/repos/{repo}/issues/{pr}/comments?per_page=100&page={page}", token)
        if not data:
            return None
        for c in data:
            if marker in (c.get("body") or ""):
                return c
        page += 1


def post_summary(api_base, repo, pr, token, body, marker):
    existing = find_existing(api_base, repo, pr, token, marker)
    if existing:
        updated = api("PATCH", f"{api_base}/repos/{repo}/issues/comments/{existing['id']}", token, {"body": body})
        print(f"[post_comment] 已更新总结评论: {updated.get('html_url')}")
    else:
        created = api("POST", f"{api_base}/repos/{repo}/issues/{pr}/comments", token, {"body": body})
        print(f"[post_comment] 已发布总结评论: {created.get('html_url')}")


def post_inline(api_base, repo, pr, token, head_sha, candidates, max_inline):
    """行内评论：仅锚定 diff 内的行；整包失败则逐条回退。"""
    allowed = fetch_diff_lines(api_base, repo, pr, token)
    picked, skipped = [], []
    for c in candidates[:max_inline]:
        hits = allowed.get(c["path"])
        if hits is None or c["line"] not in hits:
            skipped.append(f"{c['path']}#L{c['line']}")
            continue
        picked.append({"path": c["path"], "line": c["line"], "side": "RIGHT", "body": c["body"]})
    if not picked:
        print("[post_comment] 无可锚定的行内评论（候选均不在本次 diff 内）")
        return 0
    review_body = (
        f"Doc Quality Review: **{len(picked)}** Critical/Major issue(s) "
        f"are anchored on diff lines below. Score and the full issue "
        f"list are in the pinned summary comment.\n\n"
        f"<!-- doc-quality-bot review v1 -->"
    )
    payload = {"commit_id": head_sha, "event": "COMMENT", "body": review_body, "comments": picked}
    try:
        review = api("POST", f"{api_base}/repos/{repo}/pulls/{pr}/reviews", token, payload)
        print(f"[post_comment] 行内评论已发布: {review.get('html_url')}（{len(picked)} 条）")
        return len(picked)
    except urllib.error.HTTPError as e:
        detail = ""
        if hasattr(e, "read"):
            with contextlib.suppress(Exception):
                detail = e.read().decode("utf-8", errors="replace")[:300]
        print(f"[post_comment] 整包提交失败（{e} {detail}），降级为逐条提交", file=sys.stderr)
    ok = 0
    for c in picked:
        try:
            api(
                "POST",
                f"{api_base}/repos/{repo}/pulls/{pr}/reviews",
                token,
                {"commit_id": head_sha, "event": "COMMENT", "body": review_body, "comments": [c]},
            )
            ok += 1
        except urllib.error.HTTPError as e:
            print(f"[post_comment] 单条失败 {c['path']}#L{c['line']}: {e}", file=sys.stderr)
    print(f"[post_comment] 逐条提交完成: {ok}/{len(picked)}")
    if skipped:
        print(f"[post_comment] 未在 diff 内的候选（仅留在总结评论）: {', '.join(skipped)}")
    return ok


def main():
    import argparse

    parser = argparse.ArgumentParser(description="发布 PR 评测评论（总结+行内+过期治理）")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--pr", required=True, type=int)
    parser.add_argument("--comment-file", required=True)
    parser.add_argument("--inline", default=None, help="inline.json 路径")
    parser.add_argument("--head-sha", default=None, help="行内评论锚定的 commit")
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN", "")
    api_base = os.environ.get("GITHUB_API_URL", "https://api.github.com").rstrip("/")
    if not token:
        print("[post_comment] 缺少 GITHUB_TOKEN，跳过发布", file=sys.stderr)
        return 2

    # 1) 过期治理：先删掉上一轮机器人留下的行内评论
    try:
        n = delete_outdated_inline(api_base, args.repo, args.pr, token, "<!-- doc-quality-bot inline v1 -->")
        if n:
            print(f"[post_comment] 已删除上一轮行内评论 {n} 条")
    except Exception as e:  # noqa: BLE001 —— 治理失败不阻塞发布
        print(f"[post_comment] 过期评论治理失败（忽略）: {e}", file=sys.stderr)

    # 2) 总结评论（幂等更新）
    with open(args.comment_file, encoding="utf-8") as f:
        body = f.read()
    if len(body) > MAX_BODY:
        body = body[:MAX_BODY] + "\n\n<sup>（超出长度限制已截断）</sup>"
    marker_line = body.splitlines()[0] if body.splitlines() else ""
    post_summary(api_base, args.repo, args.pr, token, body, marker_line)

    # 3) 行内评论
    if args.inline and args.head_sha and os.path.isfile(args.inline):
        with open(args.inline, encoding="utf-8") as f:
            candidates = json.load(f)
        max_inline = int(os.environ.get("DOC_EVAL_MAX_INLINE", "10"))
        if candidates:
            post_inline(api_base, args.repo, args.pr, token, args.head_sha, candidates, max_inline)
    else:
        print("[post_comment] 未提供 --inline/--head-sha 或文件为空，跳过行内评论")
    return 0


if __name__ == "__main__":
    sys.exit(main())
