"""Documentation configuration shim.

The legacy Sphinx ``conf.py`` exposed version variables as JSON when
invoked as a script, so tooling like the ``labeled_doctest`` workflow
and ``tests/e2e/common.sh`` could extract them with ``jq``.

After the migration to mkdocs the source of truth for those variables
is the ``extra:`` block of ``mkdocs.yml``. This file is a thin shim
that re-exports them in the same JSON shape, so existing tooling keeps
working without modification.

Output schema (printed to stdout)::

    {
        "vllm_version": "vX.Y.Z",
        "vllm_ascend_version": "vX.Y.ZrcN",
        "pip_vllm_version": "X.Y.Z",
        "pip_vllm_ascend_version": "X.Y.ZrcN",
        "cann_image_tag": "...",
        "main_python_version": "...",
        "main_cann_version": "...",
        "main_pytorch_torch_npu_version": "...",
        "main_triton_ascend_version": "...",
        "main_vllm_commit": "<sha>",
        "main_vllm_tag": "<tag>",
    }

Edit version variables in ``mkdocs.yml`` (``extra:`` block), not here.

Only the standard library is used so this script can run in minimal
environments (e.g. CI containers that do not have ``mkdocs``,
``pyyaml`` or ``pyyaml_env_tag`` installed).
"""

import json
import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MKDOCS_YML = REPO_ROOT / "mkdocs.yml"

# Only the version / substitution keys are forwarded. The mkdocs.yml
# ``extra:`` block also holds non-substitution fields (e.g. ``is_release``,
# ``social``, ``generator``) which tooling does not consume.
VERSION_KEYS = (
    "vllm_version",
    "vllm_ascend_version",
    "pip_vllm_ascend_version",
    "pip_vllm_version",
    "cann_image_tag",
    "main_python_version",
    "main_cann_version",
    "main_pytorch_torch_npu_version",
    "main_triton_ascend_version",
)

# Matches a top-level ``key: value`` entry inside the ``extra:`` mapping
# at a given indentation level. The caller passes the indent width via
# ``{indent}`` and we only match keys at exactly that depth so that
# nested structures (e.g. ``social:`` -> ``- icon:``) are ignored.
# The value is everything from the first non-space character after the
# colon up to a trailing comment (which we strip). Quoted strings are
# kept verbatim; ``!ENV`` references are resolved to their first
# non-empty env var (or the literal default).
_EXTRA_ENTRY_RE = re.compile(
    r"""
    ^(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*:\s*      # key + colon
    (?P<value>(?:
        "(?:[^"\\]|\\.)*"        |              # double-quoted string
        '(?:[^'\\]|\\.)*'        |              # single-quoted string
        \[[^\]]*\]                |              # flow sequence (e.g. !ENV [...])
        [^\s#][^#]*                            # bare scalar up to comment
    ))
    """,
    re.VERBOSE,
)


def _strip_comment(value):
    """Drop a trailing ``# ...`` comment, respecting quoted strings."""
    in_quote = None
    for i, ch in enumerate(value):
        if in_quote:
            if ch == in_quote:
                in_quote = None
            continue
        if ch in ('"', "'"):
            in_quote = ch
            continue
        if ch == "#" and (i == 0 or value[i - 1].isspace()):
            return value[:i]
    return value


def _coerce(value):
    """Resolve ``!ENV [...]`` references and strip surrounding quotes.

    The mkdocs ``!ENV`` tag has the shape ``[VAR, default, ...]``; the
    first non-empty variable wins, and the last list element is the
    literal default. We do not need full YAML semantics here, only
    enough to read the version scalars.
    """
    value = _strip_comment(value).strip()
    if not value.startswith("!ENV"):
        return _unquote(value)
    # Extract the bracketed argument list.
    m = re.match(r"!ENV\s*\[(?P<body>.*)\]\s*$", value, re.DOTALL)
    if not m:
        return _unquote(value)
    body = m.group("body")
    # Tokenise the bracketed list, respecting quotes.
    tokens, buf, in_quote = [], [], None
    for ch in body:
        if in_quote:
            buf.append(ch)
            if ch == in_quote:
                in_quote = None
            continue
        if ch in ('"', "'"):
            in_quote = ch
            buf.append(ch)
        elif ch == ",":
            tokens.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        tokens.append("".join(buf).strip())

    # Walk tokens: the last token is the literal default; any preceding
    # ones are env-var names. Use the first env var that is set.
    for token in tokens[:-1]:
        var_name = _unquote(token)
        if var_name and var_name in os.environ:
            return os.environ[var_name]
    return _unquote(tokens[-1]) if tokens else ""


def _unquote(value):
    """Strip a single layer of matching single/double quotes."""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
        return value[1:-1]
    return value


def _load_extra_block():
    """Parse the ``extra:`` mapping out of ``mkdocs.yml``.

    We avoid a full YAML load (which would require PyYAML and a
    resolver for mkdocs' ``!ENV`` / ``!!python/name`` tags) by reading
    the file line-by-line and matching the ``key: value`` entries at
    exactly the indent depth of the ``extra:`` block. Nested
    structures (e.g. ``social:`` -> ``- icon:``) are skipped by the
    indent check. The version keys we care about are all plain
    scalars, so this is sufficient.
    """
    text = MKDOCS_YML.read_text(encoding="utf-8")
    lines = text.splitlines()
    in_extra = False
    extra_indent = None
    result = {}
    for line in lines:
        stripped = line.lstrip()
        if not in_extra:
            # Look for the top-level ``extra:`` mapping.
            if re.match(r"^extra\s*:\s*(#.*)?$", stripped):
                in_extra = True
            continue
        # Inside ``extra:``. Blank and comment-only lines don't move us.
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        indent = len(line) - len(stripped)
        if extra_indent is None:
            extra_indent = indent
        # End the block when we dedent past its top level.
        if indent < extra_indent:
            break
        # Skip nested entries (e.g. inside ``social:``).
        if indent > extra_indent:
            continue
        m = _EXTRA_ENTRY_RE.match(stripped)
        if not m:
            continue
        key = m.group("key")
        if key in result:
            # Don't silently overwrite; first wins to match YAML semantics.
            continue
        result[key] = _coerce(m.group("value"))
    return result


def _read_text(path):
    return path.read_text(encoding="utf-8").strip() if path.exists() else "unknown"


def main():
    extra = _load_extra_block()
    output = {key: extra[key] for key in VERSION_KEYS if key in extra}
    # ``main_vllm_commit`` and ``main_vllm_tag`` are not stored in
    # mkdocs.yml; they live as commit-shas in the .github/ directory and
    # are read by ``main.py`` (mkdocs-macros) at build time. Include them
    # here so the diagnostic dump in labeled_doctest matches the legacy
    # output.
    output["main_vllm_commit"] = _read_text(REPO_ROOT / ".github" / "vllm-main-verified.commit")
    output["main_vllm_tag"] = _read_text(REPO_ROOT / ".github" / "vllm-release-tag.commit")
    print(json.dumps(output))


if __name__ == "__main__":
    main()
