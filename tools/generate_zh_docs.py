#!/usr/bin/env python3
"""Generate Chinese (zh) markdown files from existing .po translation files.

This script reads the English source files and applies translations from
the .po files in docs/source/locale/zh_CN/LC_MESSAGES/ to produce
Chinese markdown files under docs/source/zh/.

Usage:
    python tools/generate_zh_docs.py
"""

import sys
from pathlib import Path

import regex as re
from polib import pofile

SOURCE_DIR = Path("docs/source")
LOCALE_DIR = SOURCE_DIR / "locale" / "zh_CN" / "LC_MESSAGES"
ZH_DIR = SOURCE_DIR / "zh"

FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})[^\n]*$", re.MULTILINE)
URL_RE = re.compile(r"https?://\S+")
LINK_TARGET_RE = re.compile(r"\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
WHITESPACE_RE = re.compile(r"[ \t\n]+")
MARKDOWN_BLOCK_PREFIX_RE = re.compile(r"(?:[-+*>#|]|\d+[.)])(?:\s|$)")


def parse_po_file(po_path: Path) -> dict:
    """Parse a .po file and return msgid -> msgstr mapping."""
    po = pofile(str(po_path))
    translations = {}
    for entry in po:
        if entry.msgstr and entry.msgid:
            translations[entry.msgid] = entry.msgstr
    return translations


def get_relative_path(po_path: Path) -> Path:
    """Get the relative markdown file path from a .po file path."""
    rel = po_path.relative_to(LOCALE_DIR)
    return rel.with_suffix(".md")


def _split_by_code_blocks(content: str) -> list:
    """Split content into segments tagged as 'code' or 'text'.

    Returns a list of (segment_text, is_code) tuples.
    """
    segments = []
    fence_matches = list(FENCE_RE.finditer(content))
    if not fence_matches:
        segments.append((content, False))
        return segments

    pos = 0
    in_code = False
    fence_char = None  # e.g. '`' or '~'

    for match in fence_matches:
        marker = match.group(1).strip()
        # Extract the info string (everything after the fence markers)
        full = match.group(0).strip()
        info = full[len(marker) :].strip()
        fence_start = match.start()

        if not in_code:
            if fence_start > pos:
                segments.append((content[pos:fence_start], False))
            pos = fence_start
            in_code = True
            fence_char = marker[0]
        else:
            # A closing fence must use the same character and have an empty
            # info string (bare ```).  In CommonMark, a fence with an info
            # string (e.g. ```bash) is always an opening fence, never a
            # closing one.  A bare ``` can close any same-character fence.
            if marker[0] == fence_char and not info:
                segments.append((content[pos : match.end()], True))
                pos = match.end()
                in_code = False
                fence_char = None

    if pos < len(content):
        segments.append((content[pos:], in_code))

    return segments


def _inline_code_ranges(text: str) -> list[tuple[int, int]]:
    """Return Markdown code-span ranges, including multi-backtick spans."""
    ranges = []
    pos = 0

    while True:
        start = text.find("`", pos)
        if start == -1:
            break

        opener_end = start + 1
        while opener_end < len(text) and text[opener_end] == "`":
            opener_end += 1
        delimiter = text[start:opener_end]
        search_from = opener_end

        while True:
            close = text.find(delimiter, search_from)
            if close == -1:
                pos = opener_end
                break

            close_end = close + len(delimiter)
            exact_run = (close == 0 or text[close - 1] != "`") and (close_end == len(text) or text[close_end] != "`")
            if exact_run:
                ranges.append((start, close_end))
                pos = close_end
                break
            search_from = close + 1

    return ranges


def _protected_ranges(text: str) -> list[tuple[int, int]]:
    """Return merged ranges for inline code, URLs, and link targets."""
    ranges = _inline_code_ranges(text)
    ranges.extend(match.span() for match in URL_RE.finditer(text))
    ranges.extend(match.span() for match in LINK_TARGET_RE.finditer(text))
    ranges.sort()

    merged = []
    for start, end in ranges:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _is_reflow_safe(msgid: str) -> bool:
    """Return whether a multiline msgid is a reflowed prose paragraph.

    Markdown block markers on continuation lines make line breaks semantic, so
    those entries must keep using exact matching.
    """
    lines = msgid.splitlines()
    return len(lines) > 1 and all(not MARKDOWN_BLOCK_PREFIX_RE.match(line.lstrip()) for line in lines[1:])


def _reflowed_msgid_pattern(msgid: str) -> re.Pattern:
    """Build a pattern that tolerates single prose line-wrap differences."""
    parts = []
    cursor = 0
    for whitespace in WHITESPACE_RE.finditer(msgid):
        parts.append(re.escape(msgid[cursor : whitespace.start()]))
        value = whitespace.group()
        if value.count("\n") <= 1:
            # Match either spaces on one line or one wrapped line. Never cross
            # a blank line, which would join separate Markdown blocks.
            parts.append(r"(?:[ \t]+|[ \t]*\n[ \t]*)")
        else:
            parts.append(re.escape(value))
        cursor = whitespace.end()
    parts.append(re.escape(msgid[cursor:]))
    return re.compile("".join(parts))


def _apply_segment_translations(text: str, translations: list[tuple[str, str]]) -> str:
    """Apply the longest non-overlapping translations to the source once.

    Matches contained entirely in code spans, URLs, or link targets are
    ignored. Replacements are selected from the original text, so translated
    content cannot be processed again by a shorter msgid.
    """
    protected = _protected_ranges(text)
    candidates = []

    for msgid, msgstr in translations:
        if not msgid.strip() or not msgstr.strip():
            continue
        matches = []
        search_from = 0
        while True:
            start = text.find(msgid, search_from)
            if start == -1:
                break
            end = start + len(msgid)
            matches.append((start, end))
            search_from = end

        # PO entries can retain an older prose line wrapping even when the
        # English paragraph is unchanged. Fall back to a whitespace-tolerant
        # match for prose only; list, quote, table, and heading boundaries keep
        # exact matching because their newlines are structural Markdown.
        if not matches and _is_reflow_safe(msgid):
            matches.extend(match.span() for match in _reflowed_msgid_pattern(msgid).finditer(text))

        for start, end in matches:
            fully_protected = any(start >= range_start and end <= range_end for range_start, range_end in protected)
            if not fully_protected:
                candidates.append((start, end, msgstr))

    # Prefer complete paragraphs over shorter msgids contained within them.
    candidates.sort(key=lambda item: (-(item[1] - item[0]), item[0]))
    selected = []
    selected_starts = []
    for candidate in candidates:
        start, end, _msgstr = candidate
        low = 0
        high = len(selected_starts)
        while low < high:
            middle = (low + high) // 2
            if selected_starts[middle] < start:
                low = middle + 1
            else:
                high = middle
        index = low
        overlaps_previous = index > 0 and selected[index - 1][1] > start
        overlaps_next = index < len(selected) and selected[index][0] < end
        if overlaps_previous or overlaps_next:
            continue
        selected.insert(index, candidate)
        selected_starts.insert(index, start)

    result = []
    cursor = 0
    for start, end, msgstr in selected:
        result.append(text[cursor:start])
        result.append(msgstr)
        cursor = end
    result.append(text[cursor:])
    return "".join(result)


def apply_translations(content: str, translations: dict) -> str:
    """Apply translations to markdown content.

    Replacements are applied once to prose outside fenced code blocks. Longer
    msgids take precedence, while inline code, URLs, and link targets remain
    protected from shorter translations.
    """
    if not translations:
        return content

    sorted_items = sorted(
        translations.items(),
        key=lambda x: len(x[0]),
        reverse=True,
    )

    segments = _split_by_code_blocks(content)
    result_parts = []

    for seg_text, is_code in segments:
        if is_code:
            result_parts.append(seg_text)
            continue

        result_parts.append(_apply_segment_translations(seg_text, sorted_items))

    return "".join(result_parts)


def generate_zh_file(source_path: Path, translations: dict, zh_path: Path) -> bool:
    """Generate a single Chinese markdown file."""
    if not source_path.exists():
        return False

    content = source_path.read_text(encoding="utf-8")
    translated = apply_translations(content, translations)

    zh_path.parent.mkdir(parents=True, exist_ok=True)
    zh_path.write_text(translated, encoding="utf-8")
    return translated != content


def copy_assets():
    """Copy logos and other non-markdown assets to zh directory."""
    import shutil

    # Copy logos
    logos_src = SOURCE_DIR / "logos"
    logos_dst = ZH_DIR / "logos"
    if logos_src.exists():
        if logos_dst.exists():
            shutil.rmtree(logos_dst)
        shutil.copytree(logos_src, logos_dst)

    # Copy assets directory (images, etc.)
    assets_src = SOURCE_DIR / "assets"
    assets_dst = ZH_DIR / "assets"
    if assets_src.exists():
        if assets_dst.exists():
            shutil.rmtree(assets_dst)
        shutil.copytree(assets_src, assets_dst)

    # Copy all images directories referenced by markdown files
    for img_dir in [
        SOURCE_DIR / "community" / "images",
        SOURCE_DIR / "user_guide" / "feature_guide" / "images",
    ]:
        if img_dir.exists():
            dst = ZH_DIR / img_dir.relative_to(SOURCE_DIR)
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(img_dir, dst)


def main():
    if not LOCALE_DIR.exists():
        print(f"Locale directory not found: {LOCALE_DIR}")
        sys.exit(1)

    ZH_DIR.mkdir(parents=True, exist_ok=True)

    # Find all .po files
    po_files = sorted(LOCALE_DIR.rglob("*.po"))
    print(f"Found {len(po_files)} .po files")

    generated = 0
    for po_path in po_files:
        rel_path = get_relative_path(po_path)
        source_path = SOURCE_DIR / rel_path
        zh_path = ZH_DIR / rel_path

        if source_path.exists():
            translations = parse_po_file(po_path)
            if translations:
                changed = generate_zh_file(source_path, translations, zh_path)
                if changed:
                    print(f"  Generated: {zh_path}")
                    generated += 1
                else:
                    print(f"  No changes: {zh_path}")
            else:
                # No translations — copy English source so the file exists for
                # mkdocs strict-mode nav/link checks.
                import shutil

                zh_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, zh_path)
                print(f"  Copied (no translations): {zh_path}")
                generated += 1
        else:
            print(f"  Source not found: {source_path}")

    # Also copy any English .md that has no .po file at all, so mkdocs strict
    # mode can find every file referenced in the nav.
    import shutil

    for src in sorted(SOURCE_DIR.rglob("*.md")):
        if src.is_relative_to(ZH_DIR):
            continue
        rel = src.relative_to(SOURCE_DIR)
        dst = ZH_DIR / rel
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"  Copied (no .po): {dst}")

    # Copy assets
    copy_assets()

    print(f"\nGenerated {generated} Chinese files in {ZH_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
