#!/usr/bin/env python3
"""Detect new or changed English paragraphs and update .po skeleton files.

Walks English markdown sources under docs/source/ (excluding zh/ and
_templates/), extracts translatable paragraphs, and compares them against
existing .po files in docs/source/locale/zh_CN/LC_MESSAGES/.

For each source file:
- If no .po exists → create one with msgid entries for every paragraph,
  all msgstr empty.
- If .po exists → add new msgid entries not yet present; obsolete
  entries (paragraphs no longer in the source) are left in place but
  not flagged for re-translation.

Output is a JSON file listing .po files that need translation (i.e.
contain at least one empty msgstr).

Usage:
    python detect_po_changes.py [--output-json /tmp/po_changes.json]
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from polib import POEntry, POFile, pofile

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
SOURCE_DIR = REPO_ROOT / "docs" / "source"
LOCALE_DIR = SOURCE_DIR / "locale" / "zh_CN" / "LC_MESSAGES"

# Sections of a markdown file that should NOT be extracted as translatable
# paragraphs: frontmatter, fenced code blocks, HTML comments, blank lines.
FRONTMATTER_DELIM = "---"
FENCE_RE = __import__("re").compile(r"^(`{3,}|~{3,})")
COMMENT_RE = __import__("re").compile(r"^<!--.*-->$")

# MkDocs Material extensions that should be recognized but whose
# translatable content is extracted as separate paragraphs.
# - !!! type ["title"]   → admonition (note, warning, tip, etc.)
# - ??? ["title"]        → collapsible/details
# - === "tab label"      → content tabs
ADMONITION_RE = __import__("re").compile(r'^!!!\s+\w+(\s+"[^"]*")?\s*$')
DETAILS_RE = __import__("re").compile(r'^\?\?\?(\s+"[^"]*")?\s*$')
TAB_RE = __import__("re").compile(r'^===\s+"[^"]*"\s*$')

# Table rows that are purely structural or data-only (no natural language
# prose).  Matches rows that look like markdown table rows where every
# cell is either a number, a date, a link, a single word, a checkmark,
# or a short identifier.
TABLE_ROW_RE = __import__("re").compile(r"^\|.*\|$")


def _is_translatable_paragraph(paragraph: str) -> bool:
    """Return True if *paragraph* contains human language worth translating."""
    text = paragraph.strip()
    if not text:
        return False

    # Skip pure markdown anchors like ``<a id="..."></a>``.
    if COMMENT_RE.match(text):
        return False

    # Skip MkDocs Material structural directives that have no
    # translatable text content (e.g. bare "!!! note").
    if ADMONITION_RE.match(text) and '"' not in text:
        return False

    # Skip pure table data rows (contributor tables, feature matrices,
    # supported models, etc.) that contain structured data but no
    # natural language sentences.  Only skip rows that are clearly
    # data-only: every cell is a token, number, date, link, or emoji.
    if TABLE_ROW_RE.match(text) and _is_table_data_row(text):
        return False

    # Count characters that typically appear in natural language.
    alpha = sum(1 for c in text if c.isalpha())
    if alpha == 0:
        return False

    # Skip pure image / link references (e.g. ``![alt](url)`` alone in a
    # paragraph).
    stripped = text
    for pattern in [
        __import__("re").compile(r"!\[.*?\]\(.*?\)"),  # images
        __import__("re").compile(r"\[.*?\]\(.*?\)"),  # links
    ]:
        stripped = pattern.sub("", stripped)
    if stripped.strip() == "":
        return False

    # Skip pure markdown syntax lines (e.g. "---", "***", "===").
    unique = set(c for c in text if not c.isspace())
    return not (unique <= {"-", "*", "=", "_", "~", "|", ":", "+"})


def _is_table_data_row(row: str) -> bool:
    """Return True if *row* is a markdown table row with structured data only.

    A row is considered data-only when it has at least one cell that
    looks like structured data (number, date, URL, checkmark, etc.).
    Pure-text header rows like "| Feature | Support | Note |" are
    left for translation.
    """
    cells = [c.strip() for c in row.split("|")[1:-1]]
    if not cells:
        return False

    has_data_cell = False
    for cell in cells:
        cell = cell.strip()
        if not cell:
            continue
        if cell in {"✅", "❌", "❔", "✔", "✘", "🟠", "🔵", "—"}:
            has_data_cell = True
            continue
        if __import__("re").match(r"^[\d\s\.\/\-,:]+$", cell):
            has_data_cell = True
            continue
        if __import__("re").match(r"^\[.*\]\(.*\)$", cell):
            has_data_cell = True
            continue
        # Pure-text cell with a sentence (period/question/exclamation)
        # means this is a prose row, not a data row.
        if __import__("re").search(r"[.!?]", cell):
            return False

    return has_data_cell


def _extract_paragraphs(content: str) -> list[str]:
    """Extract translatable paragraphs from a markdown source.

    Frontmatter and fenced code blocks are excluded.  Within the remaining
    prose, text runs between blank lines form a paragraph (msgid).  The PO
    format treats multi-line msgid values as a single logical unit, but we
    store each paragraph as a separate entry so that a small change in one
    paragraph does not invalidate all other translations for the same file.
    """
    lines = content.splitlines()
    paragraphs: list[str] = []
    in_frontmatter = False
    in_code = False
    fence_marker = ""
    frontmatter_started = False

    current: list[str] = []

    for line in lines:
        stripped = line.strip()

        # --- Frontmatter ---
        if stripped == FRONTMATTER_DELIM and not frontmatter_started:
            frontmatter_started = True
            in_frontmatter = True
            continue
        if stripped == FRONTMATTER_DELIM and in_frontmatter:
            in_frontmatter = False
            continue

        if in_frontmatter:
            continue

        # --- Fenced code blocks ---
        m = FENCE_RE.match(stripped)
        if m:
            if not in_code:
                # Entering code block.
                if current:
                    para = "\n".join(current).strip()
                    if _is_translatable_paragraph(para):
                        paragraphs.append(para)
                    current = []
                in_code = True
                fence_marker = m.group(1)[0]
                continue
            else:
                # Exiting code block — match marker character (backtick or tilde).
                if stripped[0] == fence_marker:
                    in_code = False
                    fence_marker = ""
                    continue

        if in_code:
            continue

        # --- Blank line → paragraph boundary ---
        if not stripped:
            if current:
                para = "\n".join(current).strip()
                if _is_translatable_paragraph(para):
                    paragraphs.append(para)
                current = []
            continue

        # --- Prose line ---
        current.append(line)

    # Flush final paragraph.
    if current:
        para = "\n".join(current).strip()
        if _is_translatable_paragraph(para):
            paragraphs.append(para)

    return paragraphs


def _build_po_entry(msgid: str) -> POEntry:
    """Create a new POEntry with empty msgstr."""
    return POEntry(msgid=msgid, msgstr="")


def _po_header(source_rel: str) -> str:
    """Return the standard PO header block for a new .po file."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M%z")
    return (
        "# SOME DESCRIPTIVE TITLE.\n"
        "# Copyright (C) 2026, vllm-ascend team\n"
        "# This file is distributed under the same license as the vllm-ascend package.\n"
        "# FIRST AUTHOR <EMAIL@ADDRESS>, 2026.\n"
        "#\n"
        'msgid ""\n'
        'msgstr ""\n'
        '"Project-Id-Version:  vllm-ascend\\n"\n'
        '"Report-Msgid-Bugs-To: \\n"\n'
        f'"POT-Creation-Date: {timestamp}\\n"\n'
        f'"PO-Revision-Date: {timestamp}\\n"\n'
        '"Last-Translator: \\n"\n'
        '"Language: zh_CN\\n"\n'
        '"Language-Team: zh_CN <LL@li.org>\\n"\n'
        '"Plural-Forms: nplurals=1; plural=0;\\n"\n'
        '"MIME-Version: 1.0\\n"\n'
        '"Content-Type: text/plain; charset=utf-8\\n"\n'
        '"Content-Transfer-Encoding: 8bit\\n"\n'
        '"Generated-By: detect_po_changes.py\\n"\n\n'
    )


def _relative_to_source(path: Path) -> str:
    """Return the path relative to SOURCE_DIR."""
    return str(path.relative_to(SOURCE_DIR))


def _write_po_from_paragraphs(po_path: Path, rel: str, paragraphs: list[str], dry_run: bool = False) -> bool:
    """Write a fresh .po file from extracted paragraphs, discarding any existing translations."""
    header = _po_header(str(rel))
    body_entries = "\n\n".join(_po_entry_block(p) for p in paragraphs)
    if dry_run:
        print(f"  [DRY-RUN] Would force-regenerate: {po_path} ({len(paragraphs)} entries)")
        return True
    po_path.parent.mkdir(parents=True, exist_ok=True)
    po_path.write_text(header + body_entries + "\n", encoding="utf-8")
    print(f"  Force-regenerated: {po_path} ({len(paragraphs)} entries)")
    return True


def _po_entry_block(msgid: str) -> str:
    """Build a valid PO entry block (msgid + msgstr) for *msgid*.

    Multi-line msgid values are wrapped with empty-string continuation
    lines as required by the PO format.  The final continuation line
    does NOT carry a trailing \\n.
    """
    lines = msgid.split("\n")
    if len(lines) == 1:
        escaped = msgid.replace("\\", "\\\\").replace('"', '\\"')
        return f'msgid "{escaped}"\nmsgstr ""'
    parts = ['msgid ""']
    for i, line in enumerate(lines):
        escaped = line.replace("\\", "\\\\").replace('"', '\\"')
        if i < len(lines) - 1:
            parts.append(f'"{escaped}\\n"')
        else:
            parts.append(f'"{escaped}"')
    parts.append('msgstr ""')
    return "\n".join(parts)


def process_file(source_path: Path, dry_run: bool = False, force: bool = False) -> bool:
    """Create or update the .po file for *source_path*.

    If *force* is True, the entire .po file is regenerated from the source
    markdown, discarding any existing translations.  Otherwise only new
    paragraphs are appended and existing translations are preserved.

    Returns True if the .po file was created or modified and contains
    at least one untranslated entry.
    """
    if not source_path.exists():
        return False

    rel = source_path.relative_to(SOURCE_DIR)
    po_path = LOCALE_DIR / rel.with_suffix(".po")

    content = source_path.read_text(encoding="utf-8")
    paragraphs = _extract_paragraphs(content)

    if not paragraphs:
        return False

    if force and po_path.exists():
        return _write_po_from_paragraphs(po_path, str(rel), paragraphs, dry_run)

    if not po_path.exists():
        return _write_po_from_paragraphs(po_path, str(rel), paragraphs, dry_run)

    # Incremental update: detect new, removed, and modified paragraphs.
    po = pofile(str(po_path))
    entries_by_msgid: dict[str, POEntry] = {}
    for entry in po:
        if entry.msgid:
            entries_by_msgid[entry.msgid] = entry

    new_count = 0
    modified_count = 0
    removed_count = 0

    for para in paragraphs:
        if para in entries_by_msgid:
            continue
        # Check if this is a modification of an existing paragraph
        # (similar msgid that got updated in the source).
        matched = _find_similar_entry(para, entries_by_msgid)
        if matched is not None:
            old_entry = entries_by_msgid.pop(matched)
            new_entry = _build_po_entry(para)
            new_entry.msgstr = ""  # force re-translation for modified paragraph
            po.append(new_entry)
            modified_count += 1
        else:
            po.append(_build_po_entry(para))
            new_count += 1

    # Mark paragraphs that no longer exist in source as obsolete.
    # Any entry still in entries_by_msgid is not in the new paragraphs list.
    for old_entry in entries_by_msgid.values():
        old_entry.obsolete = True
        removed_count += 1

    change_count = new_count + modified_count + removed_count
    if change_count == 0:
        return _has_empty_msgstr(po)

    if dry_run:
        parts = []
        if new_count:
            parts.append(f"+{new_count} new")
        if modified_count:
            parts.append(f"~{modified_count} modified")
        if removed_count:
            parts.append(f"-{removed_count} removed")
        print(f"  [DRY-RUN] Would update: {po_path} ({', '.join(parts)})")
        return True

    po.save(str(po_path))
    parts = []
    if new_count:
        parts.append(f"+{new_count} new")
    if modified_count:
        parts.append(f"~{modified_count} modified")
    if removed_count:
        parts.append(f"-{removed_count} removed")
    print(f"  Updated: {po_path} ({', '.join(parts)})")
    return True


def _find_similar_entry(new_para: str, entries: dict[str, POEntry]) -> str | None:
    """Check if *new_para* is likely a modified version of an existing entry.

    Returns the msgid of the matching entry, or None.
    Uses a simple heuristic: the first non-trivial line of each paragraph
    must be identical, which handles cases where a paragraph was edited
    by adding/removing lines in the middle or end.
    """
    new_first = _first_significant_line(new_para)
    if not new_first:
        return None
    for msgid in entries:
        if _first_significant_line(msgid) == new_first:
            return msgid
    return None


def _first_significant_line(text: str) -> str:
    """Return the first non-empty, non-link-reference line of *text*."""
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("[") and stripped.endswith(")") and "](" in stripped:
            continue
        return stripped
    return ""


def _has_empty_msgstr(po: POFile) -> bool:
    """Return True if *po* contains at least one entry with an empty msgstr."""
    return any(entry.msgid and not entry.msgstr for entry in po if not entry.obsolete)


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect .po files that need translation.")
    parser.add_argument(
        "--output-json",
        default=os.environ.get("OUTPUT_JSON", "/tmp/po_changes.json"),
        help="Path to write JSON output (default: /tmp/po_changes.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write .po files, just report changes.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force full regeneration of ALL .po files, discarding existing translations.",
    )
    args = parser.parse_args()

    # Collect English markdown files.
    en_files: list[Path] = []
    for f in sorted(SOURCE_DIR.rglob("*.md")):
        if "zh" in f.parts:
            continue
        if "_templates" in f.parts:
            continue
        if "_build" in f.parts:
            continue
        en_files.append(f)

    print(f"Scanning {len(en_files)} English markdown files...")

    if args.force:
        print("--force enabled: regenerating ALL .po files from scratch")

    needs_translation: list[str] = []
    for source_path in en_files:
        try:
            if process_file(source_path, dry_run=args.dry_run, force=args.force):
                rel = _relative_to_source(source_path)
                po_rel = str(LOCALE_DIR / Path(rel).with_suffix(".po"))
                needs_translation.append(po_rel)
        except Exception as exc:
            print(f"  ERROR processing {source_path}: {exc}", file=sys.stderr)

    # Write output JSON.
    result = {
        "files": needs_translation,
        "count": len(needs_translation),
        "has_changes": len(needs_translation) > 0,
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nDone. {len(needs_translation)} file(s) need translation → {args.output_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
