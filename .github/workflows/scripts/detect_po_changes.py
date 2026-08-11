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

    # Multi-line table blocks are handled by _flush_paragraph() during
    # extraction — they are split at the separator line, keeping only
    # header rows for translation.

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
            # Link-only cells (e.g. feature names with docs links,
            # GitHub usernames) are structural — they are neither prose
            # nor pure data.  Skip them and let other cells decide.
            continue
        # Pure-text cell with a sentence (period/question/exclamation)
        # means this is a prose row, not a data row.
        if __import__("re").search(r"[.!?]", cell):
            return False

    return has_data_cell


def _flush_paragraph(current: list[str], paragraphs: list[str]) -> None:
    """Flush *current* accumulated lines into *paragraphs*.

    If the paragraph is a multi-line table block (every line is a
    ``|...|`` table row and there's a separator line like ``|:---:|``),
    the block is split at the separator: header rows are kept, data rows
    are discarded.
    """
    para = "\n".join(current).strip()
    if not para:
        current.clear()
        return

    # Check if this is a multi-line table block.
    lines = para.split("\n")
    if len(lines) > 1 and all(TABLE_ROW_RE.match(line.strip()) for line in lines):
        sep_re = __import__("re").compile(r"^\|[\s:-]+\|")
        for i, line in enumerate(lines):
            if sep_re.match(line.strip()):
                # Keep header rows (before separator).
                if i > 0:
                    header = "\n".join(lines[:i])
                    if _is_translatable_paragraph(header):
                        paragraphs.append(header)
                # Treat each data row (after separator) as an individual
                # paragraph and let _is_translatable_paragraph decide
                # whether it contains natural language worth translating.
                for data_line in lines[i + 1 :]:
                    if _is_translatable_paragraph(data_line):
                        paragraphs.append(data_line)
                current.clear()
                return
        # No separator found — fall through to normal handling.

    if _is_translatable_paragraph(para):
        paragraphs.append(para)
    current.clear()


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
    any_content_seen = False

    current: list[str] = []

    for line in lines:
        stripped = line.strip()

        # --- Frontmatter ---
        # Frontmatter ``---`` delimiters are only valid when they appear at the
        # very beginning of the file (before any other content).  A ``---`` that
        # appears after prose has already been encountered is a horizontal rule,
        # not a frontmatter delimiter.
        if stripped == FRONTMATTER_DELIM and not any_content_seen and not frontmatter_started:
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
                _flush_paragraph(current, paragraphs)
                current = []
            continue

        # --- Prose line ---
        any_content_seen = True
        current.append(line)

    # Flush final paragraph.
    if current:
        _flush_paragraph(current, paragraphs)

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
        return {"has_new_or_modified": True, "has_empty": True}
    po_path.parent.mkdir(parents=True, exist_ok=True)
    po_path.write_text(header + body_entries + "\n", encoding="utf-8")
    print(f"  Force-regenerated: {po_path} ({len(paragraphs)} entries)")
    return {"has_new_or_modified": True, "has_empty": True}


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


def _dedup_po_entries(po: POFile) -> int:
    """Remove duplicate entries from *po*, keeping only the first.

    Covers two cases:
    - Duplicate msgid values (same paragraph extracted multiple times).
    - Duplicate empty msgid (``msgid ""``) — only the first (which carries
      the PO header metadata like ``Project-Id-Version``) is kept; any
      additional bare ``msgid ""`` entries are removed.  Without this,
      ``polib`` may shuffle the bare entry's position relative to the
      header block on every save, causing spurious diffs.

    Returns the number of entries removed.
    """
    seen: set[str] = set()
    removed = 0
    empty_seen = False
    for entry in list(po):
        if not entry.msgid:
            if empty_seen:
                po.remove(entry)
                removed += 1
            else:
                empty_seen = True
            continue
        if entry.msgid in seen:
            po.remove(entry)
            removed += 1
        else:
            seen.add(entry.msgid)
    return removed


def process_file(source_path: Path, dry_run: bool = False, force: bool = False) -> dict | None:
    """Create or update the .po file for *source_path*.

    If *force* is True, the entire .po file is regenerated from the source
    markdown, discarding any existing translations.  Otherwise only new
    paragraphs are appended and existing translations are preserved.

    Returns a dict with info about the changes, or None if no changes needed.
    """
    if not source_path.exists():
        return None

    rel = source_path.relative_to(SOURCE_DIR)
    po_path = LOCALE_DIR / rel.with_suffix(".po")

    content = source_path.read_text(encoding="utf-8")
    paragraphs = _extract_paragraphs(content)

    if not paragraphs:
        return None

    if force and po_path.exists():
        return _write_po_from_paragraphs(po_path, str(rel), paragraphs, dry_run)

    if not po_path.exists():
        return _write_po_from_paragraphs(po_path, str(rel), paragraphs, dry_run)

    # Incremental update: detect new, removed, and modified paragraphs.
    po = pofile(str(po_path))

    # Deduplicate existing entries — keep only the first occurrence of each
    # msgid; remove duplicates.  Without this step, repeated
    # invocations on files that already contain duplicate msgid entries
    # (e.g. from prior force rebuilds or buggy incremental runs) would
    # keep appending more copies.
    dedup_removed = _dedup_po_entries(po)

    entries_by_msgid: dict[str, POEntry] = {}
    for entry in po:
        if entry.msgid:
            entries_by_msgid[entry.msgid] = entry

    new_count = 0
    modified_count = 0
    removed_count = 0

    # Track how many times each paragraph appears in the source so that
    # repeated paragraphs (e.g. "Offline example:") consume one PO entry
    # each without creating spurious "new" entries.
    para_counts: dict[str, int] = {}
    for para in paragraphs:
        para_counts[para] = para_counts.get(para, 0) + 1

    for para in paragraphs:
        remaining = para_counts.get(para, 0)
        if para in entries_by_msgid and remaining > 0:
            # This paragraph already has a translated entry in the PO —
            # consume it and keep the translation intact.
            del entries_by_msgid[para]
            para_counts[para] = remaining - 1
            continue
        # If the source has more occurrences of *para* than the PO has
        # entries, we need to create additional entries.  However, if the
        # first occurrence already consumed the sole PO entry for this
        # msgid, the remaining occurrences should NOT create new empty
        # entries — that would overwrite the existing translation.
        # gettext only allows one entry per msgid anyway, so skip
        # additional occurrences of already-matched paragraphs.
        if remaining > 0:
            para_counts[para] = remaining - 1
            # Check whether a PO entry for this msgid already exists (it
            # may have been consumed by an earlier occurrence above, or it
            # may simply not exist).  If it exists anywhere in the PO,
            # reuse it; otherwise create a new entry.
            if _po_has_msgid(po, para):
                continue
            po.append(_build_po_entry(para))
            new_count += 1
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

    # Remove paragraphs that no longer exist in source.
    # Any entry still in entries_by_msgid is not in the new paragraphs list.
    for old_entry in entries_by_msgid.values():
        po.remove(old_entry)
        removed_count += 1

    change_count = new_count + modified_count + removed_count + dedup_removed
    if change_count == 0:
        if _has_empty_msgstr(po):
            untranslated = sum(1 for e in po if e.msgid and not e.msgstr)
            if dry_run:
                print(f"  [DRY-RUN] {po_path}")
                print(f"            -> {untranslated} untranslated (no structural changes)")
            else:
                print(f"  Untranslated: {po_path}")
                print(f"                -> {untranslated} untranslated (no structural changes)")
            return {"has_new_or_modified": False, "has_empty": True}
        return None

    # Classify the reason this file needs attention.
    reasons: list[str] = []
    untranslated = sum(1 for e in po if e.msgid and not e.msgstr)
    if new_count:
        reasons.append(f"{new_count} new paragraph(s) to translate")
    if modified_count:
        reasons.append(f"{modified_count} paragraph(s) modified (re-translation needed)")
    if untranslated and not new_count and not modified_count:
        reasons.append(f"{untranslated} untranslated")
    if removed_count:
        reasons.append(f"{removed_count} obsolete (source paragraph removed)")
    if dedup_removed:
        reasons.append(f"{dedup_removed} duplicate(s) cleaned")
    reason_str = "; ".join(reasons) if reasons else "structural changes"

    if dry_run:
        print(f"  [DRY-RUN] {po_path}")
        print(f"            -> {reason_str}")
        return {"has_new_or_modified": new_count > 0 or modified_count > 0, "has_empty": untranslated > 0}

    po.save(str(po_path))
    # Work around polib 1.2.0 unstable header serialization (the bare
    # ``msgid ""``/``msgstr ""`` block drifts relative to the metadata
    # header block on consecutive saves).  Loading and saving a second
    # time stabilizes the output — polib's own round-trip is then
    # deterministic.
    po2 = pofile(str(po_path))
    po2.save(str(po_path))
    print(f"  Updated: {po_path}")
    print(f"           -> {reason_str}")
    return {"has_new_or_modified": new_count > 0 or modified_count > 0, "has_empty": untranslated > 0}


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


def _po_has_msgid(po: POFile, msgid: str) -> bool:
    """Return True if *po* contains an entry with the given *msgid*."""
    return any(entry.msgid == msgid for entry in po)


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
    return any(entry.msgid and not entry.msgstr for entry in po)


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
    # Classify files by why they need translation.
    files_with_new: list[str] = []  # has untranslated new/modified paragraphs
    files_obsolete_only: list[str] = []  # only obsolete entries removed, no untranslated
    files_existing_empty: list[str] = []  # no structural changes but has pre-existing empty entries

    for source_path in en_files:
        try:
            info = process_file(source_path, dry_run=args.dry_run, force=args.force)
            if info:
                rel = _relative_to_source(source_path)
                po_rel = str(LOCALE_DIR / Path(rel).with_suffix(".po"))
                if info.get("has_new_or_modified") or info.get("has_empty"):
                    needs_translation.append(po_rel)
                if info.get("has_new_or_modified"):
                    files_with_new.append(po_rel)
                elif info.get("has_empty"):
                    files_existing_empty.append(po_rel)
                else:
                    files_obsolete_only.append(po_rel)
        except Exception as exc:
            print(f"  ERROR processing {source_path}: {exc}", file=sys.stderr)

    # Print summary.
    print(f"\n{'─' * 60}")
    print(f"  Scanned: {len(en_files)} English markdown files")
    print(f"  Files with changes: {len(needs_translation)}")
    if files_with_new:
        print(f"    Need translation (new/modified paragraphs): {len(files_with_new)}")
    if files_existing_empty:
        print(f"    Have pre-existing untranslated entries:     {len(files_existing_empty)}")
    if files_obsolete_only:
        print(f"    Obsolete cleanup only (no translation needed): {len(files_obsolete_only)}")
    print(f"{'─' * 60}")

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
