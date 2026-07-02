#!/usr/bin/env python3
"""Generate Chinese (zh) markdown files from existing .po translation files.

This script reads the English source files and applies translations from
the .po files in docs/source/locale/zh_CN/LC_MESSAGES/ to produce
Chinese markdown files under docs/source/zh/.

Usage:
    python tools/generate_zh_docs.py
"""

import shutil
import sys
from pathlib import Path

import regex as re
from polib import pofile

SOURCE_DIR = Path("docs/source")
LOCALE_DIR = SOURCE_DIR / "locale" / "zh_CN" / "LC_MESSAGES"
ZH_DIR = SOURCE_DIR / "zh"

FENCE_RE = re.compile(r"^(`{3,}|~{3,})", re.MULTILINE)
INLINE_CODE_RE = re.compile(r"`[^`]+`")
URL_RE = re.compile(r"https?://\S+")
# Protect markdown link destinations so that path components (e.g. "mode"
# in "supported_models.md") are not corrupted by text-level replacements.
MD_LINK_RE = re.compile(r"\]\((?P<dest>[^)]+)\)")


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
    fence_marker = None

    for match in fence_matches:
        marker = match.group(1)
        fence_start = match.start()

        if not in_code:
            if fence_start > pos:
                segments.append((content[pos:fence_start], False))
            pos = match.end()
            in_code = True
            fence_marker = marker[0]
        else:
            if marker[0] == fence_marker:
                segments.append((content[pos : match.end()], True))
                pos = match.end()
                in_code = False
                fence_marker = None

    if pos < len(content):
        segments.append((content[pos:], in_code))

    return segments


def _protect_spans(text: str) -> list:
    """Split a text segment into parts, protecting inline code, URLs, and
    markdown link destinations (paths inside ``[...](...)``).

    Returns a list of (text_part, is_protected) tuples.
    """
    parts = []
    last_end = 0

    combined = re.compile(
        rf"{INLINE_CODE_RE.pattern}"
        rf"|{URL_RE.pattern}"
        rf"|{MD_LINK_RE.pattern}",
    )

    for match in combined.finditer(text):
        if match.start() > last_end:
            parts.append((text[last_end : match.start()], False))
        parts.append((match.group(), True))
        last_end = match.end()

    if last_end < len(text):
        parts.append((text[last_end:], False))

    return parts


def apply_translations(content: str, translations: dict) -> str:
    """Apply translations to markdown content.

    Replacements are restricted to text outside of fenced code blocks,
    inline code, and URLs to avoid corrupting code examples or links.
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

        parts = _protect_spans(seg_text)
        for part_text, is_protected in parts:
            if is_protected:
                result_parts.append(part_text)
                continue

            for msgid, msgstr in sorted_items:
                if msgid.strip() and msgstr.strip():
                    part_text = part_text.replace(msgid, msgstr)

            result_parts.append(part_text)

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
    """Copy logos, images and other non-markdown assets to zh directory."""

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

    # Copy all images/ directories recursively
    for img_dir in sorted(SOURCE_DIR.rglob("images")):
        if img_dir.is_dir():
            rel = img_dir.relative_to(SOURCE_DIR)
            dst = ZH_DIR / rel
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
    translated_paths: set = set()
    for po_path in po_files:
        rel_path = get_relative_path(po_path)
        source_path = SOURCE_DIR / rel_path
        zh_path = ZH_DIR / rel_path

        if source_path.exists():
            translations = parse_po_file(po_path)
            if translations:
                changed = generate_zh_file(source_path, translations, zh_path)
                translated_paths.add(rel_path)
                if changed:
                    print(f"  Generated: {zh_path}")
                    generated += 1
                else:
                    print(f"  No changes: {zh_path}")
            else:
                print(f"  No translations: {po_path}")
        else:
            print(f"  Source not found: {source_path}")

    # Fallback: copy English source files that have no .po translation yet
    # so that the Chinese site still renders those pages (in English).
    # Skip the zh output directory itself to avoid recursive copying.
    copied = 0
    for md_path in sorted(SOURCE_DIR.rglob("*.md")):
        rel = md_path.relative_to(SOURCE_DIR)
        # Skip files already handled by a .po translation.
        if rel in translated_paths:
            continue
        # Skip the zh output directory to prevent self-replication.
        if rel.parts and rel.parts[0] == ZH_DIR.name:
            continue
        zh_dest = ZH_DIR / rel
        zh_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(md_path, zh_dest)
        print(f"  Copied (no .po): {zh_dest}")
        copied += 1

    # Copy assets
    copy_assets()

    print(f"\nGenerated {generated} Chinese files in {ZH_DIR}")
    if copied:
        print(f"Copied {copied} English fallback files (no .po translation)")
    print("Done!")


if __name__ == "__main__":
    main()
