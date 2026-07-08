#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from pathlib import Path

import regex as re
from openai import AsyncOpenAI
from polib import POEntry, pofile

SYSTEM_PROMPT = (
    "You are a professional technical documentation translator specializing in "
    "translating MkDocs markdown documentation from English to Chinese. "
    "You produce accurate, consistent translations of all gettext PO file entries "
    "without skipping any. You never add explanations, markdown fences, or extra text "
    "outside the PO file content."
)

TRANSLATION_PROMPT = """Translate these PO file entries (gettext format) from English to Chinese.

You are given a list of msgid/msgstr pairs where every msgstr is empty ("").
For each msgid, fill in the Chinese translation in the corresponding msgstr.

This content comes from MkDocs (Material for Mkdocs) markdown documentation
for the vLLM Ascend project (an NPU hardware plugin for vLLM). The documentation
covers installation, user guides, developer guides, and API references.

CRITICAL RULES — violations will cause the translation to be rejected:

--- OUTPUT FORMAT ---
1. Return ONLY the same list of msgid/msgstr pairs with msgstr filled in.
   No markdown code fences (```), no explanations, no summaries, no greetings.
2. Every msgid from the input MUST appear exactly once in the output with its
   corresponding msgstr filled in. Do not drop, merge, split, or reorder entries.
3. Keep msgid lines COMPLETELY UNCHANGED — never modify source text.

--- WHAT TO PRESERVE ---
4. All format specifiers: %s, %d, %f, {}, {{}}, {name}, etc.
5. All markdown syntax: **bold**, *italic*, `inline code`, ```code blocks```,
   [links](urls), ![images](urls), # headings, - lists, 1. ordered lists,
   > blockquotes, | tables, --- horizontal rules.
6. HTML tags and attributes: <div>, <a href>, <img>, etc.
7. Environment variables, file paths, command names, code identifiers.
8. Proper nouns: person names, contributor names, author names, company names,
   product names (vLLM, Ascend, CANN, Huawei, etc.).

--- MkDocs MATERIAL EXTENSIONS ---
These are special MkDocs syntax elements. Keep the KEYWORDS and STRUCTURE
exactly as-is; only translate the human-readable TEXT parts.

9. ADMONITIONS: Lines starting with "!!! type" or "!!! type \"title\"".
   The type keyword (note, warning, tip, danger, etc.) and the "!!!" marker
   MUST stay in English.
   Examples:
     msgid "!!! note"               → msgstr "!!! note"  (no translatable text)
     msgid "!!! warning"            → msgstr "!!! warning"
     msgid "!!! note \"Important\""  → msgstr "!!! note \"重要\""

10. COLLAPSIBLE BLOCKS: Lines starting with "??? \"title\"".
    Keep "???" and the quote syntax; translate only the title text inside quotes.
    Example:
      msgid "??? \"Click here to see 'Build from Dockerfile'\""  → msgstr "??? \"点击这里查看'从Dockerfile构建'\""

11. CONTENT TABS: Lines starting with "=== \"label\"".
    Keep "===" and the quote syntax; translate only the label text.
    Example:
      msgid "=== \"Before using pip\""  → msgstr "=== \"使用pip之前\""

--- TRANSLATION QUALITY ---
12. Use natural, fluent Chinese technical documentation style. Avoid word-by-word
    literal translation. Restructure long English sentences into natural Chinese
    sentence flow.
13. Use standard Chinese technical terminology consistently.
14. For markdown links [text](url): translate the display text in [] but keep the
    URL in () exactly as-is. Example: [Quick Start](quick_start.md) → [快速开始](quick_start.md)
15. For headings (# Title): translate the heading text.
16. DO NOT add "#, fuzzy" markers.
17. If a msgid is purely structural (symbols, code, file paths only), copy it
    verbatim to msgstr — do not attempt to translate.
18. Never invent or guess content. If genuinely unsure about a term, leave it in
    English rather than creating a wrong translation.

{content}"""


class POTranslator:
    def __init__(self, api_key: str, max_concurrent: int = 5):
        self.client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.max_concurrent = max_concurrent

    async def _call_api(self, content: str, chunk_info: str = "") -> str | None:
        prompt = TRANSLATION_PROMPT.format(content=content)
        system = SYSTEM_PROMPT
        if chunk_info:
            system = f"{SYSTEM_PROMPT} ({chunk_info})"
        response = await self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=8000,
            temperature=0.3,
        )
        text = response.choices[0].message.content
        return self._clean_response(text) if text else None

    async def translate_file(self, po_path: str) -> bool:
        path = Path(po_path)
        if not path.exists() or path.suffix != ".po":
            print(f"  Skip: {po_path} (not found or not .po)")
            return False

        backup = po_path + ".bak"
        shutil.copy2(po_path, backup)

        try:
            po = pofile(str(path))
            untranslated = [entry for entry in po if entry.msgid and not entry.msgstr and not entry.obsolete]
            if not untranslated:
                print(f"  {path.name} (0 untranslated, skip)", flush=True)
                return True

            total_entries = len([e for e in po if e.msgid and not e.obsolete])
            print(
                f"  {path.name} ({len(untranslated)}/{total_entries} untranslated)",
                end=" ",
                flush=True,
            )

            # Build a minimal PO snippet with only untranslated entries.
            snippet = self._build_snippet(untranslated)
            chunks = self._split_entries(snippet)
            if len(chunks) > 1:
                translated_chunks = await self._translate_chunks(chunks)
                if translated_chunks is None:
                    shutil.copy2(backup, po_path)
                    print("FAILED")
                    return False
                translated_snippet = "\n\n".join(translated_chunks)
            else:
                translated_snippet = await self._call_api(snippet)
                if translated_snippet is None:
                    shutil.copy2(backup, po_path)
                    print("FAILED")
                    return False

            # Parse the translated snippet and merge back.
            if not self._merge_translations(po, untranslated, translated_snippet):
                shutil.copy2(backup, po_path)
                print("FAILED (merge)")
                return False

            po.save(str(path))
            print("OK")
            return True
        except Exception as e:
            print(f"ERROR: {e}")
            shutil.copy2(backup, po_path)
            return False
        finally:
            Path(backup).unlink(missing_ok=True)

    @staticmethod
    def _build_snippet(entries: list[POEntry]) -> str:
        """Build a minimal PO content string containing only *entries*."""
        parts = []
        for entry in entries:
            escaped = entry.msgid.replace('"', '\\"')
            parts.append(f'msgid "{escaped}"\nmsgstr ""')
        return "\n\n".join(parts) + "\n"

    @staticmethod
    def _split_entries(snippet: str, max_chars: int = 6000) -> list[str]:
        """Split a snippet of msgid/msgstr pairs into chunks on entry boundaries."""
        entries = re.split(r"\n{2,}", snippet.strip())
        chunks: list[str] = []
        current: list[str] = []
        current_chars = 0

        for entry in entries:
            entry_chars = len(entry)
            if current_chars + entry_chars > max_chars and current:
                chunks.append("\n\n".join(current) + "\n")
                current = []
                current_chars = 0
            current.append(entry)
            current_chars += entry_chars

        if current:
            chunks.append("\n\n".join(current) + "\n")

        return chunks if len(chunks) > 1 else [snippet]

    async def _translate_chunks(self, chunks: list[str]) -> list[str] | None:
        total = len(chunks)
        sem = asyncio.Semaphore(self.max_concurrent)

        async def do_chunk(idx: int) -> tuple[int, str | None, str | None]:
            async with sem:
                info = f"chunk {idx + 1}/{total}"
                try:
                    result = await self._call_api(chunks[idx], chunk_info=info)
                    if result is None:
                        return (idx, None, "empty response")
                    return (idx, result, None)
                except Exception as e:
                    return (idx, None, str(e)[:50])

        print(f"({total} chunks, {self.max_concurrent} parallel)", end=" ", flush=True)
        results = await asyncio.gather(*[do_chunk(i) for i in range(total)])

        translated: list[str | None] = [None] * total
        for idx, chunk_text, error in results:
            if error:
                print(f"\n    Chunk {idx + 1} failed: {error}")
                return None
            translated[idx] = chunk_text.strip("\n")

        return translated

    @staticmethod
    def _merge_translations(po, untranslated: list[POEntry], translated_snippet: str) -> bool:
        """Parse translated snippet and merge msgstr values back into *po*."""
        try:
            translated_po = pofile(translated_snippet)
        except Exception:
            return False

        translated_map: dict[str, str] = {}
        for entry in translated_po:
            if entry.msgid and entry.msgstr:
                translated_map[entry.msgid] = entry.msgstr

        for entry in untranslated:
            if entry.msgid in translated_map:
                entry.msgstr = translated_map[entry.msgid]
            else:
                print(f"\n    Missing translation for: {entry.msgid[:60]}...")
                return False

        return True

    @staticmethod
    def _clean_response(response: str) -> str:
        response = response.strip()
        if response.startswith("```"):
            lines = response.split("\n")
            lines = lines[1:]
            while lines and lines[-1].strip() == "```":
                lines.pop()
            response = "\n".join(lines).strip()
        return response


def validate_coverage(files_arg: str) -> int:
    """Check that every msgstr in the given PO files is non-empty.

    Prints a per-file summary and returns 0 when all files pass,
    1 when any file has untranslated entries.
    """
    file_list = [f.strip() for f in files_arg.split(",") if f.strip()]
    total_entries = 0
    untranslated = 0
    failed_files: list[str] = []

    for fp in file_list:
        path = Path(fp)
        if not path.exists():
            print(f"  WARN: {fp} not found, skipping")
            continue
        try:
            po = pofile(str(path))
        except Exception as e:
            print(f"  ERROR: cannot parse {fp}: {e}")
            failed_files.append(fp)
            continue

        empty = [entry for entry in po if entry.msgid and not entry.msgstr and not entry.obsolete]
        file_entries = len([e for e in po if e.msgid and not e.obsolete])
        total_entries += file_entries
        untranslated += len(empty)

        if empty:
            print(f"  FAIL: {path.name} — {len(empty)}/{file_entries} untranslated")
            for e in empty[:5]:
                preview = e.msgid[:80].replace("\n", "\\n")
                print(f'         msgid="{preview}..."')
            if len(empty) > 5:
                print(f"         ... and {len(empty) - 5} more")
            failed_files.append(fp)
        else:
            print(f"  OK:   {path.name} — {file_entries}/{file_entries} translated")

    print(
        f"\nCoverage: {total_entries - untranslated}/{total_entries} translated "
        f"({untranslated} missing) in {len(file_list)} file(s)"
    )
    return 1 if failed_files else 0


async def async_main():
    parser = argparse.ArgumentParser(description="PO File Translator (DeepSeek)")
    parser.add_argument("--files", required=True, help="Comma-separated PO file paths")
    parser.add_argument("--output-json", default=os.getenv("OUTPUT_JSON", "/tmp/translation_results.json"))
    parser.add_argument("--api-key", default=os.getenv("DEEPSEEK_API_KEY"))
    parser.add_argument("--max-concurrent", type=int, default=5)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate translation coverage, do not translate",
    )
    args = parser.parse_args()

    if args.validate_only:
        return validate_coverage(args.files)

    api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("Error: DEEPSEEK_API_KEY not set")
        return 1

    file_list = [f.strip() for f in args.files.split(",") if f.strip()]
    print(f"Translating {len(file_list)} file(s), max_concurrent={args.max_concurrent}")

    translator = POTranslator(api_key=api_key, max_concurrent=args.max_concurrent)
    success_files = []

    for fp in file_list:
        if await translator.translate_file(fp):
            success_files.append(fp)

    results = {
        "success_files": success_files,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_files": len(file_list),
        "success_count": len(success_files),
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nResult: {len(success_files)}/{len(file_list)} translated -> {args.output_json}")
    return 0 if success_files else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(async_main()))
