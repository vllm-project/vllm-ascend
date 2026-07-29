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
from polib import POEntry, POFile, pofile

SYSTEM_PROMPT = (
    "You are a professional technical documentation translator specializing in "
    "English-to-Chinese translations for Sphinx documentation. Return valid gettext "
    "PO entries only, without explanations or markdown fences."
)

TRANSLATION_PROMPT = """Translate these gettext PO entries from English to Chinese.

Each entry has a msgid containing the English source and an empty msgstr.
Fill every msgstr and return exactly the same entries in the same order.

Rules:
1. Return only msgid/msgstr entries. Do not add a PO header, comments, or explanations.
2. Keep every msgid exactly unchanged. Do not add, remove, merge, split, or reorder entries.
3. Preserve format markers, code blocks, references, variables, commands, paths, URLs,
   Markdown syntax, HTML tags, and other technical syntax.
4. Keep person names, contributor names, GitHub usernames, dates, commit hashes, and
   other proper nouns unchanged.
5. For Markdown links, translate display text but keep the URL unchanged.
6. For numbered Chinese list items, do not insert a space after the marker:
   use "1.中文" instead of "1. 中文", because Sphinx may ignore the latter translation.
7. Use consistent and natural Chinese technical terminology. If a term is uncertain,
   keep the original English.
8. Do not add fuzzy flags.

{content}"""

SINGLE_ENTRY_SYSTEM_PROMPT = (
    "You are a professional technical documentation translator specializing in "
    "English-to-Chinese translations. Return only the translated text, without "
    "explanations, labels, or wrapper fences."
)

SINGLE_ENTRY_PROMPT = """Translate the following documentation text from English to Chinese.

Preserve format markers, code blocks, references, variables, commands, paths, URLs,
Markdown/Sphinx syntax, HTML tags, names, and other technical syntax. For Markdown
links, translate the display text but keep the target unchanged.

Return only the translated text.

<source>
{content}
</source>"""

_HEADER_BLOCK_RE = re.compile(r'(?ms)^msgid ""\nmsgstr ""\n(?:"[^\n]*\\n"\n)+(?=\n|$)')
_MARKDOWN_TARGET_RE = re.compile(r"!?\[[^\]\n]*\]\(\s*(?P<target><[^>\n]+>|[^\s)\n]+)")
_RST_TARGET_RE = re.compile(r":(?:ref|doc):`(?:[^`<>]*<)?(?P<target>[^`<>]+)>?`")
_VARIABLE_RE = re.compile(
    r"(?:\$\{[A-Za-z_][A-Za-z0-9_]*\}"
    r"|\$[A-Za-z_][A-Za-z0-9_]*"
    r"|\{[A-Za-z_][A-Za-z0-9_.:-]*\}"
    r"|%\([^)]+\)[#0+\-]?[0-9.]*(?:[diouxXeEfFgGcrs%])"
    r"|(?<!%)%[#0+\-]?[0-9.]*(?:[diouxXeEfFgGcrs%]))"
)
_INVISIBLE_PREFIX_RE = re.compile(r"^[\ufeff\u200b\u200c\u200d\u2060]*")


def _normalize_msgid(text: str) -> str:
    """Normalize harmless whitespace differences in an API-returned msgid."""
    lines = [line.rstrip() for line in text.split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def _restore_ordered_targets(
    source: str,
    translation: str,
    pattern: re.Pattern,
    label: str,
) -> tuple[str, str | None]:
    """Restore protected targets by position while keeping translated labels."""
    source_matches = list(pattern.finditer(source))
    translated_matches = list(pattern.finditer(translation))
    if len(source_matches) != len(translated_matches):
        return translation, (f"{label} count changed ({len(source_matches)} -> {len(translated_matches)})")

    for source_match, translated_match in reversed(list(zip(source_matches, translated_matches))):
        start, end = translated_match.span("target")
        translation = translation[:start] + source_match.group("target") + translation[end:]
    return translation, None


def _normalize_translation_syntax(
    source: str,
    translation: str,
) -> tuple[str, str | None]:
    """Repair safe markup drift and reject changes to protected syntax."""
    normalized = translation
    for pattern, label in (
        (_MARKDOWN_TARGET_RE, "Markdown link target"),
        (_RST_TARGET_RE, "Sphinx reference target"),
    ):
        normalized, error = _restore_ordered_targets(
            source,
            normalized,
            pattern,
            label,
        )
        if error:
            return translation, error

    source_prefix = _INVISIBLE_PREFIX_RE.match(source).group()
    translated_prefix = _INVISIBLE_PREFIX_RE.match(normalized).group()
    normalized = source_prefix + normalized[len(translated_prefix) :]

    source_tokens = _VARIABLE_RE.findall(source)
    translated_tokens = _VARIABLE_RE.findall(normalized)
    if source_tokens != translated_tokens:
        return translation, (f"variable or format marker changed: {source_tokens[:3]} -> {translated_tokens[:3]}")

    return normalized, None


def _remove_extra_headers(content: str) -> str:
    """Remove embedded PO headers while preserving the first catalog header."""
    matches = list(_HEADER_BLOCK_RE.finditer(content))
    if len(matches) <= 1:
        return content

    for match in reversed(matches[1:]):
        start, end = match.span()
        if end < len(content) and content[end] == "\n":
            end += 1
        content = content[:start] + content[end:]
    return content


def _load_po(content: str) -> POFile:
    """Parse a catalog after removing duplicate embedded headers."""
    return pofile(_remove_extra_headers(content))


def _active_entries(po: POFile) -> list[POEntry]:
    return [entry for entry in po if not entry.obsolete]


def _pending_entries(po: POFile, retranslate_all: bool) -> list[POEntry]:
    entries = _active_entries(po)
    if retranslate_all:
        return entries
    return [entry for entry in entries if not entry.msgstr or "fuzzy" in entry.flags]


class POTranslator:
    def __init__(self, api_key: str, max_concurrent: int = 5):
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )
        self.max_concurrent = max_concurrent

    async def _call_api(
        self,
        content: str,
        chunk_info: str = "",
    ) -> str | None:
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
        if not text:
            return None
        cleaned = self._clean_response(text)
        return cleaned if cleaned else None

    async def _call_single_entry_api(
        self,
        entry: POEntry,
        chunk_info: str = "",
    ) -> str | None:
        """Translate one msgid with a simpler response contract."""
        system = SINGLE_ENTRY_SYSTEM_PROMPT
        if chunk_info:
            system = f"{SINGLE_ENTRY_SYSTEM_PROMPT} ({chunk_info})"
        response = await self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": SINGLE_ENTRY_PROMPT.format(content=entry.msgid),
                },
            ],
            max_tokens=8000,
            temperature=0.1,
        )
        text = response.choices[0].message.content
        if not text:
            return None

        po_response = self._clean_response(text)
        if po_response:
            return po_response
        translation = self._clean_plain_translation(text, entry.msgid)
        if not translation:
            return None
        return f"{POEntry(msgid=entry.msgid, msgstr=translation)}\n"

    async def translate_file(
        self,
        po_path: str,
        retranslate_all: bool = False,
    ) -> bool:
        """Translate one PO file and restore it unless every target is valid."""
        path = Path(po_path)
        if not path.exists() or path.suffix != ".po":
            print(f"  Skip: {po_path} (not found or not .po)")
            return False

        backup = po_path + ".bak"
        shutil.copy2(po_path, backup)

        try:
            raw_content = path.read_text(encoding="utf-8-sig")
            po = _load_po(raw_content)
            entries = _active_entries(po)
            targets = _pending_entries(po, retranslate_all)
            mode = "all" if retranslate_all else "pending"
            print(
                f"  {path.name} ({len(targets)}/{len(entries)} {mode})",
                end=" ",
                flush=True,
            )

            if targets:
                snippet = self._build_snippet(targets)
                chunks = self._split_entries(snippet)
                translated_chunks = await self._translate_chunks(chunks)
                if translated_chunks is None:
                    self._restore(backup, po_path)
                    print("FAILED (API)")
                    return False

                translations = self._collect_translations(
                    translated_chunks,
                    targets,
                )
                if translations is None:
                    self._restore(backup, po_path)
                    print("FAILED (merge)")
                    return False

                for entry in targets:
                    entry.msgstr = translations[entry.msgid]
                    entry.flags = [flag for flag in entry.flags if flag != "fuzzy"]

            po.save(str(path), newline="\n")

            error = validate_po_file(path)
            if error:
                self._restore(backup, po_path)
                print(f"FAILED ({error})")
                return False

            print("OK")
            return True
        except Exception as exc:
            self._restore(backup, po_path)
            print(f"ERROR: {exc}")
            return False
        finally:
            Path(backup).unlink(missing_ok=True)

    @staticmethod
    def _restore(backup: str, po_path: str) -> None:
        shutil.copy2(backup, po_path)

    @staticmethod
    def _build_snippet(entries: list[POEntry]) -> str:
        """Serialize only source msgids, never the catalog header or old msgstr."""
        return "\n".join(str(POEntry(msgid=entry.msgid)) for entry in entries)

    @staticmethod
    def _split_entries(
        snippet: str,
        max_chars: int = 6000,
    ) -> list[str]:
        """Split on entry boundaries without leaving an avoidably tiny tail."""
        entries = re.split(r"\n{2,}", snippet.strip())
        chunk_entries: list[list[str]] = []
        current: list[str] = []
        current_chars = 0

        for entry in entries:
            entry_chars = len(entry)
            separator_chars = 2 if current else 0
            if current_chars + separator_chars + entry_chars > max_chars and current:
                chunk_entries.append(current)
                current = []
                current_chars = 0
                separator_chars = 0
            current.append(entry)
            current_chars += separator_chars + entry_chars

        if current:
            chunk_entries.append(current)

        if len(chunk_entries) > 1:
            minimum_tail_chars = min(1000, max_chars // 2)
            previous = chunk_entries[-2]
            tail = chunk_entries[-1]

            def serialized_chars(group: list[str]) -> int:
                return sum(len(entry) for entry in group) + 2 * (len(group) - 1)

            while len(previous) > 1 and serialized_chars(tail) < minimum_tail_chars:
                candidate = previous[-1]
                if serialized_chars([candidate, *tail]) > max_chars:
                    break
                tail.insert(0, previous.pop())

        return ["\n\n".join(group) + "\n" for group in chunk_entries]

    async def _translate_chunks(
        self,
        chunks: list[str],
    ) -> list[str] | None:
        """Translate chunks and recover failed chunks with smaller requests."""
        total = len(chunks)
        sem = asyncio.Semaphore(self.max_concurrent)

        async def attempt_chunk(
            content: str,
            info: str,
            single_entry_contract: bool = False,
        ) -> str | None:
            source_entries = _active_entries(pofile(content))
            for attempt in range(3):
                try:
                    async with sem:
                        if single_entry_contract:
                            result = await self._call_single_entry_api(
                                source_entries[0],
                                chunk_info=info,
                            )
                        else:
                            result = await self._call_api(
                                content,
                                chunk_info=info,
                            )
                    if (
                        result
                        and self._collect_translations(
                            [result],
                            source_entries,
                            quiet=True,
                        )
                        is not None
                    ):
                        return result
                except Exception as exc:
                    if attempt == 2:
                        print(f"\n    {info} API error: {exc}", flush=True)
                        return None
                if attempt < 2:
                    await asyncio.sleep(2 * (attempt + 1))
            print(f"\n    {info} returned an invalid translation", flush=True)
            return None

        async def recover_chunk(
            content: str,
            info: str,
        ) -> list[str] | None:
            source_entries = _active_entries(pofile(content))
            if len(source_entries) == 1:
                recovery_info = f"{info} plain-text single-entry recovery"
                print(
                    f"\n    {info} failed; retrying with the single-entry contract",
                    flush=True,
                )
                result = await attempt_chunk(
                    content,
                    recovery_info,
                    single_entry_contract=True,
                )
                return [result] if result else None

            midpoint = len(source_entries) // 2
            first_half = source_entries[:midpoint]
            second_half = source_entries[midpoint:]
            smaller_chunks = [
                self._build_snippet(first_half),
                self._build_snippet(second_half),
            ]
            print(
                f"\n    {info} failed; retrying as {len(first_half)}+{len(second_half)} entries",
                flush=True,
            )
            recovered = []
            for part, smaller in enumerate(smaller_chunks, start=1):
                part_info = f"{info}.{part}"
                result = await attempt_chunk(smaller, part_info)
                if result:
                    recovered.append(result)
                    continue
                nested = await recover_chunk(smaller, part_info)
                if nested is None:
                    return None
                recovered.extend(nested)
            return recovered

        if total > 1:
            print(
                f"({total} chunks, {self.max_concurrent} parallel)",
                end=" ",
                flush=True,
            )
        results = await asyncio.gather(
            *[attempt_chunk(chunk, f"chunk {idx + 1}/{total}") for idx, chunk in enumerate(chunks)]
        )
        translated: list[list[str] | None] = [[result] if result else None for result in results]

        for idx, chunk_group in enumerate(translated):
            if chunk_group is not None:
                continue
            recovered = await recover_chunk(
                chunks[idx],
                f"chunk {idx + 1}/{total}",
            )
            if recovered is None:
                print(
                    f"\n    Chunk {idx + 1} could not be recovered",
                    flush=True,
                )
                return None
            translated[idx] = recovered

        return [chunk for chunk_group in translated if chunk_group is not None for chunk in chunk_group]

    @staticmethod
    def _collect_translations(
        translated_chunks: list[str],
        expected_entries: list[POEntry],
        quiet: bool = False,
    ) -> dict[str, str] | None:
        """Parse API output and match every translation to an original msgid."""
        expected_by_normalized: dict[str, POEntry] = {}
        for entry in expected_entries:
            normalized = _normalize_msgid(entry.msgid)
            if normalized in expected_by_normalized:
                if not quiet:
                    print("\n    Duplicate normalized msgid in source")
                return None
            expected_by_normalized[normalized] = entry

        translations: dict[str, str] = {}
        try:
            for chunk in translated_chunks:
                translated_po = pofile(chunk)
                for translated in _active_entries(translated_po):
                    normalized = _normalize_msgid(translated.msgid)
                    original = expected_by_normalized.get(normalized)
                    if original is None:
                        if not quiet:
                            print(f"\n    API changed or added msgid: {translated.msgid[:80]}")
                        return None
                    if original.msgid in translations:
                        if not quiet:
                            print(f"\n    API returned a duplicate msgid: {translated.msgid[:80]}")
                        return None
                    if not translated.msgstr:
                        if not quiet:
                            print(f"\n    API left msgstr empty: {translated.msgid[:80]}")
                        return None
                    normalized_msgstr, syntax_error = _normalize_translation_syntax(
                        original.msgid,
                        translated.msgstr,
                    )
                    if syntax_error:
                        if not quiet:
                            print(f"\n    API changed protected syntax for {translated.msgid[:80]}: {syntax_error}")
                        return None
                    translations[original.msgid] = normalized_msgstr
        except Exception as exc:
            if not quiet:
                print(f"\n    Cannot parse API response: {exc}")
            return None

        missing = [entry.msgid for entry in expected_entries if entry.msgid not in translations]
        if missing:
            if not quiet:
                print(f"\n    API omitted {len(missing)} msgid(s), first: {missing[0][:80]}")
            return None
        return translations

    @staticmethod
    def _clean_response(response: str) -> str:
        response = response.strip()
        if response.startswith("```"):
            lines = response.split("\n")
            lines = lines[1:]
            while lines and lines[-1].strip() == "```":
                lines.pop()
            response = "\n".join(lines).strip()
        if 'msgid "' not in response or 'msgstr "' not in response:
            return ""
        return response

    @staticmethod
    def _clean_plain_translation(response: str, source: str) -> str:
        """Remove response wrappers while rejecting malformed PO fragments."""
        response = response.strip()
        source = source.strip()
        if response.startswith("```") and not source.startswith("```"):
            lines = response.split("\n")
            if len(lines) >= 2 and lines[-1].strip() == "```":
                response = "\n".join(lines[1:-1]).strip()

        if re.search(r"(?m)^\s*msg(?:id|str)\b", response):
            return ""

        response = re.sub(
            r"^(?:翻译(?:如下|结果)?|译文|Translation)\s*[:：]\s*",
            "",
            response,
            count=1,
            flags=re.IGNORECASE,
        )
        return response.strip()


def validate_po_file(
    path: Path,
) -> str | None:
    """Return an error message when a translated PO file is unsafe."""
    content = path.read_text(encoding="utf-8")
    if content.startswith("\ufeff"):
        return "UTF-8 BOM"

    headers = list(_HEADER_BLOCK_RE.finditer(content))
    if len(headers) != 1:
        return f"{len(headers)} PO headers"

    try:
        po = pofile(content)
    except Exception as exc:
        return f"parse error: {exc}"

    entries = _active_entries(po)
    empty = [entry for entry in entries if not entry.msgstr]
    if empty:
        return f"{len(empty)} empty msgstr"

    fuzzy = [entry for entry in entries if "fuzzy" in entry.flags]
    if fuzzy:
        return f"{len(fuzzy)} fuzzy entries"
    return None


def validate_files(files_arg: str) -> int:
    file_list = [item.strip() for item in files_arg.split(",") if item.strip()]
    failed = 0
    total_entries = 0

    for filename in file_list:
        path = Path(filename)
        if not path.exists():
            print(f"  FAIL: {filename} does not exist")
            failed += 1
            continue
        error = validate_po_file(path)
        if error:
            print(f"  FAIL: {filename}: {error}")
            failed += 1
            continue
        count = len(_active_entries(pofile(path.read_text(encoding="utf-8"))))
        total_entries += count
        print(f"  OK:   {filename}: {count} translated entries")

    print(f"\nValidated {len(file_list) - failed}/{len(file_list)} file(s), {total_entries} translated entries")
    return 1 if failed or not file_list else 0


async def async_main() -> int:
    parser = argparse.ArgumentParser(description="Sphinx PO file translator (DeepSeek)")
    parser.add_argument("--files", required=True, help="Comma-separated PO file paths")
    parser.add_argument("--output-json", default=os.getenv("OUTPUT_JSON", "/tmp/translation_results.json"))
    parser.add_argument("--api-key", default=os.getenv("DEEPSEEK_API_KEY"))
    parser.add_argument("--max-concurrent", type=int, default=5)
    parser.add_argument(
        "--retranslate-all",
        action="store_true",
        help="Translate every active entry, including non-empty msgstr values",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate PO structure and coverage without calling the API",
    )
    args = parser.parse_args()

    if args.validate_only:
        return validate_files(args.files)

    if not args.api_key:
        print("Error: DEEPSEEK_API_KEY not set")
        return 1

    file_list = [item.strip() for item in args.files.split(",") if item.strip()]
    mode = "full retranslation" if args.retranslate_all else "incremental"
    print(f"Translating {len(file_list)} file(s) in {mode} mode, max_concurrent={args.max_concurrent}")

    translator = POTranslator(
        api_key=args.api_key,
        max_concurrent=args.max_concurrent,
    )
    success_files = []
    for filename in file_list:
        if await translator.translate_file(
            filename,
            retranslate_all=args.retranslate_all,
        ):
            success_files.append(filename)

    failed_files = [filename for filename in file_list if filename not in success_files]
    results = {
        "success_files": success_files,
        "failed_files": failed_files,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "total_files": len(file_list),
        "success_count": len(success_files),
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nResult: {len(success_files)}/{len(file_list)} translated -> {args.output_json}")
    return 0 if file_list and not failed_files else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(async_main()))
