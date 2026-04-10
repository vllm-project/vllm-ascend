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

from openai import AsyncOpenAI

SYSTEM_PROMPT = (
    "You are a professional technical documentation translation expert, "
    "proficient in English-Chinese technical document translation."
)

TRANSLATION_PROMPT = """Translate this Sphinx PO file (gettext format) from English to Chinese.

Rules:
1. Only modify msgstr "", keep msgid unchanged
2. Preserve format markers: %s, %d, {{}}, **, *, `, etc.
3. Keep code blocks, references, variable names unchanged
4. For already translated msgstr, optimize while maintaining style
5. Maintain complete PO file format and structure
6. Use standard Chinese technical terminology
7. For difficult parts, keep original English
8. Remove "#, fuzzy" markers

Return ONLY the complete PO file content, no extra explanations.

{chunk_info}
{content}"""


class POTranslator:
    def __init__(self, api_key: str, max_concurrent: int = 5):
        self.client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.max_concurrent = max_concurrent

    async def _call_api(self, content: str, chunk_info: str = "") -> str | None:
        """Make a single translation API call."""
        prompt = TRANSLATION_PROMPT.format(content=content, chunk_info=chunk_info)
        response = await self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=8000,
            temperature=0.3,
        )
        text = response.choices[0].message.content
        return self._clean_response(text) if text else None

    async def translate_file(self, po_path: str) -> bool:
        """Translate a single PO file with backup/restore on failure."""
        path = Path(po_path)
        if not path.exists() or path.suffix != ".po":
            print(f"  Skip: {po_path} (not found or not .po)")
            return False

        backup = po_path + ".bak"
        shutil.copy2(po_path, backup)

        try:
            content = path.read_text(encoding="utf-8")
            lines = content.split("\n")
            print(f"  {path.name} ({len(lines)} lines)", end=" ", flush=True)

            if len(lines) > 500:
                success = await self._translate_chunked(po_path, lines)
            else:
                result = await self._call_api(content)
                if result:
                    Path(po_path).write_text(result, encoding="utf-8")
                    success = True
                else:
                    success = False

            if not success:
                shutil.copy2(backup, po_path)
                print("FAILED")
            else:
                print("OK")
            return success
        except Exception as e:
            print(f"ERROR: {e}")
            shutil.copy2(backup, po_path)
            return False
        finally:
            Path(backup).unlink(missing_ok=True)

    async def _translate_chunked(self, po_path: str, lines: list[str]) -> bool:
        """Translate large file in parallel chunks."""
        chunk_size = 300
        total = (len(lines) + chunk_size - 1) // chunk_size
        sem = asyncio.Semaphore(self.max_concurrent)

        async def do_chunk(idx: int) -> tuple[int, list[str] | None, str | None]:
            async with sem:
                start = idx * chunk_size
                end = min((idx + 1) * chunk_size, len(lines))
                chunk = "\n".join(lines[start:end])
                info = f"[Chunk {idx + 1}/{total}]"
                try:
                    result = await self._call_api(chunk, chunk_info=info)
                    if result is None:
                        return (idx, None, "empty response")
                    return (idx, result.split("\n"), None)
                except Exception as e:
                    return (idx, None, str(e)[:50])

        print(f"({total} chunks, {self.max_concurrent} parallel)", end=" ", flush=True)
        results = await asyncio.gather(*[do_chunk(i) for i in range(total)])

        # Check for failures
        translated = [None] * total
        for idx, chunk_lines, error in results:
            if error:
                print(f"\n    Chunk {idx + 1} failed: {error}")
                return False
            translated[idx] = chunk_lines

        # Write result
        final = "\n".join(line for chunk in translated for line in chunk)
        Path(po_path).write_text(final, encoding="utf-8")
        return True

    @staticmethod
    def _clean_response(response: str) -> str:
        """Strip markdown code block wrappers from API response."""
        response = response.strip()
        if response.startswith("```"):
            lines = response.split("\n")
            lines = lines[1:]  # remove opening ```
            while lines and lines[-1].strip() == "```":
                lines.pop()
            response = "\n".join(lines).strip()
        return response


async def async_main():
    parser = argparse.ArgumentParser(description="PO File Translator (DeepSeek)")
    parser.add_argument("--files", required=True, help="Comma-separated PO file paths")
    parser.add_argument("--output-json", default=os.getenv("OUTPUT_JSON", "/tmp/translation_results.json"))
    parser.add_argument("--api-key", default=os.getenv("DEEPSEEK_API_KEY"))
    parser.add_argument("--max-concurrent", type=int, default=5)
    args = parser.parse_args()

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

    # Save results
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
