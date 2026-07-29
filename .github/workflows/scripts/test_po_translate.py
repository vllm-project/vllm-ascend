import asyncio
from pathlib import Path
from types import SimpleNamespace

from po_translate import (
    POTranslator,
    _pending_entries,
    _remove_extra_headers,
    validate_po_file,
)
from polib import POEntry, POFile, pofile


def _write_po(path: Path, entries: list[POEntry]) -> None:
    po = POFile()
    po.metadata = {
        "Project-Id-Version": "vllm-ascend",
        "Language": "zh_CN",
        "Content-Type": "text/plain; charset=UTF-8",
    }
    po.extend(entries)
    po.save(str(path), newline="\n")


def _response(
    entries: dict[str, str],
    include_header: bool = False,
) -> str:
    po = POFile()
    if include_header:
        po.metadata = {
            "Project-Id-Version": "api-generated-header",
            "Language": "zh_CN",
        }
    for msgid, msgstr in entries.items():
        po.append(POEntry(msgid=msgid, msgstr=msgstr))
    if include_header:
        return str(po)
    return "\n\n".join(str(entry) for entry in po) + "\n"


def test_pending_entries_include_empty_and_fuzzy() -> None:
    po = POFile()
    translated = POEntry(msgid="Stable", msgstr="稳定")
    empty = POEntry(msgid="New", msgstr="")
    fuzzy = POEntry(msgid="Changed", msgstr="旧翻译", flags=["fuzzy"])
    po.extend([translated, empty, fuzzy])

    assert _pending_entries(po, retranslate_all=False) == [empty, fuzzy]
    assert _pending_entries(po, retranslate_all=True) == [
        translated,
        empty,
        fuzzy,
    ]


def test_incremental_translation_preserves_existing_msgstr(
    tmp_path: Path,
) -> None:
    path = tmp_path / "messages.po"
    _write_po(
        path,
        [
            POEntry(msgid="Stable", msgstr="稳定"),
            POEntry(msgid="New", msgstr=""),
            POEntry(
                msgid="Changed",
                msgstr="旧翻译",
                flags=["fuzzy"],
            ),
        ],
    )
    translator = POTranslator(api_key="test")

    async def call_api(content: str, chunk_info: str = "") -> str:
        assert 'msgid "Stable"' not in content
        return _response({"New": "新增", "Changed": "新翻译"})

    translator._call_api = call_api  # type: ignore[method-assign]
    assert asyncio.run(translator.translate_file(str(path)))

    translated = pofile(str(path))
    assert translated.find("Stable").msgstr == "稳定"
    assert translated.find("New").msgstr == "新增"
    assert translated.find("Changed").msgstr == "新翻译"
    assert "fuzzy" not in translated.find("Changed").flags
    assert validate_po_file(path) is None


def test_full_retranslation_replaces_non_empty_msgstr(
    tmp_path: Path,
) -> None:
    path = tmp_path / "messages.po"
    _write_po(
        path,
        [
            POEntry(msgid="First", msgstr="旧一"),
            POEntry(msgid="Second", msgstr="旧二"),
        ],
    )
    translator = POTranslator(api_key="test")

    async def call_api(content: str, chunk_info: str = "") -> str:
        assert 'msgid "First"' in content
        assert 'msgid "Second"' in content
        return _response({"First": "新一", "Second": "新二"})

    translator._call_api = call_api  # type: ignore[method-assign]
    assert asyncio.run(translator.translate_file(str(path), retranslate_all=True))

    translated = pofile(str(path))
    assert translated.find("First").msgstr == "新一"
    assert translated.find("Second").msgstr == "新二"


def test_partial_api_response_restores_original_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "messages.po"
    _write_po(
        path,
        [
            POEntry(msgid="First", msgstr=""),
            POEntry(msgid="Second", msgstr=""),
        ],
    )
    original = path.read_bytes()
    translator = POTranslator(api_key="test")

    async def call_api(content: str, chunk_info: str = "") -> str:
        return _response({"First": "第一"})

    async def no_sleep(seconds: float) -> None:
        return None

    async def reject_single_entry(
        entry: POEntry,
        chunk_info: str = "",
    ) -> None:
        return None

    translator._call_api = call_api  # type: ignore[method-assign]
    translator._call_single_entry_api = (  # type: ignore[method-assign]
        reject_single_entry
    )
    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    assert not asyncio.run(translator.translate_file(str(path)))
    assert path.read_bytes() == original


def test_failed_chunk_is_recovered_with_smaller_requests(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "messages.po"
    _write_po(
        path,
        [
            POEntry(msgid="First", msgstr=""),
            POEntry(msgid="Second", msgstr=""),
            POEntry(msgid="Third", msgstr=""),
            POEntry(msgid="Fourth", msgstr=""),
        ],
    )
    translator = POTranslator(api_key="test")

    async def call_api(content: str, chunk_info: str = "") -> str:
        entries = [entry for entry in pofile(content) if entry.msgid]
        if len(entries) > 1:
            first = entries[0]
            return _response({first.msgid: f"译-{first.msgid}"})
        only = entries[0]
        return _response({only.msgid: f"译-{only.msgid}"})

    async def no_sleep(seconds: float) -> None:
        return None

    translator._call_api = call_api  # type: ignore[method-assign]
    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    assert asyncio.run(translator.translate_file(str(path)))

    translated = pofile(str(path))
    for msgid in ("First", "Second", "Third", "Fourth"):
        assert translated.find(msgid).msgstr == f"译-{msgid}"


def test_translation_restores_protected_markdown_syntax(
    tmp_path: Path,
) -> None:
    path = tmp_path / "messages.po"
    msgid = "\u200b### See [Section](#source-anchor)"
    _write_po(path, [POEntry(msgid=msgid, msgstr="")])
    translator = POTranslator(api_key="test")

    async def call_api(content: str, chunk_info: str = "") -> str:
        return _response(
            {
                msgid: "### 参见[章节](#translated-anchor)",
            }
        )

    translator._call_api = call_api  # type: ignore[method-assign]
    assert asyncio.run(translator.translate_file(str(path)))

    msgstr = pofile(str(path)).find(msgid).msgstr
    assert msgstr == "\u200b### 参见[章节](#source-anchor)"


def test_split_entries_counts_separators() -> None:
    entries = [
        POEntry(msgid="a" * 20),
        POEntry(msgid="b" * 20),
        POEntry(msgid="c" * 20),
    ]
    snippet = POTranslator._build_snippet(entries)
    chunks = POTranslator._split_entries(snippet, max_chars=60)

    assert len(chunks) == 3
    assert all(len(chunk.rstrip("\n")) <= 60 for chunk in chunks)


def test_split_entries_rebalances_tiny_tail() -> None:
    entries = [POEntry(msgid=f"entry-{index}-{'x' * 100}") for index in range(5)]
    serialized = [str(POEntry(msgid=entry.msgid, msgstr="")) for entry in entries]
    max_chars = sum(len(entry) for entry in serialized[:4]) + 6
    snippet = POTranslator._build_snippet(entries)

    chunks = POTranslator._split_entries(snippet, max_chars=max_chars)
    chunk_sizes = [len([entry for entry in pofile(chunk) if entry.msgid]) for chunk in chunks]
    chunk_msgids = [entry.msgid for chunk in chunks for entry in pofile(chunk) if entry.msgid]

    assert sum(chunk_sizes) == len(entries)
    assert chunk_sizes[-1] > 1
    assert chunk_msgids == [entry.msgid for entry in entries]
    assert all(len(chunk.rstrip("\n")) <= max_chars for chunk in chunks)


def test_single_entry_plain_text_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "messages.po"
    msgid = "See [the guide](#source-anchor)"
    _write_po(path, [POEntry(msgid=msgid, msgstr="")])
    translator = POTranslator(api_key="test")

    async def call_api(content: str, chunk_info: str = "") -> None:
        return None

    class FakeCompletions:
        async def create(self, **kwargs):
            message = SimpleNamespace(
                content="翻译：参见[指南](#translated-anchor)",
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message)],
            )

    async def no_sleep(seconds: float) -> None:
        return None

    translator._call_api = call_api  # type: ignore[method-assign]
    translator.client = SimpleNamespace(  # type: ignore[assignment]
        chat=SimpleNamespace(completions=FakeCompletions()),
    )
    monkeypatch.setattr(asyncio, "sleep", no_sleep)

    assert asyncio.run(translator.translate_file(str(path)))
    assert pofile(str(path)).find(msgid).msgstr == "参见[指南](#source-anchor)"


def test_plain_text_fallback_rejects_partial_po() -> None:
    response = 'msgid "Source"\n这里缺少 msgstr'

    assert POTranslator._clean_plain_translation(response, "Source") == ""


def test_api_generated_header_is_not_merged(tmp_path: Path) -> None:
    path = tmp_path / "messages.po"
    _write_po(path, [POEntry(msgid="New", msgstr="")])
    translator = POTranslator(api_key="test")

    async def call_api(content: str, chunk_info: str = "") -> str:
        return _response({"New": "新增"}, include_header=True)

    translator._call_api = call_api  # type: ignore[method-assign]
    assert asyncio.run(translator.translate_file(str(path)))

    content = path.read_text(encoding="utf-8")
    assert content.count("Project-Id-Version:") == 1
    assert "api-generated-header" not in content
    assert pofile(str(path)).find("New").msgstr == "新增"


def test_remove_extra_headers_keeps_entries() -> None:
    header = 'msgid ""\nmsgstr ""\n"Project-Id-Version: vllm-ascend\\\\n"\n"Language: zh_CN\\\\n"\n'
    embedded = 'msgid ""\nmsgstr ""\n"Project-Id-Version: generated-chunk\\\\n"\n"Language: zh_CN\\\\n"\n'
    content = f'{header}\nmsgid "Before"\nmsgstr "之前"\n\n{embedded}\nmsgid "After"\nmsgstr "之后"\n'

    cleaned = _remove_extra_headers(content)
    assert cleaned.count("Project-Id-Version:") == 1
    assert 'msgid "Before"' in cleaned
    assert 'msgid "After"' in cleaned
