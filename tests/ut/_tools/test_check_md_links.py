import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / ".github/workflows/scripts/check_md_links.py"
spec = importlib.util.spec_from_file_location("check_md_links", SCRIPT)
assert spec is not None and spec.loader is not None
check_md_links = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_md_links)


@pytest.mark.parametrize("indent", ["", "    ", "        ", "            "])
@pytest.mark.parametrize("fence", ["```", "~~~"])
def test_indented_fences_preserve_prose_links(indent, fence):
    content = (
        "https://example.com/before\n"
        f"{indent}{fence}shell\n"
        f'{indent}curl "http://<PROXY_IP>:8000/v1/models"\n'
        f"{indent}{fence}\n"
        "https://example.com/after\n"
    )
    assert check_md_links.extract_urls(content) == {
        "https://example.com/before",
        "https://example.com/after",
    }
