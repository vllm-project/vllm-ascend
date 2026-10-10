# SPDX-License-Identifier: Apache-2.0
"""Requires docs dependencies, pytest, Playwright and Google Chrome.

python -m pytest tests/e2e/docs/test_tabbed_table_layout.py --confcutdir=tests/e2e/docs
"""

from pathlib import Path

import pytest

material = pytest.importorskip("material")
playwright = pytest.importorskip("playwright.sync_api")
DOCS = Path(__file__).resolve().parents[3] / "docs/source"


@pytest.fixture(scope="module")
def browser():
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(channel="chrome")
        yield browser
        browser.close()


def table_markup(wide):
    width = ' style="min-width: 1600px"' if wide else ""
    return (
        '<div class="table-fullscreen-wrapper"><div class="md-typeset__scrollwrap">'
        f'<div class="md-typeset__table"><table{width}>'
        "<thead><tr><th>Model</th><th>Support</th></tr></thead>"
        "<tbody><tr><td>Model A</td><td>Yes</td></tr></tbody>"
        "</table></div></div></div>"
    )


@pytest.mark.parametrize("width", [390, 768, 1280])
def test_tabbed_tables_scroll_inside_border_and_plain_tables_keep_layout(browser, width):
    with browser.new_page(viewport={"width": width, "height": 800}) as page:
        labels = ["950DT Products", "A2/A3", "Atlas 300I DUO"]
        radios = "".join(
            f'<input type="radio" id="tab-{i}" name="hardware" {"checked" if i == 0 else ""}>' for i in range(3)
        )
        tabs = "".join(f'<label for="tab-{i}">{label}</label>' for i, label in enumerate(labels))
        panels = "".join(f'<div class="tabbed-block">{table_markup(i < 2)}</div>' for i in range(3))
        page.set_content(
            '<html class="no-js"><body data-md-color-scheme="default"><article class="md-typeset fixture">'
            f'<div id="plain">{table_markup(True)}</div>'
            f'<div class="tabbed-set tabbed-alternate">{radios}'
            f'<div class="tabbed-labels">{tabs}</div><div class="tabbed-content">{panels}</div>'
            "</div></article></body></html>"
        )
        theme_css = Path(material.__file__).parent / "templates/assets/stylesheets"
        page.add_style_tag(path=next(theme_css.glob("main.*.min.css")))
        page.add_style_tag(path=DOCS / "stylesheets/extra.css")
        page.add_style_tag(content=".fixture { margin: 24px; }")

        # Ordinary tables retain Material's gutter-spanning scroll viewport.
        plain = page.locator("#plain .md-typeset__scrollwrap").bounding_box()
        plain_wrapper = page.locator("#plain .table-fullscreen-wrapper").bounding_box()
        assert plain["x"] < plain_wrapper["x"]
        assert plain["x"] + plain["width"] > plain_wrapper["x"] + plain_wrapper["width"]

        for i in range(3):
            page.locator(f'label[for="tab-{i}"]').click()
            area = page.locator(".tabbed-block").nth(i).locator(".md-typeset__scrollwrap")
            viewport = area.bounding_box()
            box = page.locator(".tabbed-set").bounding_box()
            assert viewport["x"] >= box["x"]
            assert viewport["x"] + viewport["width"] <= box["x"] + box["width"]
            if i < 2:
                assert area.evaluate("el => { el.scrollLeft = 200; return el.scrollLeft; }") > 0
            else:
                assert area.evaluate("el => el.scrollWidth <= el.clientWidth")
