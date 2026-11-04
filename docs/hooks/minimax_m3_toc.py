"""Include fourth-level TOC entries for MiniMax-M3 without changing other pages."""

from copy import deepcopy

from markdown import Markdown
from mkdocs.config.defaults import MkDocsConfig
from mkdocs.structure.pages import Page
from mkdocs.structure.toc import get_toc

MODEL_PAGE = "tutorials/models/MiniMax-M3.md"
TOC_DEPTH = 4


def on_page_content(html: str, page: Page, config: MkDocsConfig, **kwargs) -> str:
    if page.file.src_uri != MODEL_PAGE or page.markdown is None:
        return html

    # Rebuild only this page's TOC using an independent Markdown configuration.
    # Keep the rendered HTML and site-wide extension settings unchanged.
    extension_configs = deepcopy(config["mdx_configs"])
    extension_configs.setdefault("toc", {})["toc_depth"] = TOC_DEPTH
    markdown = Markdown(
        extensions=config["markdown_extensions"],
        extension_configs=extension_configs,
    )
    markdown.convert(page.markdown)
    page.toc = get_toc(markdown.toc_tokens)
    return html
