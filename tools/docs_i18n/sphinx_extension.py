from docutils import nodes
from sphinx.errors import SphinxError
from sphinx.transforms import SphinxTransform


def _is_inside_tab_content(node: nodes.Node) -> bool:
    """Return whether a node is inside a sphinx-design tab content node."""
    parent = node.parent
    while parent is not None:
        if isinstance(parent, nodes.Element) and parent.get("design_component") == "tab-content":
            return True
        parent = parent.parent
    return False


class RestoreTabTableCellSource(SphinxTransform):
    """Restore source metadata for tables nested in sphinx-design tabs."""

    # Run before Sphinx's PreserveTranslatableMessages (10) and Locale (20)
    # transforms so both gettext extraction and localized HTML see the cells.
    default_priority = 5

    def apply(self, **kwargs) -> None:
        for table in self.document.findall(nodes.table):
            if not _is_inside_tab_content(table):
                continue

            source = table.source or self.document.get("source")
            line = table.line or 0
            for paragraph in table.findall(nodes.paragraph):
                if not paragraph.source:
                    paragraph.source = source
                if paragraph.line is None:
                    paragraph.line = line


class ValidateTabTableCellTranslations(SphinxTransform):
    """Fail a localized build when a table in a tab was not translated."""

    # Run after Sphinx's Locale transform (20), which sets the ``translated``
    # attribute after applying the message catalog to each translatable node.
    default_priority = 30

    def apply(self, **kwargs) -> None:
        if not self.config.validate_tab_table_translations:
            return

        untranslated = []
        for table in self.document.findall(nodes.table):
            if not _is_inside_tab_content(table):
                continue

            for paragraph in table.findall(nodes.paragraph):
                # Complex tables may contain structural paragraphs without
                # visible text. Sphinx does not extract or translate them.
                if paragraph.astext().strip() and not paragraph.get("translated", False):
                    untranslated.append(f"{paragraph.source}:{paragraph.line}: {paragraph.astext()!r}")

        if untranslated:
            details = "\n".join(f"- {item}" for item in untranslated)
            raise SphinxError("localized table cells nested in tabs were not translated:\n" + details)


def setup(app):
    app.add_config_value("validate_tab_table_translations", False, "env")
    app.add_transform(RestoreTabTableCellSource)
    app.add_transform(ValidateTabTableCellTranslations)
    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
