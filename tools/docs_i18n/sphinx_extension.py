from docutils import nodes
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


def setup(app):
    app.add_transform(RestoreTabTableCellSource)
    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
