# MkDocs documentation build file
#
# Usage:
#   make serve      - Build and serve docs locally (English)
#   make serve-zh   - Build and serve docs locally (Chinese)
#   make build      - Build English docs
#   make build-zh   - Build Chinese docs
#   make gen-zh     - Generate Chinese docs from .po files
#   make clean      - Clean build artifacts
#
# Both `build*` targets delegate to tools/rtd_build.sh so the
# Chinese-mode env vars, language detection, and mkdocs flags stay in
# one place (the same script is also used by .readthedocs.yaml).

SPHINXOPTS    ?=
BUILDDIR      = site

.PHONY: help serve serve-zh build build-zh gen-zh clean

help:
	@echo "Available targets:"
	@echo "  serve      - Build and serve docs locally (English)"
	@echo "  serve-zh   - Build and serve docs locally (Chinese)"
	@echo "  build      - Build English docs to $(BUILDDIR)/"
	@echo "  build-zh   - Build Chinese docs to $(BUILDDIR)/zh/"
	@echo "  gen-zh     - Generate Chinese docs from .po files"
	@echo "  clean      - Clean build artifacts"

serve:
	mkdocs serve

serve-zh:
	DOCS_LANG=zh mkdocs serve

build:
	SITE_DIR=$(CURDIR)/$(BUILDDIR) bash tools/rtd_build.sh

build-zh:
	DOCS_LANG=zh SITE_DIR=$(CURDIR)/$(BUILDDIR)/zh bash tools/rtd_build.sh

gen-zh:
	python tools/generate_zh_docs.py

clean:
	rm -rf $(BUILDDIR)
