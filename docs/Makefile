# Minimal makefile for Sphinx documentation

SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = denoise
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help Makefile

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
