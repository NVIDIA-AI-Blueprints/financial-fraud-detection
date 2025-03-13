# Minimal makefile for Sphinx documentation
SPHINXOPTS    = "-v"

SPHINXBUILD = sphinx-build

SOURCEDIR = source

BUILDDIR = build


.PHONY: html clean

html:
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html

clean:
	rm -rf $(BUILDDIR)
