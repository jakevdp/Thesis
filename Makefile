all : thesis.dvi

pdf : thesis.pdf

thesis.pdf : thesis.dvi
	dvipdf thesis.dvi

thesis.dvi : thesis.tex thesis.bib prelim.tex vita.tex chapter1.tex appendixA.tex
	latex thesis
	latex thesis
	bibtex thesis
	latex thesis
