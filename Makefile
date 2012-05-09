all : thesis.dvi

pdf : thesis.pdf

thesis.pdf : thesis.dvi
	dvipdf thesis.dvi

thesis.dvi : thesis.tex thesis.bib *.tex
	latex thesis
	latex thesis
	bibtex thesis
	latex thesis
