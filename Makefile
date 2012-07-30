all : thesis.pdf

thesis.pdf : thesis.tex thesis.bib *.tex
	pdflatex thesis
	bibtex thesis
	pdflatex thesis
	pdflatex thesis
clean :
	rm *.aux
	rm *.bbl
	rm *.blg
	rm *.log