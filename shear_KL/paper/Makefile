
all: shear_KL.dvi

pdf: shear_KL.pdf

ps: shear_KL.ps

shear_KL.pdf: shear_KL.dvi
	dvipdf shear_KL.dvi

shear_KL.ps: shear_KL.dvi
	dvips shear_KL.dvi

shear_KL.dvi : shear_KL.aux shear_KL.bbl
	latex shear_KL
	latex shear_KL

shear_KL.aux : shear_KL.tex shear_KL.bib
	latex shear_KL.tex

shear_KL.bbl : shear_KL.bib
	bibtex shear_KL

tar: shear_KL.dvi
	tar -czvf vand0922.tgz shear_KL.tex shear_KL.bib shear_KL.bbl  *.eps

clean:
	rm shear_KL.dvi
	rm shear_KL.aux
	rm shear_KL.log
	rm shear_KL.bbl
	rm shear_KL.blg