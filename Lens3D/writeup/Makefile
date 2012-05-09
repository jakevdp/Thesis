
all: Lens3D.dvi

pdf: Lens3D.pdf

ps: Lens3D.ps

Lens3D.pdf: Lens3D.dvi
	dvipdf Lens3D.dvi

Lens3D.ps: Lens3D.dvi
	dvips Lens3D.dvi

Lens3D.dvi : Lens3D.tex Lens3D.bib
	latex Lens3D
	bibtex Lens3D
	latex Lens3D
	latex Lens3D

tar: Lens3D.dvi
	tar -czvf vand1119.tgz Lens3D.tex Lens3D.bib Lens3D.bbl  *.eps halo_table.tex

clean:
	rm Lens3D.dvi
	rm Lens3D.aux
	rm Lens3D.log
	rm Lens3D.bbl
	rm Lens3D.blg