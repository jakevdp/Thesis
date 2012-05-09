
all: COSMOS_KL.dvi

pdf: COSMOS_KL.pdf

ps: COSMOS_KL.ps

COSMOS_KL.pdf: COSMOS_KL.dvi
	dvipdf COSMOS_KL.dvi

COSMOS_KL.ps: COSMOS_KL.dvi
	dvips COSMOS_KL.dvi

COSMOS_KL.dvi : COSMOS_KL.aux COSMOS_KL.bbl
	latex COSMOS_KL
	latex COSMOS_KL

COSMOS_KL.aux : COSMOS_KL.tex COSMOS_KL.bib
	latex COSMOS_KL.tex

COSMOS_KL.bbl : COSMOS_KL.bib
	bibtex COSMOS_KL

tar: COSMOS_KL.dvi
	tar -czvf vand1119.tgz COSMOS_KL.tex COSMOS_KL.bib COSMOS_KL.bbl  *.eps halo_table.tex

clean:
	rm COSMOS_KL.dvi
	rm COSMOS_KL.aux
	rm COSMOS_KL.log
	rm COSMOS_KL.bbl
	rm COSMOS_KL.blg