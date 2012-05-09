
texfile = 'letter.tex'
bibfile = 'letter.bbl'

outfile = 'letter_bib.tex'

biblines = open(bibfile).readlines()
texlines = open(texfile).readlines()

OF = open(outfile,'w')

fig = False
for line in texlines:
    ls = line.strip()
    if ls.startswith(r'\bibliography{'):
        OF.writelines(biblines)
    elif ls.startswith(r'\begin{figure'):
        fig = True
    elif fig:
        if ls.startswith(r'\end{figure'):
            fig = False
    else:
        OF.write(line)
OF.close()
