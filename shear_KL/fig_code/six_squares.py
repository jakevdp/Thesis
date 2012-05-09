import pylab
from matplotlib.ticker import NullFormatter

def six_squares(width = 11,
                height = 8,
                Loffset = 0.07,
                Roffset = 0.03,
                Toffset = 0.1,
                Boffset = 0.08,
                Hsep = 0.03,
                ):
    pylab.figure(figsize=(width,height))

    ratio = width * 1./height

    Hwidth = ( 1-Loffset-Roffset-2*Hsep ) * 1./3
    Vwidth = ratio*Hwidth
    Vsep = ( 1-Toffset-Boffset-2*Vwidth ) * 1./2

    ax = [None for i in range(6)]

    rows = ( 1-Toffset-Vwidth,
             Boffset )

    cols = ( Loffset,
             Loffset+Hsep+Hwidth,
             Loffset+2*Hsep+2*Hwidth )

    for r in range(2):
        for c in range(3):
            ax[3*r+c] = pylab.axes( (cols[c],rows[r],
                                    Hwidth,Vwidth) )

    for i in (0,1,2):
        ax[i].xaxis.set_major_formatter(NullFormatter())

    for i in (1,2,4,5):
        ax[i].yaxis.set_major_formatter(NullFormatter())

    return ax

if __name__ == '__main__':
    six_squares(11,8)
    pylab.savefig('tmp.pdf')
