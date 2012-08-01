import numpy as np
import pylab
from matplotlib import ticker

from read_shear_catalog import read_shear_catalog

def plot_distribution(filename):
    RA,DEC = read_shear_catalog(filename,('Ra','Dec'),None)

    RAmin,RAmax = min(RA), max(RA)
    DECmin,DECmax = min(DEC),max(DEC)
    
    xmin = np.floor(RAmin*60.)
    xmax = np.ceil(RAmax*60.)
    
    ymin = np.floor(DECmin*60.)
    ymax = np.ceil(DECmax*60.)
    
    xrange = np.arange(xmin,xmax+1)/60.
    yrange = np.arange(ymin,ymax+1)/60.
    
    H,xedges,yedges = np.histogram2d(RA,DEC,bins=(xrange,yrange))
    
    pylab.imshow(H.T,
                 origin='lower',
                 interpolation='nearest',
                 extent = (xrange[0],xrange[-1],yrange[0],yrange[-1]),
                 cmap=pylab.cm.binary)
    pylab.colorbar().set_label(r'$\mathdefault{n_{gal}\ (arcmin^{-2})}$')
    pylab.xlabel('RA (deg)')
    pylab.ylabel('DEC (deg)')

    return len(RA)

if __name__ == '__main__':
    from params import BRIGHT_CAT_NPZ, FAINT_CAT_NPZ
    
    pylab.figure()
    NB = plot_distribution(BRIGHT_CAT_NPZ)
    pylab.title('Bright Objects (N = %i)' % NB)
    pylab.savefig('fig/locations_bright.pdf')

    pylab.figure()
    NF = plot_distribution(FAINT_CAT_NPZ)
    pylab.title('Faint Objects (N = %i)' % NF)
    pylab.savefig('fig/locations_faint.pdf')

    pylab.show()
