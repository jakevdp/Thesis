import numpy
import pylab
from matplotlib.ticker import NullFormatter


def plot_modes_6(filename,modes=[0,1,2,3,49,499]):
    assert(len(modes)==6) 
    X = numpy.load(filename)
    evecs = X['evecs']
    print evecs.shape

    N = int( numpy.sqrt(evecs.shape[0]) )

    pylab.figure(figsize=(7,10))
    for i in range(6):
        evec = evecs[:,modes[i]]
        pylab.subplot(321+i)
        pylab.imshow(evec.reshape((N,N)),
                     origin='lower',
                     cmap=pylab.cm.RdGy,
                     extent=(0,60,0,60) )

        pylab.title('n=%i' % (modes[i]+1))
        pylab.colorbar()
        cmax = numpy.max(abs(evec))
        pylab.clim(-cmax,cmax)
        pylab.xlabel('arcmin')
        
        if i%2 == 1:
            pylab.gca().yaxis.set_major_formatter(NullFormatter())
        else:
            pylab.ylabel('arcmin')

def plot_modes_9(filename,modes=[0,1,2,3,4,5,49,499,899]):
    assert(len(modes)==9) 
    X = numpy.load(filename)
    evecs = X['evecs']
    print evecs.shape

    N = int( numpy.sqrt(evecs.shape[0]) )

    pylab.figure(figsize=(8,8))
    for i in range(9):
        evec = evecs[:,modes[i]]
        pylab.subplot(331+i)
        pylab.imshow(evec.reshape((N,N)),
                     origin='lower',
                     cmap=pylab.cm.RdGy,
                     extent=(0,60,0,60) )

        pylab.title('n=%i' % (modes[i]+1))
        #pylab.colorbar()
        cmax = numpy.max(abs(evec))
        pylab.clim(-cmax,cmax)
        
        if i in (0,3,6):
            #pylab.gca().yaxis.set_major_formatter(NullFormatter())
            pylab.ylabel('arcmin')
        if i in (6,7,8):
            pylab.xlabel('arcmin')
            
                              
if __name__ == '__main__':
    import os
    import sys

    sys.path.append(os.path.abspath('../'))
    from shear_KL_source import params
    from shear_KL_source.DES_tile.tools import get_basis_filename

    params.load('../run/base_params.dat')
    
    plot_modes_9( get_basis_filename() )
    pylab.savefig('figs/fig01_eigenmodes.eps')
    pylab.savefig('figs/fig01_eigenmodes.pdf')
    pylab.show()
