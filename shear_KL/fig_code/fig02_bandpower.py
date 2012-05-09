import pylab
import numpy
import sys
import os
from matplotlib.ticker import FuncFormatter

sys.path.append(os.path.abspath('../'))
from shear_KL_source.tools.calcpow import calcpow
from shear_KL_source.DES_tile.tools import get_basis_filename


def plot_bandpower(filename,nmodes=4096,Nell=100,normalize=True,
                   load_saved = True):
    """
    filename is the location of the saved KL decomposition
    """
    powfile = 'pow.npz'

    if load_saved:
        X = numpy.load(powfile)
        pow = X['pow']
        ell = X['ell']
    else:
        X = numpy.load(filename)
        evecs = X['evecs']
        dtheta = X['dtheta']
        
        N2 = evecs.shape[0]
        N = int(numpy.sqrt(N2))
        
        if nmodes is None:
            nmodes = N2

        pow = numpy.zeros( (Nell,nmodes), dtype=float )

        for i in range(nmodes):
            if i%500==0:
                print '%i/%i' % (i,N2)
            evec = evecs[:,i].reshape((N,N))
            ell,p = calcpow(evec,dtheta,dtheta,Nell)

            if normalize:
                p /= numpy.sum(p)

            pow[:,i] = p

    pylab.imshow(pow,
                 origin='lower',
                 interpolation='nearest',
                 cmap=pylab.cm.binary,
                 extent=[0,nmodes,ell[0],ell[-1]],
                 aspect='auto')
    pylab.xlabel('KL mode number') 
    pylab.ylabel(r'$\ell$') 

    return ell,pow

if __name__ == '__main__':
    from shear_KL_source import params
    params.load('../run/base_params.dat')
    
    plot_bandpower( get_basis_filename() )
    draw_lines = False

    if draw_lines:
        xlim = pylab.xlim()
        ylim = pylab.ylim()
        
        pylab.plot([0,900],[6144,6144],'--g')
        pylab.plot([900,900],[0,6144],'--g')
        pylab.savefig('figs/bandpower_marked.eps')
        
        pylab.xlim(xlim)
        pylab.ylim(ylim)
        
    pylab.savefig('figs/fig02_bandpower.eps')
    pylab.savefig('figs/fig02_bandpower.pdf')
    
    pylab.show()
