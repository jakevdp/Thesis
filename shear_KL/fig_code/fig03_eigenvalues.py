import numpy
import pylab

from matplotlib.ticker import FuncFormatter


def plot_evals(filename,logplot = True,normcuml=True):
    X = numpy.load(filename)
    evals = X['evals']

    xvals = numpy.arange(1,4097)

    #--------------------------------------------------
    ax = pylab.subplot(211)
    if logplot:
        pylab.loglog(xvals,evals,label='Signal+Noise')
        pylab.loglog(xvals,evals-1,label='Signal')
        pylab.loglog(xvals,numpy.ones(len(xvals)),'--k',label='Noise')
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x,*args: '%i'%x))
        pylab.ylim(0.01,10)
        pylab.legend(loc=3)
    else:
        pylab.semilogy(xvals,evals)
        pylab.semilogy(xvals,evals-1)
        pylab.semilogy(xvals,numpy.ones(len(xvals)),'--k')
        
    pylab.grid(True,c='#AAAAAA',zorder=9)
    pylab.ylabel('value per mode')
    pylab.xlim(1,4096)

    
    #--------------------------------------------------
    ax = pylab.subplot(212)
    S = numpy.cumsum(evals-1)
    SN = numpy.cumsum(evals)
    N = 1.*xvals

    if normcuml:
        S /= S[-1]
        SN /= SN[-1]
        N /= N[-1]

    if logplot:
        pylab.semilogx(xvals,SN,label='Signal+Noise')
        pylab.semilogx(xvals,S,label='Signal')
        pylab.semilogx(xvals,N,'--k',label='Noise')
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x,*args: '%i'%x))
    else:
        pylab.plot(xvals,SN,label='Signal+Noise')
        pylab.plot(xvals,S,label='Signal')
        pylab.plot(xvals,N,'--k',label='Noise')
        pylab.legend(loc=4)
    
    pylab.xlim(1,4096)
    pylab.grid(True,c='#AAAAAA')
    pylab.xlabel('mode number')
    pylab.ylabel('cumulative value')

if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.abspath('../'))
    from shear_KL_source import params
    params.load('../run/base_params.dat')

    from shear_KL_source.DES_tile.tools import get_basis_filename
    
    plot_evals( get_basis_filename() )
    pylab.savefig('figs/fig03_eigenvalues.eps')
    pylab.savefig('figs/fig03_eigenvalues.pdf')
    pylab.show()
