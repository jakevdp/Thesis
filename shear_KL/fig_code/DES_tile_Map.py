import os
import sys
import pylab
import numpy

sys.path.append(os.path.abspath('../'))
from shear_KL_source import params

from shear_KL_source.Map_peaks import *

from shear_KL_source.DES_tile.tools import get_from_parsed, get_mask

from shear_KL_source.DES_KL.tools import whiskerplot

from shear_KL_source.DES_tile.mass_maps import tiled_Map

params.load('../run/base_params.dat')

def my_show(X,**kwargs):
    N1,N2 = X.shape
    pylab.figure()
    pylab.imshow(X.T,
                 origin='lower',
                 interpolation='nearest',
                 cmap=pylab.cm.binary,
                 extent=[0,N1,0,N2])
    pylab.colorbar()
    pylab.xlim(0,N1)
    pylab.ylim(0,N2)

def plot_line_hist(vals,fmt=None,label=None,cumulative=False,
                   *args,**kwargs):
    hist,bins = numpy.histogram(vals,*args,**kwargs)

    if cumulative:
        hist = numpy.cumsum(hist)

    h = numpy.zeros(2*len(hist))
    b = numpy.zeros(2*len(hist))

    h[::2] = hist
    h[1::2] = hist

    b[::2] = bins[:-1]
    b[1::2] = bins[1:]

    if fmt is None:
        return pylab.plot(b,h,label=label)
    else:
        return pylab.plot(b,h,fmt,label=label)
    #return pylab.plot(bins[1:],hist,label=label)

def plot_EB_ratio_hist(valsE,valsB,
                       fmt=None,label=None,cumulative=False,
                       *args,**kwargs):
    histE,binsE = numpy.histogram(valsE,*args,**kwargs)
    histB,binsB = numpy.histogram(valsB,*args,**kwargs)

    if cumulative:
        histE = numpy.cumsum(histE)
        histB = numpy.cumsum(histB)

    hist = histE*1./histB

    h = numpy.zeros(2*len(hist))
    h[::2] = hist
    h[1::2] = hist

    b = numpy.zeros(2*len(histE))
    b[::2] = binsE[:-1]
    b[1::2] = binsE[1:]

    if fmt is None:
        return pylab.plot(b,h,label=label)
    else:
        return pylab.plot(b,h,fmt,label=label)
    

if __name__ == '__main__':
    gamma,kappa,Ngal = get_from_parsed( 'gamma','kappa','Ngal',
                                        RAlim=(10,15),
                                        DEClim=(35,40) )

    r = 5.6 #arcmin

    dtheta = params.dtheta
    N1,N2 = gamma.shape

    slow = False
    if slow:
        pos_x = dtheta * numpy.arange(N1)
        pos_y = dtheta * numpy.arange(N2)
        pos = pos_x[:,None] + 1j*pos_y[None,:]
        Map_E, Map_B = Map_map(gamma,pos,pos,r)
    else:
        Map_E, Map_B = tiled_Map(gamma,dtheta,r)

    peaks_E = find_peaks(Map_E,r=r/dtheta)
    peaks_B = find_peaks(Map_B,r=r/dtheta)
    peaks_k = find_peaks(kappa,r=r/dtheta)

    xE,yE = numpy.where(peaks_E>0)
    xB,yB = numpy.where(peaks_B>0)
    xk,yk = numpy.where(peaks_k>0)

    my_show(kappa)
    pylab.title('kappa')
    pylab.plot(xk+0.5,yk+0.5,'rx')
    
    my_show(Map_E)
    pylab.title('Map (E)')
    pylab.plot(xE+0.5,yE+0.5,'rx')
    
    my_show(Map_B)
    pylab.title('Map (B)')
    pylab.clim(numpy.min(Map_E),numpy.max(Map_E))
    pylab.plot(xB+0.5,yB+0.5,'rx')

    valsE = Map_E[xE,yE]
    valsB = Map_B[xB,yB]
    valsk = kappa[xk,yk]

    rmin = min( min(valsE),min(valsB) )
    rmax = max( max(valsE),max(valsB) )

    pylab.figure()
    plot_line_hist(valsE,bins=30,
                   range=(rmin,rmax) )
    plot_line_hist(valsB,bins=30,
                   range=(rmin,rmax) )
    pylab.legend(['E peaks','B peaks'])

    pylab.figure()
    plot_line_hist(valsk,bins=30,
                   range=(min(valsE),max(valsE)) )

    pylab.show()
    
