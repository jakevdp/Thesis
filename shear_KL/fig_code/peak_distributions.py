import sys
import os
import numpy
from scipy.ndimage import filters
import pylab

sys.path.append(os.path.abspath('../'))
from shear_KL_source import params
from shear_KL_source.Map_peaks import find_peaks

params.load('../run/base_params.dat')


def peak_distribution(filename,
                      r = 1.5*params.dtheta,  #radius of peak finder (arcmin)
                      scale_by_noise = False,
                      correct_border = True,
                      filter = 0, #radius of gaussian filter in arcmin
                      rmin = None,
                      rmax = None,
                      bins = None):
    """
    computes the peak distribution for aperture masses in filename

    returns bins, hist_E, hist_B
    """
    dtheta = params.dtheta

    filter = filter*1./dtheta
    
    print "reading file %s" % filename
    X = numpy.load( filename )

    MapE = X['Map_E']
    MapB = X['Map_B']
    Map_noise = X['Map_noise']

    if correct_border:
        Npix = params.Npix
        MapE = MapE[Npix/2:-Npix/2,Npix/2:-Npix/2]
        MapB = MapB[Npix/2:-Npix/2,Npix/2:-Npix/2]
        Map_noise = Map_noise[Npix/2:-Npix/2,Npix/2:-Npix/2]

    if scale_by_noise:
        MapE = MapE/Map_noise
        MapB = MapB/Map_noise

    if filter:
        MapE = filters.gaussian_filter( MapE,filter )
        MapB = filters.gaussian_filter( MapE,filter )

    peaksE = find_peaks(MapE,r=r/dtheta)
    peaksB = find_peaks(MapB,r=r/dtheta)

    iE = numpy.where(peaksE>0)
    iB = numpy.where(peaksB>0)
    
    valsE = MapE[iE]
    valsB = MapB[iB]
    
    histE,binsE = numpy.histogram(valsE,
                                  bins = bins,
                                  range=(rmin,rmax))
    histB,binsB = numpy.histogram(valsB,
                                  bins = bins,
                                  range=(rmin,rmax))

    return binsE, histE, histB

def convert_to_hist(xdata,ydata,
                    cumulative = False,
                    reverse_cumulative = False):
    """
    converts xdata and ydata to data for a step plot
    """
    if cumulative:
        ydata = numpy.cumsum(ydata)
    elif reverse_cumulative:
        ydata = numpy.cumsum(ydata[::-1])[::-1]
        
            
    y = numpy.zeros(2*len(ydata))
    y[::2] = ydata
    y[1::2] = ydata

    x = numpy.zeros(2*len(ydata))
    x[::2] = xdata[:-1]
    x[1::2] = xdata[1:]

    return x,y

def plot_peak_func(filename,
                   r = 1.5*params.dtheta,  #radius of peak finder (arcmin)
                   scale_by_noise = False,
                   correct_border = True,
                   cumulative = False,
                   filter = 0, #radius of gaussian filter in arcmin
                   rmin = None,
                   rmax = None,
                   bins = None,
                   plot_E = True,
                   plot_B = True,
                   kwargs_E = {},
                   kwargs_B = {}):

    bins, histE, histB = peak_distribution(filename,
                                           r = r,
                                           scale_by_noise = scale_by_noise,
                                           correct_border = correct_border,
                                           filter = filter,
                                           rmin = rmin,
                                           rmax = rmax,
                                           bins = bins)
    
    x,E = convert_to_hist(bins,histE,
                          cumulative = cumulative)
    x,B = convert_to_hist(bins,histB,
                          cumulative = cumulative)

    if plot_E: pylab.plot(x,E,**kwargs_E)
    if plot_B: pylab.plot(x,B,**kwargs_B)

    pylab.xlim(rmin,rmax)

    if scale_by_noise:
        xlabel = r'$M_{ap}/\sigma_M$'
    else:
        xlabel = r'$M_{ap}$'
    
    if cumulative:
        ylabel = '$N(<\ %s)$' % xlabel.strip('$')
    else:
        ylabel = '$N(%s)$' % xlabel.strip('$')
        
    pylab.xlabel( xlabel )
    pylab.ylabel( ylabel )
