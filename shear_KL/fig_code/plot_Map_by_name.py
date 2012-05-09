import sys
import os
import numpy
import pylab
from time import time

from scipy.ndimage import filters

sys.path.append(os.path.abspath('../'))
from shear_KL_source import params
params.load('../run/base_params.dat')

from shear_KL_source.DES_tile.tools import show_kappa
from shear_KL_source.Map_peaks import *

from DES_tile_Map import plot_line_hist,params,plot_EB_ratio_hist

def plot_peak_func_mean(filenames,
                        r=1.5*params.dtheta,
                        label=None,
                        cumulative=False,
                        ratio=False,
                        filter=0,
                        correct_border=True,
                        scale_by_rms=False,
                        scale_by_noise = False,
                        rmin = None,
                        rmax = None,
                        bins = None,
                        color = None,
                        plot_errorbars=False):
    """
    filter gives gaussian filter width in arcmin
    """
    dtheta = params.dtheta

    filter = filter*1./dtheta
    
    if rmin is None:
        rmin = 0
    if rmax is None:
        rmax = 0.04
        if scale_by_noise:
            rmax = 8
        if scale_by_rms:
            rmax = 5
    if bins is None:
        bins = 100

    #containers for mean values
    xvals = 0
    if ratio:
        yvals = 0
        yvals2 = 0
    else:
        yvalsE = 0
        yvalsB = 0
        yvalsE2 = 0
        yvalsB2 = 0

    for F in filenames:
        print "reading file %s" % F
        X = numpy.load( F )
        
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
    
        if scale_by_rms:
            rms = numpy.sqrt(numpy.mean( MapE**2 + MapB**2 ))
            print "  rms =",rms
        
            valsE = valsE/rms
            valsB = valsB/rms

        if ratio:
            histE,binsE = numpy.histogram(valsE,
                                          bins = bins,
                                          range=(rmin,rmax))
            histB,binsB = numpy.histogram(valsB,
                                          bins = bins,
                                          range=(rmin,rmax))

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

            xvals = b
            yvals += h
            yvals2 += h*h
                           
        else:
            histE,binsE = numpy.histogram(valsE,
                                          bins=bins,
                                          range=(rmin,rmax))
            histB,binsB = numpy.histogram(valsB,
                                          bins=bins,
                                          range=(rmin,rmax))

            if cumulative:
                histE = numpy.cumsum(histE)
                histB = numpy.cumsum(histB)

            hE = numpy.zeros(2*len(histE))
            bE = numpy.zeros(2*len(histE))
            hB = numpy.zeros(2*len(histB))
            bB = numpy.zeros(2*len(histB))
            
            hE[::2] = histE
            hE[1::2] = histE
            hB[::2] = histB
            hB[1::2] = histB
            
            bE[::2] = binsE[:-1]
            bE[1::2] = binsE[1:]
            bB[::2] = binsB[:-1]
            bB[1::2] = binsB[1:]

            xvals = bE
            yvalsE += hE
            yvalsB += hB
            yvalsE2 += hE*hE
            yvalsB2 += hB*hB
            
            #return pylab.plot(b,h,color,label=label)

    if color is not None: color += '-'
    N = 1.0*len(filenames)
    if ratio:
        h = yvals/N
        dh = numpy.sqrt(yvals2/N - h**2)
        if plot_errorbars:
            pylab.errorbar(xvals,h,dh,fmt=color)
        else:
            pylab.plot(xvals,h,label=label)
    else:
        hE = yvalsE / N
        hB = yvalsB / N
        dhE = numpy.sqrt( yvalsE2/N - hE**2 )
        dhB = numpy.sqrt( yvalsB2/N - hB**2 )
        if plot_errorbars:
            l = pylab.errorbar(xvals,hE,dhE,fmt=color)
            c = l[0].get_color()
        else:
            l = pylab.plot(xvals,hE,color,label=label)
            c = l[0].get_color()
        if plot_Bmode:
            pylab.plot(xvals,hB,':'+c,label=label)

    pylab.title('Peak Functions')
    
    if scale_by_rms:
        xlabel = r'\nu'
    elif scale_by_noise:
        xlabel = r'M_{ap}/\sigma_M'
    else:
        xlabel = r'M_{ap}'

    pylab.xlabel('$%s$' % xlabel)

    if ratio:
        if cumulative:
            pylab.ylabel('$N_E(<\ %s)/N_B(<\ %s)$' % (xlabel,xlabel))
        else:
            pylab.ylabel('$N_E(%s)/N_B(%s)$' % (xlabel,xlabel))
    else:
        if cumulative:
            pylab.ylabel('$N(<\ %s)$' % xlabel )
        else:
            pylab.ylabel('$N(%s)$' % xlabel )

    if ratio:
        return xvals,(h,dh)
    else:
        return xvals,(hE,dhE),(hB,dhB)
    

def plot_peak_func(filename,
                   r=1.5*params.dtheta,
                   label=None,
                   cumulative=False,
                   ratio=False,
                   filter=0,
                   correct_border=True,
                   scale_by_rms=False,
                   scale_by_noise = False,
                   plot_Bmode = True,
                   rmin = None,
                   rmax = None,
                   bins = None,
                   color = None):
    """
    filter gives gaussian filter width in arcmin
    """
    F = filename
    
    dtheta = params.dtheta

    filter = filter*1./dtheta
    
    print "reading file %s" % F
    X = numpy.load( F )

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

    if rmin is None:
        rmin = 0
    if rmax is None:
        rmax = 0.04
        if scale_by_noise:
            rmax = 8
        if scale_by_rms:
            rmax = 5
    if bins is None:
        bins = 100
    
    if scale_by_rms:
        rms = numpy.sqrt(numpy.mean( MapE**2 + MapB**2 ))
        print "  rms =",rms
        
        valsE = valsE/rms
        valsB = valsB/rms

    if ratio:
        plot_EB_ratio_hist(valsE,valsB,
                           bins = bins,
                           range=(rmin,rmax),
                           label=label,
                           cumulative=cumulative)
                           
    else:
        if color is not None:
            color += '-'
        l, = plot_line_hist(valsE,
                             fmt = color,
                             bins=bins,
                             range=(rmin,rmax),
                             label = label,
                             cumulative = cumulative)
        if plot_Bmode:
            l2, = plot_line_hist(valsB, ':'+l.get_color(),
                                 bins=bins, 
                                 range=(rmin,rmax),
                                 cumulative = cumulative)
        else:
            l2 = None
            
    pylab.title('Peak Functions')
    
    if scale_by_rms:
        xlabel = r'\nu'
    elif scale_by_noise:
        xlabel = r'M_{ap}/\sigma_M'
    else:
        xlabel = r'M_{ap}'

    pylab.xlabel('$%s$' % xlabel)

    if ratio:
        if cumulative:
            pylab.ylabel('$N_E(<\ %s)/N_B(<\ %s)$' % (xlabel,xlabel))
        else:
            pylab.ylabel('$N_E(%s)/N_B(%s)$' % (xlabel,xlabel))
    else:
        if cumulative:
            pylab.ylabel('$N(<\ %s)$' % xlabel )
        else:
            pylab.ylabel('$N(%s)$' % xlabel )

    return l,l2
    

if __name__ == '__main__':
    outdir = params.shear_recons_dir
    L = os.listdir(outdir)

    import optparse

    parser = optparse.OptionParser()
    parser.add_option("-m", "--rmin", dest="rmin",type='float',
                      help="specify minimum x value", metavar="RMIN")
    parser.add_option("-M", "--rmax", dest="rmax",type='float',
                      help="specify maximum x value", metavar="RMAX")
    parser.add_option("-b", "--bins", dest="bins",type='int',
                      help="specify number of bins", metavar="BINS")
    parser.add_option("-x", "--xlim", dest="xlim",
                      help="specify x-axis limits", metavar="XLIM")
    parser.add_option("-y", "--ylim", dest="ylim",
                      help="specify y-axis limits", metavar="YLIM")
    
    parser.add_option("-c", "--cumulative",
                      action="store_true", dest="cumulative", default=False,
                      help="cumulative histogram")
    parser.add_option("-l", "--lin",
                      action="store_false", dest="log", default=True,
                      help="linear y-axis")
    parser.add_option("-n", "--signal_to_noise",
                      action="store_true", dest="SN", default=False,
                      help="plot signal-to-noise")
    parser.add_option("-r", "--ratio",
                      action="store_true", dest="ratio", default=False,
                      help="plot E/B ratio")

    (options, args) = parser.parse_args()
    
    pylab.figure()
    if options.log:
        pylab.subplot(111,yscale='log')

    for arg in args:
        if arg in L and not os.path.isdir(os.path.join(outdir,arg)):
            F = os.path.join(outdir,arg)
        elif arg+'.npz' in L:
            F = os.path.join(outdir,arg+'.npz')
        elif 'Map_'+arg+'.npz':
            F = os.path.join(outdir,'Map_'+arg+'.npz')
        else:
            print "not using",arg
            continue

        plot_peak_func(F,
                       label = arg,
                       cumulative = options.cumulative,
                       ratio = options.ratio,
                       filter = 0,
                       scale_by_rms = False,
                       scale_by_noise = options.SN,
                       rmin = options.rmin,
                       rmax = options.rmax,
                       bins = options.bins)
        
    pylab.legend(loc=0)

    if options.xlim is not None:
        pylab.xlim( map(float,options.xlim.split(',')) )

    if options.ylim is not None:
        pylab.ylim( map(float,options.ylim.split(',')) )

    pylab.show()
    
