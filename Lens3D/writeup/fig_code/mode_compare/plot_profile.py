import numpy
import pylab
import sys

def get_sortindex(aitken,svd):
    aitken_copy = numpy.copy(aitken)
    aitken_copy.sort()
    dx = aitken_copy[1:]-aitken_copy[:-1]
    dx.sort()
    dx = dx[numpy.where(dx>0)]
    dx = dx[0]

    M = numpy.max(svd)
    sort_arr = numpy.asarray(aitken) - 0.9*dx/M * numpy.asarray(svd)
    return numpy.argsort(sort_arr)

def main(args):
    use = None
    alpha = None
    normalize = False
    scale_trans = True
    logplot = False
    filename = None
    i = 1
    while i<len(args):
        if args[i] == '-r':
            use = 'rows'
        elif args[i] == '-c':
            use = 'cols'
        elif args[i] == '-a':
            alpha = float(args[i+1])
            i+=1
        elif args[i] == '-n':
            normalize = True
        elif args[i] == '-s':
            scale_trans = True
        elif args[i] == '-l':
            logplot = True
        elif args[i] == '-f':
            filename = args[i+1]
            i+=1
        else:
            raise ValueError, "unrecognized argument: %s" % args[i]
        i+=1

    if use is None or alpha is None:
        raise ValueError, "usage: plot_profile [-r/-c] -a <alpha> [-n]"

    aitken = numpy.loadtxt('%s_aitken.txt' % use)

    rad = numpy.loadtxt('%s_rad_%.2g.txt' % (use,alpha) )
    trans = numpy.loadtxt('%s_trans_%.2g.txt' % (use,alpha) )
    svd = numpy.loadtxt('%s_svd_%.2g.txt' % (use,alpha) )

    i = get_sortindex(aitken,svd)
    rad = rad[i]
    trans = trans[i]
    svd = svd[i]
    aitken = aitken[i]

    if normalize:
        rad /= aitken
        trans /= aitken
        svd /= aitken
        aitken /= aitken

    if scale_trans:
        print "scaling transverse WF"
        scale_factor = 10**numpy.floor(numpy.log10(rad[0]/trans[0]))
        trans *= scale_factor

    if logplot:
        plotfunc = pylab.semilogy
    else:
        plotfunc = pylab.plot

    plotfunc(aitken,'--k')
    plotfunc(rad,label=r'$\rm{radial\ WF}$')
    if scale_trans:
        label = r'$\rm{transverse\ WF\ }[\times 10^{%i}]$' % numpy.log10(scale_factor)
    else:
        label = r'$\rm{transverse\ WF}$'
    plotfunc(trans,label=label)
    plotfunc(svd,label=r'$\rm{SVD}$')
    
    pylab.legend(loc=0)

    if normalize and logplot:
        ylim = pylab.ylim()
        pylab.ylim(1E-10,1E1)

    pylab.xlabel('Mode number')
    if normalize:
        pylab.ylabel('Strength relative to unfiltered')
    else:
        pylab.ylabel('Strength')

    pylab.title(r'$\alpha = %.2g,\ v_{\rm{cut}} = %.2g$' % (alpha,alpha) )

    if filename is not None:
        pylab.savefig(filename)
    
    pylab.show()
    
if __name__ == '__main__':
    main(sys.argv)
