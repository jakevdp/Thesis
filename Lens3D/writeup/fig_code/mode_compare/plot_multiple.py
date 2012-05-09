import numpy
import pylab
from matplotlib.ticker import MultipleLocator

from plot_profile import get_sortindex

rowcol = 'rows'

plotfunc = pylab.semilogy

alphas = [0.1,0.001]
v_cuts = alphas

aitken_linestyle = ':k'
svd_linestyle = '-b'
svd_lw = 3
rad_linestyle = '-r'
rad_lw = 2
trans_linestyle = '-g'
trans_lw = 1


aitken = numpy.loadtxt('%s_aitken.txt' % rowcol)
x = 16*numpy.arange(len(aitken))
plotfunc(x,numpy.ones(len(aitken)),aitken_linestyle)

scale_factor = 1E4


for alpha in alphas:
    rad = numpy.loadtxt('%s_rad_%.2g.txt' % (rowcol,alpha) )
    trans = numpy.loadtxt('%s_trans_%.2g.txt' % (rowcol,alpha) )
    svd = numpy.loadtxt('%s_svd_%.2g.txt' % (rowcol,alpha) )
    aitken = numpy.loadtxt('%s_aitken.txt' % rowcol)
    
    i = get_sortindex(aitken,svd)

    rad = rad[i]
    trans = trans[i]
    svd = svd[i]
    aitken = aitken[i]

    rad /= aitken
    trans /= aitken
    svd /= aitken
    aitken /= aitken

    print "scaling transverse WF"
    #scale_factor = 10**numpy.floor(numpy.log10(rad[0]/trans[0]))
    trans *= scale_factor

    if alpha == alphas[0]:
        label_trans = r'$\rm{transverse\ WF\ }[\times 10^{%i}]$' % numpy.log10(scale_factor)
        label_svd = r'$\rm{SVD}$'
        label_rad = r'$\rm{radial\ WF}$'
    else:
        label_trans = None
        label_svd = None
        label_rad = None

    
    plotfunc(x,svd,svd_linestyle,
             label = label_svd,
             lw = svd_lw)
    plotfunc(x,rad,rad_linestyle,
             label = label_rad,
             lw = rad_lw)
    plotfunc(x,trans,trans_linestyle,
             label = label_trans,
             lw = trans_lw)

fontsize=14

pylab.text(9500,1.4E-5,r'$\alpha=0.1$',
           fontdict=dict(color='g',fontsize=fontsize))
pylab.text(31500,1.4E-5,r'$\alpha=0.001$',
           fontdict=dict(color='g',fontsize=fontsize))

pylab.text(25500,2.1E-3,r'$\alpha=0.1$',
           fontdict=dict(color='r',fontsize=fontsize))
pylab.text(44500,2.1E-3,r'$\alpha=0.001$',
           fontdict=dict(color='r',fontsize=fontsize))

pylab.text(4500,3E-7,r'$v_{\rm{cut}}=0.1$',
           fontdict=dict(color='b',fontsize=fontsize))
pylab.text(24000,4.5E-7,r'$v_{\rm{cut}}=0.001$',
           fontdict=dict(color='b',fontsize=fontsize))

pylab.legend(loc=2)
pylab.ylim(1E-7,1E3)
pylab.xlim(0,70000)
pylab.ylabel('Coefficient')
pylab.xlabel('Principal Component')

#plot inset
pylab.axes([0.63,0.63,0.25,0.25])
pylab.plot(x,trans,trans_linestyle)
pylab.xlim(0,3200)
pylab.ylim(0.05,0.07)
pylab.gca().yaxis.set_major_locator(MultipleLocator(0.01))
pylab.gca().xaxis.set_major_locator(MultipleLocator(1000))
pylab.text(100,0.054,'zoom on',
           fontdict=dict(color='g',fontsize=fontsize))
pylab.text(100,0.051,r'$\alpha=0.001$',
           fontdict=dict(color='g',fontsize=fontsize))


pylab.savefig('../mode_compare.eps')
pylab.show()

    
