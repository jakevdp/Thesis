import os
import sys
import pylab
from matplotlib.ticker import MultipleLocator, NullFormatter

sys.path.append(os.path.abspath('../'))
from shear_KL_source import params
params.load('../run/base_params.dat')

from peak_distributions import *

filename = lambda s: os.path.join(params.shear_recons_dir,'Map_'+s+'.npz')

rmin = 3.25
rmax = 6.5
Mmin = 0.01
Mmax = 0.04
bins = 40

pylab.figure(figsize=(6,4.5))
pylab.axes( (0.15,0.125,0.775,0.775) ,yscale='log')

#pylab.figure()
#pylab.subplot(111,yscale='log')

plot_peak_func(filename('noisy_n'),
               scale_by_noise = False,
               rmin = Mmin,
               rmax = Mmax,
               bins = bins,
               plot_E = False,
               kwargs_B = dict(color='r',
                               ls = '-',
                               label='no KL (B-mode)') )
               
plot_peak_func(filename('noiseonly_n'),
               scale_by_noise = False,
               rmin = Mmin,
               rmax = Mmax,
               bins = bins,
               plot_B = False,
               kwargs_E = dict(color='r',
                               ls = ':',
                               label='    (noise peaks)'),
               kwargs_B = dict(color='r',
                               ls = ':',
                               label='    (noise peaks)') )

plot_peak_func(filename('900_n_a0.15y'),
               scale_by_noise = False,
               rmin = Mmin,
               rmax = Mmax,
               bins = bins,
               plot_E = False,
               kwargs_B = dict(color='g',
                               ls = '-',
                               label='with KL (B-mode)') )


plot_peak_func(filename('900_n_a0.15y_no'),
               scale_by_noise = False,
               rmin = Mmin,
               rmax = Mmax,
               bins = bins,
               plot_B = False,
               kwargs_E = dict(color='g',
                               ls = ':',
                               label='    (noise peaks)'),
               kwargs_B = dict(color='g',
                               ls = ':',
                               label='    (noise peaks)') )

if False:
    x,E,B = peak_distribution(filename('noiseonly_n'),
                              scale_by_noise = False,
                              rmin = Mmin,
                              rmax = Mmax,
                              bins = bins)
    x,B = convert_to_hist(x,0.5*(E+B))

    x = numpy.concatenate( (x,x[::-1]) )
    B = numpy.concatenate( ((B+numpy.sqrt(B)),(B-numpy.sqrt(B))[::-1]) )
    B[numpy.where(B<1)]=0.1
    pylab.fill(x,B,color='#AAAAFF')
    
    x,E,B = peak_distribution(filename('900_n_a0.15y_no'),
                              scale_by_noise = False,
                              rmin = Mmin,
                              rmax = Mmax,
                              bins = bins)
    x,B = convert_to_hist(x,0.5*(E+B))

    x = numpy.concatenate( (x,x[::-1]) )
    B = numpy.concatenate( ((B+numpy.sqrt(B)),(B-numpy.sqrt(B))[::-1]) )
    B[numpy.where(B<1)]=0.1
    pylab.fill(x,B,color='#FFAAAA')

    
    pylab.ylim(1,pylab.ylim()[1])

pylab.title('B-modes vs noise-only (unmasked)')

pylab.legend(prop=dict(size=13))

pylab.xlim(0.01,0.03)

pylab.savefig('figs/fig09_B_noise.pdf')
pylab.savefig('figs/fig09_B_noise.eps')
pylab.show()
