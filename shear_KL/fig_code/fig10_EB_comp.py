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

pylab.figure()

#------------------------------------------------------------
ax = pylab.axes( (0.12,0.41,0.83,0.49))

plot_peak_func(filename('noisy_n'),
               scale_by_noise = True,
               cumulative = True,
               rmin = rmin,
               rmax = rmax,
               bins = bins,
               kwargs_E = dict(ls = '-',
                               c  = 'r',
                               label = 'no KL'),
               kwargs_B = dict(ls = ':',
                               c  = 'r') )

plot_peak_func(filename('900_n_a0.15y'),
               scale_by_noise = True,
               cumulative = True,
               rmin = rmin,
               rmax = rmax,
               bins = bins,
               kwargs_E = dict(ls = '-',
                               c  = 'g',
                               label = 'with KL'),
               kwargs_B = dict(ls = ':',
                               c  = 'g') )

leg = pylab.legend(loc=2)

#dummy lines for E/B legend
l1 = pylab.plot([-1,1],[0,0],'-k')
l2 = pylab.plot([-1,1],[0,0],':k')
pylab.legend([l1,l2],['E-mode','B-mode'],frameon=False,
             loc='upper left',bbox_to_anchor=(0.23,1.0))
ax.add_artist(leg)

ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(0.25))
ax.xaxis.set_major_formatter(NullFormatter())
pylab.xlim(rmin,rmax)
ylim = pylab.ylim()
pylab.ylim(1,1200)


L = ax.lines
xlabel = ax.get_xlabel()

#------------------------------------------------------------
ax = pylab.axes( (0.12,0.1,0.83,0.29))

for i in range(2):
    E = L[2*i].get_ydata()
    B = L[2*i+1].get_ydata()
    x = L[2*i].get_xdata()
    pylab.plot(x,B/E,
               color=L[2*i].get_color())
pylab.xlabel(xlabel)
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(0.25))
ax.yaxis.set_major_locator(MultipleLocator(0.2))
pylab.ylabel(r'$N_B/N_E$')

pylab.ylim(0.0,0.8)
pylab.xlim(rmin,rmax)

pylab.text(0.98,0.9,'B/E ratio',
           fontsize=14,
           transform=ax.transAxes,va='top',ha='right')

#------------------------------------------------------------

pylab.savefig('figs/fig10_EB_comp.pdf')
pylab.savefig('figs/fig10_EB_comp.eps')

pylab.show()
