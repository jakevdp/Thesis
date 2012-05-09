import os
import sys
import pylab
from matplotlib.ticker import MultipleLocator, NullFormatter, FuncFormatter, LogFormatter

import numpy as np

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

bins, histE, histB = peak_distribution(filename('900_n_a0.15y'),
                                       scale_by_noise=False,
                                       rmin=Mmin,
                                       rmax=Mmax,
                                       bins = bins)

bins1, histE1, histB1 = peak_distribution(filename('noisy_n'),
                                       scale_by_noise=False,
                                       rmin=Mmin,
                                       rmax=Mmax,
                                       bins = bins)

bins2, histE2, histB2 = peak_distribution(filename('perfect_n'),
                                          scale_by_noise=False,
                                          rmin=Mmin,
                                          rmax=Mmax,
                                          bins = bins)

pylab.axes(yscale='log')
pylab.plot(bins1[:-1], histE2, c='#000000', ls='steps')

#pylab.plot(bins[:-1], histE, c='#FF0000', ls='steps')
#pylab.plot(bins[:-1], histB, c='#FFAAAA', ls='steps')

x = bins[:-1] - 0.7 * (bins[1] - bins[0])
y = histE - histB
dy_up = histB + np.sqrt(histE)
dy_down = [min(yi - 0.1, dyi) for (yi,dyi) in zip(y,dy_up)]

xlim = pylab.xlim()

pylab.errorbar(x, y, (dy_down, dy_up),
               fmt = '.', c='#FF0000', ecolor='#FFAAAA')
pylab.xlim(xlim)
#pylab.plot(bins1[:-1], histE1, c='#00FF00', ls='steps')
#pylab.plot(bins1[:-1], histB1, c='#AAFFAA', ls='steps')

x = bins1[:-1] - 0.3 * (bins1[1] - bins1[0])
y = histE1 - histB1
dy_up = histB1 + np.sqrt(histE1)
dy_down = [min(yi - 0.1, dyi) for (yi,dyi) in zip(y,dy_up)]

pylab.errorbar(x, y, (dy_down, dy_up),
               fmt='.', c='#00AA00', ecolor='#00AA00')
pylab.show()
