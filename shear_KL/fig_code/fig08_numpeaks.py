import os
import sys
import pylab
from matplotlib.ticker import MultipleLocator, NullFormatter, FuncFormatter, LogFormatter

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

if 1:
    plot_peak_func(filename('noisy_n'),
                   scale_by_noise = False,
                   rmin=Mmin,
                   rmax=Mmax,
                   bins = bins,
                   plot_B = False,
                   kwargs_E = dict(color='k',
                                   ls = ':',
                                   label='noisy (no KL)'),
                   kwargs_B = dict(color='b',ls=':') )
    
    plot_peak_func(filename('900_n_a0.15y'),
                   scale_by_noise = False,
                   rmin=Mmin,
                   rmax=Mmax,
                   bins = bins,
                   plot_B = False,
                   kwargs_E = dict(color='k',
                                   ls = '--',
                                   label='noisy (with KL)'),
                   kwargs_B = dict(color='r',ls=':') )
    
    plot_peak_func(filename('perfect_n'),
                   scale_by_noise = False,
                   rmin=Mmin,
                   rmax=Mmax,
                   bins = bins,
                   plot_B = False,
                   kwargs_E = dict(color='k',
                                   ls = '-',
                                   label='noiseless') )

    pylab.title('Unmasked Peak Distributions')
    pylab.legend()

elif 1:
    ax1 = pylab.axes((0.125,0.31,0.8,0.58),yscale='log')
    ax1.xaxis.set_major_formatter(NullFormatter())
    
    ax2 = pylab.axes((0.125,0.1,0.8,0.18),yscale='log')
                     
    
    #--------------------------------------------------
    b,E,B = peak_distribution(filename('perfect_n'),
                              scale_by_noise = False,
                              rmin=Mmin,
                              rmax=Mmax,
                              bins = bins)
    b,E_true = convert_to_hist(b,E)

    #--------------------------------------------------
    b,E,B = peak_distribution(filename('noisy_n'),
                                 scale_by_noise = False,
                                 rmin=Mmin,
                                 rmax=Mmax,
                                 bins = bins)
    b,E = convert_to_hist(b,E)

    Ediff1 = E-E_true
    Ediff1[numpy.where(Ediff1<1)] = 0.1
    
    Ediff2 = E_true-E
    Ediff2[numpy.where(Ediff2<1)] = 0.1
    
    ax1.plot(b, Ediff1,
             color='b',
             ls='-',
             label='noisy (no KL)')
    ax2.plot(b, Ediff2,
             color='b',
             ls='-')

    #--------------------------------------------------
    b,E,B = peak_distribution(filename('900_n_a0.15y'),
                              scale_by_noise = False,
                              rmin=Mmin,
                              rmax=Mmax,
                              bins = bins)
    b,E = convert_to_hist(b,E)

    Ediff1 = E-E_true
    Ediff1[numpy.where(Ediff1<1)] = 0.1
    
    Ediff2 = E_true-E
    Ediff2[numpy.where(Ediff2<1)] = 0.1
    
    ax1.plot(b, Ediff1,
             color='r',
             ls='-',
             label='noisy (with KL)')
    ax2.plot(b, Ediff2,
             color='r',
             ls='-')
    
    ax2.set_xlabel( r'$\mathdefault{M_{ap}}$' )
    ax1.set_ylabel( r'$\mathdefault{N(M_{ap})}$' )

    ax1.set_xlim(0.011,0.04)
    ax2.set_xlim(0.011,0.04)

    ax1.set_ylim(1,1E4)
    ax2.set_ylim(10,1)

    ax2.yaxis.set_major_formatter( FuncFormatter(lambda x,*args: r'$\mathdefault{-10^%i}$' % numpy.log10(x) ) )
    
    #ax1.set_title('Unmasked Peak Distributions')
    ax1.legend()

else:
    ax = pylab.subplot(111,yscale='log')

    cumulative = False
    rev = True

    Mmax = 0.038
    
    #--------------------------------------------------
    b,E,B = peak_distribution(filename('perfect_n'),
                              scale_by_noise = False,
                              rmin=Mmin,
                              rmax=Mmax,
                              bins = bins)
    b,E_true = convert_to_hist(b,E,
                               cumulative=cumulative,
                               reverse_cumulative = rev)

    #--------------------------------------------------
    b,E,B = peak_distribution(filename('noisy_n'),
                                 scale_by_noise = False,
                                 rmin=Mmin,
                                 rmax=Mmax,
                                 bins = bins)
    b,E = convert_to_hist(b,E,
                          cumulative=cumulative,
                          reverse_cumulative = rev)

    Ediff = E*1./E_true
    
    ax.plot(b, Ediff,
            color='b',
            ls='-',
            label='without KL')

    #--------------------------------------------------
    b,E,B = peak_distribution(filename('900_n_a0.15y'),
                              scale_by_noise = False,
                              rmin=Mmin,
                              rmax=Mmax,
                              bins = bins)
    b,E = convert_to_hist(b,E,
                          cumulative=cumulative,
                          reverse_cumulative = rev)

    Ediff = E*1./E_true
    
    ax.plot(b, Ediff,
            color='r',
            ls='-',
            label='with KL')
    
    ax.set_xlabel( r'$M_{ap}$' )
    ax.set_ylabel( r'$N/N_{true}$' )

    #--------------------------------------------------
    xlim = ax.get_xlim()
    ax.plot(xlim,[1,1],':k')
    ax.set_xlim(xlim)

    ax.legend()
    ax.set_title('hello')
    

pylab.savefig('figs/fig08_num_peaks.pdf')
pylab.savefig('figs/fig08_num_peaks.eps')

pylab.show()
