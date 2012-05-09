"""
Use shear KL to generate a map from COSMOS shear data
"""

import os
import sys

import numpy as np

from scipy.ndimage import filters

import pylab
from matplotlib.colors import LinearSegmentedColormap

path1 = os.path.abspath(os.path.join(os.path.dirname(__file__), '../param_estimation'))
path2 = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

sys.path.append(path1)
sys.path.append(path2)

from tools import get_gamma_vector
from shear_KL_source.tools import gamma_to_kappa
from shear_KL_source.DES_KL.tools import whiskerplot


def whiskerplot(shear, RArange, DECrange, scale=5, **kwargs):
    RA = 0.5 * (RArange[1:] + RArange[:-1])
    DEC = 0.5 * (DECrange[1:] + DECrange[:-1])

    theta = shear**0.5
    
    pylab.quiver(RA,DEC,
                 theta.real.T,theta.imag.T,
                 pivot = 'middle',
                 headwidth = 0,
                 headlength = 0,
                 headaxislength = 0,
                 scale=scale,
                 **kwargs)
    pylab.xlim(RArange[-1], RArange[0])
    pylab.ylim(DECrange[0], DECrange[-1])
    pylab.xlabel('RA')
    pylab.xlabel('DEC')


def BkBuW(cmax=1.0):
    return LinearSegmentedColormap('BkBuW',
                                   {'red':   [(0.0, 0.0, 0.0),
                                              (0.5, 0.0, 0.0),
                                              (1.0, cmax, cmax)],
                                    
                                    'green': [(0.0, 0.0, 0.0),
                                              (0.5, 0.0, 0.0),
                                              (1.0, cmax, cmax)],
                                    
                                    'blue':  [(0.0, 0.0, 0.0),
                                              (1.0, cmax, cmax),
                                              (1.0, cmax, cmax)]} )

def plot_mask(catalog_file, dtheta,
              RAmin = None, DECmin = None,
              NRA = None, NDEC = None,
              remove_z_problem = True,
              N_bootstraps=0):
    gamma, Ngal, RArange, DECrange = get_gamma_vector(
        catalog_file, dtheta, RAmin, DECmin, NRA, NDEC,
        N_bootstraps=N_bootstraps, remove_z_problem=remove_z_problem)

    print RArange[0], RArange[-1]
    print DECrange[0], DECrange[-1]
    print gamma.shape
    print Ngal.shape

    pylab.imshow(Ngal.T,
                 origin='lower',
                 interpolation='nearest')
    pylab.colorbar()

def plot_kappa_simple(catalog_file, remove_z_problem=False,
                      Npix=128, kappa_filter=4,
                      kappa_cutoff=0):
    """
    kappa_filter is in pixels
    """
    gamma_plot, Ngal, RArange_g, DECrange_g = get_gamma_vector(
        catalog_file, dtheta=2.0, remove_z_problem=remove_z_problem)
    
    # get a square grid for the kappa measurement
    NRA = NDEC = Npix
    
    RAmin = RArange_g[0]
    RAmax = RArange_g[-1]
    DECmin = DECrange_g[0]
    DECmax = DECrange_g[-1]

    dtheta = (RAmax-RAmin) * 1./NRA * 60

    # DEC range is smaller than RA range: make half the
    #  zeros on the bottom
    DECmin -= 0.5 * (DECmin + NDEC * dtheta/60. - DECmax)
    DECmax = DECmin + NDEC * dtheta/60.

    # get the shear realization
    gamma, Ngal, RArange, DECrange = get_gamma_vector(
        catalog_file, dtheta, RAmin, DECmin, NRA, NDEC+1,
        remove_z_problem=remove_z_problem)

    kappa = gamma_to_kappa(gamma.T, dtheta)

    if kappa_filter:
        kappa = filters.gaussian_filter(kappa.real, kappa_filter) \
                + 1j * filters.gaussian_filter(kappa.imag, kappa_filter)

    if kappa_cutoff is not None:
        kappa[np.where(kappa < kappa_cutoff)] = kappa_cutoff

    pylab.figure(figsize=(10,8))
    pylab.imshow(kappa.real.T,
                 origin='lower',
                 cmap=BkBuW(1.0),
                 extent=(RAmin, RAmax, DECmin, DECmax))
    pylab.xlim(pylab.xlim()[::-1])
    pylab.title('COSMOS convergence map: Kaiser-Squires')
    pylab.colorbar().set_label(r'$\kappa\ \mathrm{(E-mode)}$')
    

    #conjugate gamma: we're flipping the x-axis
    whiskerplot(gamma_plot.conj(), RArange_g, DECrange_g, color='w')
    pylab.ylim(DECmin, 2.9)
    
    #pylab.figure()
    #pylab.imshow(kappa.imag.T,
    #             origin='lower',
    #             cmap=BkBuW(0.8),
    #             extent=(RAmin, RAmax, DECmin, DECmax))
    #pylab.xlim(pylab.xlim()[::-1])
    #pylab.title('B-mode convergence')
    #pylab.colorbar()
    #
    #pylab.figure()
    #pylab.imshow(Ngal.T,
    #             origin='lower',
    #             extent=(RAmin, RAmax, DECmin, DECmax))
    #pylab.xlim(pylab.xlim()[::-1])
    #pylab.title('Galaxy Count')
    #pylab.colorbar()

def plot_kappa_SN(catalog_file,
                  N_realizations=1000):
    # compute a 64 x 64 grid which encompasses all the data
    RAmin = 149.4317396
    RAmax = 150.798406267
    DECmin = 1.570097085
    DECmax = 2.90343041833

    NRA = 64
    NDEC = 64

    dtheta = (RAmax-RAmin) * 1./NRA * 60

    DECmin -= 0.5 * (DECmin + NDEC * dtheta/60. - DECmax)
    
    # get the shear realization
    gamma, Ngal, d2gamma, RArange, DECrange = get_gamma_vector(
        catalog_file, dtheta, RAmin, DECmin, NRA, NDEC,
        remove_z_problem=True,
        N_bootstraps=N_realizations)

    kappa = np.zeros(gamma.shape, dtype=complex)
    kappa_2 = np.zeros(gamma.shape, dtype=float)
    
    i_zero = np.where(d2gamma <= 0)

    d2gamma[i_zero] = 1

    dgamma = np.sqrt(d2gamma)

    print "computing kappa %i times" % N_realizations
    for i in range(N_realizations):
        if (i+1)%100 == 0:
            print " >", i+1
        phase = np.exp(2j * np.pi * np.random.random(gamma.shape))
        noise = np.random.normal(0, dgamma)
        noise[i_zero] = 0
        k = gamma_to_kappa(gamma + phase * noise, dtheta)

        kappa += k
        kappa_2 += abs(k) ** 2

    kappa /= N_realizations
    d2kappa = kappa_2 / N_realizations - abs(kappa) ** 2

    pylab.figure()
    pylab.imshow(d2kappa.T, origin='lower', interpolation='nearest',
                 extent=(RAmin, RAmax, DECmin, DECmax))
    pylab.xlim(pylab.xlim()[::-1])
    pylab.colorbar()

    pylab.figure()
    pylab.imshow(Ngal.T, origin='lower', interpolation='nearest',
                 extent=(RAmin, RAmax, DECmin, DECmax))
    pylab.xlim(pylab.xlim()[::-1])

    # compute convergence signal-to-noise
    i_zero = np.where(d2kappa == 0)
    d2kappa[i_zero] = 1
    kappa_SN = kappa / np.sqrt(d2kappa)
    kappa_SN[i_zero] = np.nan
    
    pylab.figure()
    pylab.imshow(kappa_SN[4:-4,4:-4].real.T,
                 origin='lower',
                 extent=(RAmin, RAmax, DECmin, DECmax))
    pylab.xlim(pylab.xlim()[::-1])
    pylab.colorbar()

    pylab.figure()
    pylab.imshow(kappa_SN[4:-4,4:-4].imag.T,
                 origin='lower',
                 extent=(RAmin, RAmax, DECmin, DECmax))
    pylab.xlim(pylab.xlim()[::-1])
    pylab.colorbar()

def plot_shear(catalog_file, dtheta):
    gamma, Ngal, RArange, DECrange = get_gamma_vector(
        catalog_file, dtheta, remove_z_problem=False)
    
    pylab.imshow(Ngal.T, origin='lower',
                 cmap=BkBuW(0.8),
                 extent=(RArange[0], RArange[-1], DECrange[0], DECrange[-1]))
    pylab.xlim(pylab.xlim()[::-1])
    

    whiskerplot(gamma, RArange, DECrange, color='w')
    
    
if __name__ == '__main__':
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
    sys.path.append(path)
    from params import BRIGHT_CAT_NPZ, FAINT_CAT_NPZ

    import numpy as np
    
    #plot_kappa_SN((BRIGHT_CAT_NPZ, FAINT_CAT_NPZ), 1000)
    plot_kappa_simple((BRIGHT_CAT_NPZ, FAINT_CAT_NPZ),
                      Npix=128, kappa_filter=4, kappa_cutoff=-0.003)

    #plot_shear((BRIGHT_CAT_NPZ, FAINT_CAT_NPZ), 3.0)
    
    pylab.show()
