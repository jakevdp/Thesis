"""
A module to plot the diagnostic plots for the COSMOS KL.
The following functions are implemented:

plot_eigenmodes() : plot the shear KL eigenmodes based on the survey geometry

plot_eigenvalues() : plot the expected KL 'power spectrum'

plot_bandpower() : plot the correspondence between KL modes and fourier modes

plot_coeff_hist() : plot the histogram of scaled KL coefficients.  This
    should be gausssian for a gaussian random field

plot_pseudo_spectrum() : plot the KL pseudo-power-spectrum
"""

import sys
import os

try:
    from itertools import product as iter_product
except ImportError: #not available in python version < 2.6
    from my_itertools import iter_product

import numpy as np

import pylab
from matplotlib import ticker

sys.path.append(os.path.abspath('../../'))
from shear_KL_source.cosmology import Cosmology
from shear_KL_source.zdist import zdist_from_file, zdist_parametrized
from shear_KL_source.tools.calcpow import calcpow

from tools import get_gamma_vector, shear_correlation_matrix, zdist

#------------------------------------------------------------
# utility routines for the plots
#------------------------------------------------------------

def get_subplots(rect, Nx, Ny, sepx=0, sepy=0):
    """
    rect: (left, bottom, width, height) for collection of plots
    Nx: number of plots in x-direction
    Ny: number of plots in y-direction
    sepx: separation between plots in x-direction
    sepy: separation between plots in y-direction
    """
    assert len(rect) == 4
    
    #calculate width and height of plots
    dx = (rect[2] - (Nx - 1) * sepx) * 1. / Nx
    dy = (rect[3] - (Ny - 1) * sepy) * 1. / Ny

    xvals = [rect[0] + i * (dx + sepx) for i in range(Nx)]
    yvals = [rect[1] + i * (dy + sepy) for i in range(Ny)]

    return [pylab.axes((x, y, dx, dy)) for y in yvals[::-1] for x in xvals]


def plot_hist_norm(x, **kwargs):
    """
    given a list of data `x`, plot a histogram and the best-fit normal dist.
    """
    mu = x.mean()
    sig2 = np.mean(x * x) - mu * mu

    if 'color' not in kwargs:
        kwargs['color'] = 'k'
    if 'fill' not in kwargs:
        kwargs['fill'] = False
    if 'histtype' not in kwargs:
        kwargs['histtype'] = 'step'

    N, bins, line = pylab.hist(x, **kwargs)

    t = np.linspace(pylab.xlim()[0], pylab.xlim()[1], 100)
    N = np.exp( - 0.5 * (t - mu) ** 2 / sig2 ) / np.sqrt(2 * np.pi * sig2)
    N *= len(x) * (bins[1] - bins[0])

    pylab.plot(t, N, '-k')

#------------------------------------------------------------
# parse data from catalog and compute eigen-decomposition
#------------------------------------------------------------

def compute_all(catalog_file,
                dtheta,
                n_z,
                sigma=0.39,
                RAmin = None, DECmin = None,
                NRA = None, NDEC = None,
                remove_z_problem=True,
                N_bootstraps=1000,
                cosmo = None,
                **kwargs):
    """
    parse catalog data and compute eigenvalue decomposition
    
    Parameters
    ----------
    catalog_file : string or list of strings
        location of the COSMOS catalog(s) to use
    dtheta : float
        size of (square) pixels in arcmin
    sigma : float (default = 0.39)
        intrinsic ellipticity of galaxies
    cosmo : Cosmology object
        the fiducial cosmology used for determination of KL vectors
        if unspecified, it will be initialized from remaining kwargs

    Other Parameters
    ----------------
    n_z : `shear_KL_source.zdist.zdist` object
        n_z(z) returns the galaxy distribution
            
    If the following are unspecified, they will be determined from the data
    RAmin/DECmin : float
        minimum of RA/DEC bins (degrees).
    NRA/NDEC : int
        number of RA/DEC bins

    Returns
    -------
    evals, evecs, a_fit, shape
    """
    gamma, Ngal, noise, RArange, DECrange = get_gamma_vector(
        catalog_file, dtheta, RAmin, DECmin, NRA, NDEC,
        N_bootstraps=N_bootstraps, remove_z_problem=False)
        
    if cosmo is None:
        cosmo = Cosmology(**kwargs)

    NRA, NDEC = gamma.shape

    # use results of bootstrap resampling to compute
    # the noise on observed shear
    gamma = gamma.reshape(gamma.size)
    Ngal = Ngal.reshape(Ngal.size)
    noise = noise.reshape(noise.size)

    i_sigma = np.where(Ngal > 5)
    sigma_estimate = np.sqrt(np.mean(noise[i_sigma] * Ngal[i_sigma]))
    print "average sigma: %.2g" % sigma_estimate

    izero = np.where(Ngal <= 5)
    noise[izero] = sigma_estimate ** 2
    N_m_onehalf = noise ** -0.5

    # noise = sigma^2 / Ngal
    # correlation matrix takes a constant sigma and a variable
    # ngal. So we'll encode the noise as an "effective Ngal"
    # using the user-defined sigma.
    Ngal_eff = sigma**2 / noise
    Ngal_eff[np.where(Ngal == 0)] = 0

    # construct fiducial correlation matrix
    print ">> fiducial correlation matrix"
    R_fid = shear_correlation_matrix(sigma, RArange, DECrange, Ngal_eff,
                                     n_z,
                                     whiten=True,
                                     cosmo=cosmo)

    evals, evecs = np.linalg.eigh(R_fid)
    isort = np.argsort(evals)[::-1]
    evals = evals[isort]
    evecs = evecs[:,isort]

    a = np.dot(evecs.T, N_m_onehalf * gamma)

    return (evals, evecs, a, RArange, DECrange,
            Ngal, noise, N_bootstraps, dtheta)
                     
#------------------------------------------------------------
# plotting routines
#------------------------------------------------------------
def plot_bootstrap_results(Ngal, d2gamma, N_bootstraps, dtheta):
    Ngal = Ngal.reshape(Ngal.size)
    d2gamma = d2gamma.reshape(d2gamma.size)

    sigma2 = Ngal * d2gamma
    sigma2 = sigma2[np.where(Ngal > 5)]
    sigma_estimate = np.sqrt(np.mean(sigma2))

    #plot noise versus number of galaxies
    pylab.subplot(211, yscale='log')

    Ngal_fit = np.linspace(2, max(Ngal), 100)
    d2gamma_fit = sigma_estimate**2 / Ngal_fit
    pylab.plot(Ngal, d2gamma, '.k')
    pylab.plot(Ngal_fit, d2gamma_fit,'-b')
    pylab.ylim(0.0001, 0.1)
    
    pylab.xlabel(r'${\rm n_{gal}/\mathrm{pixel}}$')
    pylab.ylabel(r'${\rm \sigma^2_\gamma}$')

    pylab.title("Results for %i bootstrap resamples (%i' pixels)"
                % (N_bootstraps, dtheta))

    #plot sigma_int histogram
    pylab.subplot(212)

    sigma2 = Ngal * d2gamma
    sigma2 = sigma2
    sigma = np.sqrt(sigma2)
    sigma = sigma[Ngal > 5]
    
    mu = sigma.mean()
    s2 = ((sigma - mu)**2).mean()
    s = np.sqrt(s2)
    
    xrange = np.linspace(mu - 5 * s, mu + 5 * s, 100)
    yrange = (2*np.pi*s2)**-0.5 * np.exp( -0.5 * (xrange-mu)**2 / s2 )
    
    pylab.hist(sigma, 50, normed=True, histtype='stepfilled', color='#AAAAFF')
    pylab.plot(xrange, yrange, '-k')
    
    pylab.xlabel(r'${\rm \sigma_{int}}$')
    pylab.ylabel(r'${\rm dN/d\sigma_{int}}$')

    pylab.text(0.05, 0.95, r'$\mathrm{mean\ \sigma_{int}\ =\ %.3f \pm %.3f}$'
               % (mu, s),
               ha='left', va='top',
               transform = pylab.gca().transAxes,
               fontsize=16)
    

    
def plot_eigenmodes(evecs, evals, ax_list, mode_list, RArange, DECrange):
    """
    Plot KL eigenmodes associated with the COSMOS catalog
    """
    assert len(ax_list) == len(mode_list)

    NRA = len(RArange) - 1
    NDEC = len(DECrange) - 1

    for i in range(len(ax_list)):
        ax = ax_list[i]
        mode = mode_list[i]

        pylab.axes(ax)

        evec = evecs[:,i]
        
        pylab.imshow(evec.reshape((NRA, NDEC)).T,
                     origin='lower',
                     interpolation=None,  #'nearest',
                     cmap=pylab.cm.RdGy,
                     extent=(RArange[0], RArange[-1],
                             DECrange[0], DECrange[-1]))
        cmax = np.max(abs(evec))
        pylab.clim(-cmax,cmax)

        pylab.title(r'$\mathrm{mode\ %i}\ (v=%.3f)$' % (i+1, evals[i]))

    return ax_list

def plot_KL_spectrum(evals, ax=None):
    """
    Plot KL eigenvalues associated with the COSMOS catalog
    """
    if ax is None:
        ax = pylab.subplot(111)

    ax.semilogy(evals, '-k', label='signal + noise')
    ax.plot(evals-1, '--k', label='signal')
    ax.plot(0*evals + 1, ':k', label='noise')
    pylab.legend()

    ylim = ax.get_ylim()
    ax.set_xlim(0, len(evals))
    ax.set_ylim(1E-2, ylim[1])
    
    ax.set_xlabel('mode number (%i total)' % len(evals))
    ax.set_ylabel(r'$\lambda_n$')

    return ax

def plot_bandpower(evecs, noise, RArange, DECrange, dtheta,
                   Nell=50, nmodes=None, ax=None):
    """
    Plot fourier power spectrum of each mode
    """
    if ax is None:
        ax = pylab.axes()

    if nmodes is None:
        nmodes = evecs.shape[1]

    NRA = len(RArange) - 1
    NDEC = len(DECrange) - 1

    # KL modes are modes in signal to noise: correct for this
    evecs = noise[:, None] * evecs

    pow = np.zeros( (Nell,nmodes), dtype=float )

    normalize = True

    #if axes are not a power of 2, we need to pad them
    shape = [2 ** int(np.ceil(np.log2(NRA))),
             2 ** int(np.ceil(np.log2(NDEC)))]

    print "Power spectra:"
    for i in range(nmodes):
        if i%100==0:
            print '  %i/%i' % (i, nmodes)
        evec = evecs[:,i].reshape((NRA,NDEC))
        ell,p = calcpow(evec, dtheta, dtheta, Nell, shape=shape)
        
        if normalize:
            p /= np.sum(p)

        pow[:,i] = p

    pylab.imshow(pow,
                 origin='lower',
                 interpolation='nearest',
                 cmap=pylab.cm.binary,
                 extent=[0,nmodes,ell[0],ell[-1]],
                 aspect='auto')
    pylab.xlim(0, nmodes)
    pylab.ylim(ell[0], ell[-1])
    pylab.xlabel('KL mode number') 
    pylab.ylabel(r'$\ell$')


def plot_coeff_hist(evals, a_fit, nmodes=None, ax_re=None, ax_im=None):
    """
    Plot the histogram of KL coefficients

    For a true gaussian random field, the KL coefficients should be drawn
    from a gaussian with std deviation given by the corresponding eigenvalue.
    We'll check if the data is consistent with that
    """
    if ax_re is None:
        ax_re = pylab.axes()
    if ax_im is None:
        ax_im = pylab.axes()

    i = np.where((evals > 1E-8) & (~np.isnan(evals)))
    evals = evals[i]
    a_fit = a_fit[i]
    
    a_prime = a_fit / np.sqrt(evals)

    if nmodes is None:
        nmodes = len(a_prime)
    else:
        a_prime = a_prime[:nmodes]

    pylab.axes(ax_re)
    plot_hist_norm(a_prime.real, bins=30)
    pylab.title('real, n=%i' % nmodes)
    ax_re.set_xlabel(r'$a_i/\lambda_i$')
    ax_re.set_ylabel(r'$N(a_i/\lambda_i)$')
    ax_re.set_xlim(-3, 3)
    
    pylab.axes(ax_im)
    plot_hist_norm(a_prime.imag, bins=30)
    pylab.title('imag, n=%i' % nmodes)
    ax_im.set_xlabel(r'$a_i/\lambda_i$')
    ax_im.set_ylabel(r'$N(a_i/\lambda_i)$')
    ax_im.set_xlim(-3, 3)


def plot_pseudo_spectrum(evals, a_fit, ax=None):
    if ax is None:
        ax = pylab.axes()
    ax.plot(abs(a_fit)**2, '-k')
    ylim = ax.get_ylim()
    ax.set_ylim(1E-2, ylim[1])

    ax.set_xlabel('KL mode number')
    ax.set_ylabel(r'$|\mathrm{a_n}|^2$')

#----------------------------------------------------------------------
# Plot all: this function plots and saves all plots
#----------------------------------------------------------------------

def plot_all(evals, evecs, a_fit, RArange, DECrange,
             Ngal, noise, N_bootstraps, dtheta,
             label = ''):
    file_type = 'eps'

    #------------------------------------------------------------
    # plot bootstrap noise estimate
    #------------------------------------------------------------
    pylab.figure()
    plot_bootstrap_results(Ngal, noise, N_bootstraps, dtheta)
    pylab.savefig('fig/sigma_calc.pdf')
    pylab.savefig('fig/sigma_calc.eps')
        
    #------------------------------------------------------------
    # plot eigenmodes
    #------------------------------------------------------------
    pylab.figure(figsize=(10,10))
    ax_list = get_subplots((0.09, 0.07, 0.86, 0.86),
                           3, 3, 0.03, 0.03)
    mode_list = np.arange(9)

    plot_eigenmodes(evecs, evals, ax_list, mode_list, RArange, DECrange)
    for i in range(9):
        ax = ax_list[i]
        if i in (0,3,6):
            ax.set_ylabel('DEC (deg)')
        else:
            ax.yaxis.set_major_formatter(ticker.NullFormatter())
            
        if i in (6,7,8):
            ax.set_xlabel('RA (deg)')
            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.3))
        else:
            ax.xaxis.set_major_formatter(ticker.NullFormatter())

    pylab.savefig('fig/%s_eigenmodes.%s' % (label, file_type))
            
    #------------------------------------------------------------
    # plot expected KL power spectrum
    #------------------------------------------------------------
    pylab.figure()
    ax = pylab.axes()
    plot_KL_spectrum(evals, ax)

    pylab.savefig('fig/%s_eigenvalues.%s' % (label, file_type))

    #------------------------------------------------------------
    # plot fourier power of each KL mode
    #------------------------------------------------------------
    pylab.figure()
    ax = pylab.axes()
    plot_bandpower(evecs, noise, RArange, DECrange, dtheta, ax=ax)

    pylab.savefig('fig/%s_bandpower.%s' % (label, file_type))

    #------------------------------------------------------------
    # plot histogram of fourier coefficients
    #------------------------------------------------------------
    pylab.figure()
    ax_re = pylab.subplot(211)
    ax_im = pylab.subplot(212)
    plot_coeff_hist(evals, a_fit, ax_re=ax_re, ax_im=ax_im)
    ax_re.set_xlabel('')

    pylab.savefig('fig/%s_coeff_hist.%s' % (label, file_type))

    #------------------------------------------------------------
    # plot KL pseudo power spectrum
    #------------------------------------------------------------
    pylab.figure()
    ax = pylab.axes()
    plot_pseudo_spectrum(evals, a_fit, ax=ax)

    pylab.savefig('fig/%s_pseudo_power.%s' % (label, file_type))


    
if __name__ == '__main__':
    import cPickle
    from params import BRIGHT_CAT_NPZ, FAINT_CAT_NPZ, \
         COMBINED_ZDIST_ZPROB0, BRIGHT_ZDIST_ZPROB0, \
         BRIGHT_ZDIST, COMBINED_ZDIST

    if 0:
        n_z = zdist_from_file(COMBINED_ZDIST)
        catalog = (BRIGHT_CAT_NPZ, FAINT_CAT_NPZ)
        label = 'combined'
    else:
        n_z = zdist_from_file(BRIGHT_ZDIST)
        catalog = BRIGHT_CAT_NPZ
        label = 'bright'

    out_pkl = "diagnostic.pkl"

    if os.path.exists(out_pkl):
        print "reading data from %s" % out_pkl
        tup = cPickle.load(open(out_pkl))
    else:
        tup = compute_all(catalog,
                          dtheta=2.0,
                          n_z=n_z,
                          N_bootstraps=1000,
                          Or=0.0)
        print "writing data to %s" % out_pkl
        cPickle.dump(tup, open(out_pkl, 'w'))

    plot_all(*tup, label=label)
    pylab.show()
    
