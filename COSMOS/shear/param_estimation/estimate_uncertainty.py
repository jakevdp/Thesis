"""
Tools to estimate the uncertainty of the COSMOS shear estimates
"""
import numpy as np

import sys
sys.path.append('../')
from read_shear_catalog import read_shear_catalog
from tools import bootstrap_resample

def bootstrap_catalog(catalog_file, N_bootstraps, dtheta,
                      RAmin = None, DECmin = None,
                      NRA = None, NDEC = None):
    """
    Estimate shear uncertainty using a bootstrap resampling of shear
    data in the catalog.

    Parameters
    ----------
    catalog_file : file of COSMOS shear catalog
    N_bootstraps : number of resamplings to use
    dtheta : pixel size in arcmin

    Other Parameters
    ----------------
    If these are unspecified, they will be determined from the data
    RAmin : minimum of RA bins (degrees).
    NRA : number of RA bins
    DECmin : minimum of DEC bins (degrees)
    NDEC : number of DEC bins

    Returns
    -------
    (Ngal, dgamma2)
    Ngal : array[int], shape = (NRA, NDEC)
        number of galaxies in each shear bin
    dgamma2 : array[float], shape = (NRA, NDEC)
        estimated squared shear error in each shear bin
    """
    RA, DEC, gamma1, gamma2 = read_shear_catalog(catalog_file,
                                                 ('Ra', 'Dec',
                                                  'e1iso_rot4_gr_snCal',
                                                  'e2iso_rot4_gr_snCal'),
                                                 None)

    gamma = gamma1 - 1j*gamma2

    if RAmin is None:
        RAmin = RA.min()
    if DECmin is None:
        DECmin = DEC.min()

    if NRA is None:
        NRA = int(np.ceil((RA.max() - RAmin + 1E-8) * 60. / dtheta))
    if NDEC is None:
        NDEC = int(np.ceil((DEC.max() - DECmin + 1E-8) * 60. / dtheta))

    return bootstrap_resample(gamma, RA, DEC,
                              RAmin, dtheta/60., NRA,
                              DECmin, dtheta/60., NDEC,
                              N_bootstraps)


if __name__ == '__main__':
    import pylab
    from params import BRIGHT_CAT_NPZ, FAINT_CAT_NPZ
    from time import time

    np.random.seed(0)

    N_bootstraps = 1000
    pixel_scale = 2

    t0 = time()
    gamma_mean, Ngal, d2gamma, sigma = bootstrap_catalog(
        BRIGHT_CAT_NPZ, N_bootstraps, pixel_scale)
    print "time: %.2g sec" % (time()-t0)

    Ngal = Ngal.reshape(Ngal.size)
    d2gamma = d2gamma.reshape(d2gamma.size)

    pylab.figure()
    #plot noise versus number of galaxies
    pylab.subplot(211, yscale='log')

    x = np.linspace(2, max(Ngal), 100)
    y = sigma**2 / x
    pylab.plot(Ngal, d2gamma, '.k')
    pylab.plot(x,y,'-b')
    pylab.ylim(0.0001, 0.1)
    
    pylab.xlabel(r'${\rm n_{gal}/\mathrm{pixel}}$')
    pylab.ylabel(r'${\rm \sigma^2_\gamma}$')

    pylab.title("Results for %i bootstrap resamples (%i' pixels)"
                % (N_bootstraps, pixel_scale))

    #plot sigma_int histogram
    pylab.subplot(212)

    sigma2 = Ngal * d2gamma
    sigma2 = sigma2[np.where(sigma2 > 0.05)]
    sigma = np.sqrt(sigma2)
    
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

    pylab.savefig('fig/sigma_calc.pdf')
    pylab.savefig('fig/sigma_calc.eps')
    pylab.show()
