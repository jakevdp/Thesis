"""
This calculates a least-squares fit to a redshift distribution for galaxies
in the cosmos field.

The redshift parametrization is
  n(z) ~ z^a * exp[-(z/z0)^b]
And we fit for z0, a, and b
"""

import numpy as np
from scipy import optimize
import pylab

from read_shear_catalog import read_shear_catalog

def minfunc(x, n, z):
    """
    x is an array [z0, a, b]
    N is number of observed galaxies in bins at centers z
    """
    estimate = z**x[1] * np.exp(- (z / x[0]) ** x[2])
    estimate /= estimate.sum()
    return np.sum((estimate-n)**2)

def fit_zdist(catalog_file, nbins=100, plot=True):
    """
    using a COSMOS catalog, return (z0, a, b) such that the distribution
    n(z) ~ z^a * exp[-(z/z0)^b]
    best describes the observed redshift distribution.

    nbins gives the number of bins at which the function should be evaluated.
    """
    z = read_shear_catalog(catalog_file, ['zphot'], remove_z_problem=True)[0]

    N, bins = np.histogram(z, nbins)

    bincenters = 0.5*( bins[1:] + bins[:-1])
    N = N * 1./N.sum()

    x0 = [0.5, 2.0, 1.5]

    ret = optimize.fmin(minfunc, x0, (N, bincenters))

    z0, a, b = ret

    if plot:
        pylab.plot(bincenters, N)
        zp = bincenters#np.linspace(z.min(), z.max(), 1000)
        nz = zp**a * np.exp(-(zp/z0)**b)
        nz /= np.sum(nz)
        pylab.plot(zp, nz)

    return ret

def plot_redshift_distribution(catalog_file, nbins=100, z0=0.5, a=2.0, b=1.5):
    z = read_shear_catalog(catalog_file, ['zphot'], remove_z_problem=True)[0]

    N, bins = np.histogram(z, nbins)
    print N
    
    zmid = 0.5*( bins[1:] + bins[:-1])
    N = N * 1./N.sum()

    nz = zmid**a * np.exp(-(zmid/z0)**b)

    nz /= nz.sum()

    pylab.subplot(211)
    pylab.plot(zmid, N)
    pylab.subplot(212)
    pylab.plot(zmid, nz)
    
    
if __name__ == '__main__':
    from params import BRIGHT_CAT_NPZ

    #pylab.figure()
    #plot_redshift_distribution(BRIGHT_CAT_NPZ)

    pylab.figure()
    z0, a, b = fit_zdist(BRIGHT_CAT_NPZ)
    print "z0 = %.2f" % z0
    print "a = %.2f" % a
    print "b = %.2f" % b
    pylab.show()
