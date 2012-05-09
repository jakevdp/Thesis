"""
For large redshifts (z >~ 4.5) the halofit code is producing strange results.
In this module, we investigate why.
"""
import numpy as np
import pylab
from shear_KL_source.cosmology import Cosmology, PowerSpectrum

from shear_KL_source.shear_correlation import xi_plus

from shear_KL_source.shear_KL import compute_correlation_matrix

import sys
sys.path.append('shear/param_estimation')
from tools import shear_correlation_matrix
from z_distributions import zdist_from_file

def test_PowerSpectrum(z=1,
                       Om=0.3,
                       s8=0.85):
    cosmo = Cosmology(Om=Om, Ol=1-Om, sigma8=s8, Or=0)

    P = PowerSpectrum(z=z,
                      om_m0 = cosmo.Om,
                      om_v0 = cosmo.Ol,
                      sig8 = cosmo.sigma8)

    ell = np.logspace(-1,8,1000)
    k = ell / cosmo.Dm(z)
    k *= cosmo.h

    pylab.loglog(k, P.D2_NL(k) / k**4)

def test_Correlation_matrix():
    cosmo = Cosmology(Om=0.27, Ol=0.73, sigma8=0.85, Or=0)
    
    from params import BRIGHT_CAT_NPZ, FAINT_CAT_NPZ, COMBINED_ZDIST_ZPROB0
    n_z = zdist_from_file(COMBINED_ZDIST_ZPROB0)
    zlim = n_z.zlim
    
    #n_z = lambda z: np.sin(z * np.pi / 5)
    #zlim = (0,5)
    
    from shear_KL_source.shear_power import Pspec, gi


    Nz = 20
    z = np.linspace(zlim[0],zlim[1],Nz+1)[1:]
    zi = z[0]

    z0 = 0.5

    Dmz = cosmo.Dm(z0)
    integrand = lambda zp: n_z(zp)*(1-Dmz/cosmo.Dm(zp))

    #z = np.linspace(z0, zlim[1], 1000)
    #import pylab
    #pylab.plot(z, integrand(z))
    #pylab.show()
    #exit()
    
    ell = np.logspace(-1,8,5E4)
    P_ell = Pspec(ell, [n_z], zlim, Nz=20, cosmo=cosmo)[0][0]

    #xi = xi_plus(n_z, zlim, cosmo=cosmo)
    #compute_correlation_matrix(xi, 20, 4, 100)

    #RArange = np.arange(21) * 4./60.
    #DECrange = np.arange(21) * 4./60.
    #Ngal = 40 * np.ones((20,20))

    #shear_correlation_matrix(0.3, RArange, DECrange, Ngal,
    #                         n_z, zlim)

if __name__ == '__main__':
    test_Correlation_matrix()

