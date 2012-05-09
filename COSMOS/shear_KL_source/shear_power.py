"""
Uses equations 1-3 of Hu 1999 to construct shear power spectra
This uses the halofit model from Smith 2003
"""
import numpy as np
from scipy import integrate

from .cosmology import Cosmology, PowerSpectrum

def gi(z, ni, Nz, cosmo):
    """
    this differs from eq 3 of Hu 1999 by a factor of D_A out front

    ni is a `zdist` object which encodes the redshift distribution
    """
    if cosmo.Ok==0:
        Dmz = cosmo.Dm(z)
        weight = lambda zp: 1. - Dmz / cosmo.Dm(zp)
    else:
        weight = lambda zp: cosmo.Da12(z,zp) / cosmo.Da(zp)

    return ni.integrate_weighted(weight, z, ni.zlim[1])

def Pspec(ell,
          dist_functions,
          Nz = 100,
          cosmo=None,
          **kwargs):
    """
    dist_functions is a list of N galaxy redshift cumulative distributions.
        These should be `zdist` objects.
    for efficiency.
    Nz is the number of redshift bins to use

    returns an NxN list of (cross) power spectra.  Note that duplicate
    spectra are only computed once.
    """
    if cosmo is None:
        cosmo = Cosmology(**kwargs)

    zlim = (np.min([n.zlim[0] for n in dist_functions]),
            np.max([n.zlim[1] for n in dist_functions]))

    ell = np.asarray(ell)
    if zlim[0]==0:
        z = np.linspace(zlim[0],zlim[1],Nz+1)[1:]
    else:
        z = np.linspace(zlim[0],zlim[1],Nz)
    zlim = (z[0],z[-1])

    cosmo.sample('Dm',np.linspace(0,zlim[1],1000))

    #create wavenumbers at each redshift bin
    DA = cosmo.Dm(z)
    D =  cosmo.Dc(z)
    k = np.zeros( (Nz,len(ell)) )
    k += ell[None,:]
    k /= DA[:,None]
    k *= cosmo.h
            
    #compute differential comoving distance
    # (use trapezoidal integration)
    Ddiff_i = np.diff(D)
    Ddiff = np.zeros(len(D))
    Ddiff[:-1] += Ddiff_i
    Ddiff[1:] += Ddiff_i
    Ddiff *= 0.5

    #compute power spectrum at each redshift bin
    pspecs = np.zeros(k.shape)

    # halofit can crash for small values of z with certain extreme
    # cosmologies (small Om, small sigma8).
    # This loop will protect against that
    PSpec = PowerSpectrum(z[0],
                          om_m0 = cosmo.Om,
                          om_v0 = cosmo.Ol,
                          sig8 = cosmo.sigma8)

    #compute g_i for each galaxy distribution
    g_arrays = [(np.asarray([gi(zi, n, Nz, cosmo) for zi in z])
                 / n.integrate(n.zlim[0], n.zlim[1]))
                for n in dist_functions]
                
    for i in range(Nz):
        z_last = PSpec.z
        PSpec.z = z[i]
            
        # k^4 and other factors come from converting
        #  potential p-spec to density p-spec.  See Bartelmann 2010 eq 142
        pspecs[i] = 2.25 * (1+z[i])**2 \
                    * cosmo.Om**2 * (cosmo.H0/cosmo.c)**4 \
                    * ( PSpec.D2_NL(k[i]) / k[i]**4 )
    pspecs /= DA[:,None]
    pspecs *= Ddiff[:,None]

    N = len(dist_functions)
    ret = [[None for i in range(N)] for i in range(N)]

    for i1 in range(N):
        for i2 in range(i1,N):
            integrand = pspecs.copy()

            gg = g_arrays[i1]*g_arrays[i2]
            
            #multiply integrand following eqn. 1 of Hu99 
            integrand *= gg[:,None]
            
            P12 = integrand.sum(0)
            P12 *= ell
            P12 *= 2*np.pi**2
            
            ret[i1][i2] = P12
            ret[i2][i1] = P12

    return ret
