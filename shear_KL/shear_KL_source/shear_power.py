"""
Uses equations 1-3 of Hu 1999 to construct shear power spectra
This uses the halofit model from Smith 2003
"""
import numpy
from scipy import integrate

from .cosmology import Cosmology, PowerSpectrum

def gi(z,ni,zlim,Nz,cosmo):
    """
    this differs from eq 3 of Hu 1999 by a factor of D_A out front
    """
    if cosmo.Ok==0:
        Dmz = cosmo.Dm(z)
        integrand = lambda zp: ni(zp)*(1-Dmz/cosmo.Dm(zp))
    else:
        integrand = lambda zp: ni(zp)*cosmo.Da12(z,zp)/cosmo.Da(zp)
    I = integrate.quad(integrand,z,zlim[1])
    return I[0]

def Pspec(ell,dist_functions,
          zlim = (0,10),
          Nz = 100,
          cosmo=None,
          **kwargs):
    """
    dist_functions is a list of N galaxy redshift distributions.
    zlim is a tuple giving the nonzero range of the functions n1 and n2,
    for efficiency.
    Nz is the number of redshift bins to use

    returns an NxN list of (cross) power spectra.  Note that duplicate
    spectra are only computed once.
    """
    if cosmo is None:
        cosmo = Cosmology(**kwargs)

    ell = numpy.asarray(ell)
    if zlim[0]==0:
        z = numpy.linspace(zlim[0],zlim[1],Nz+1)[1:]
    else:
        z = numpy.linspace(zlim[0],zlim[1],Nz)
    zlim = (z[0],z[-1])

    cosmo.sample('Dm',numpy.linspace(0,zlim[1],1000))

    #compute g_i for each galaxy distribution
    g_arrays = [numpy.asarray([gi(zi,n,zlim,Nz,cosmo) for zi in z]) \
                /integrate.quad(n,zlim[0],zlim[1])[0] \
                for n in dist_functions]

    #create wavenumbers at each redshift bin
    DA = cosmo.Dm(z)
    D =  cosmo.Dc(z)
    k = numpy.zeros( (Nz,len(ell)) )
    k += ell[None,:]
    k /= DA[:,None]
    k *= cosmo.h
            
    #compute differential comoving distance
    # (use trapezoidal integration)
    Ddiff_i = numpy.diff(D)
    Ddiff = numpy.zeros(len(D))
    Ddiff[:-1] += Ddiff_i
    Ddiff[1:] += Ddiff_i
    Ddiff *= 0.5

    #compute power spectrum at each redshift bin
    pspecs = numpy.zeros(k.shape)
    PSpec = PowerSpectrum(z[0],
                          om_m0 = cosmo.Om,
                          om_v0 = cosmo.Ol,
                          sig8 = cosmo.sigma8)
    for i in range(Nz):
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
            P12 *= 2*numpy.pi**2
            
            ret[i1][i2] = P12
            ret[i2][i1] = P12

    return ret
