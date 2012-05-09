"""
code for constructing the covariance matrix for a square field,
and performing the eigenvalue decomposition of this matrix
"""

import numpy
from time import time
from scipy.special import j0, j1
from scipy import integrate
from scipy.ndimage import filters
from scipy import interpolate

from shear_correlation import *
from .cosmology import Cosmology

def compute_correlation_matrix(xi,
                               Npix,dpix,
                               ngal,
                               sigma = 0.3,
                               N_evals = 1000,
                               whiten=True):
    """
    compute and return the correlation matrix of shear measurements
    parameters:
      xi   : the shear correlation function xi_plus
      Npix : number of pixels per side for square field
      dpix : width of pixels in arcmin
      ngal : avg galaxy count per pixel
      sigma: shear dispersion
      N_evals: number of evaluations of the correlation function.
               the rest will be interpolated on a b-spline.
      whiten: True|False - whiten the resulting covariance matrix
    """
    D_min = 0
    D_max = numpy.sqrt( (dpix*(Npix+1))**2 + (dpix*(Npix+1))**2 )

    if N_evals:
        t0 = time()
        print "sampling correlation funtion at %i points" % N_evals
        xi_s = Sample_xi(xi,D_min,D_max,N_evals)
        print " - finished in %.2g sec" % (time()-t0)
    else:
        xi_s = xi

    t0 = time()
    print "monte-carlo integrating all pixels within 20 arcsec"
    N_close = int( numpy.ceil(20./dpix) )
    C_close = numpy.zeros( (N_close,N_close) )

    R = numpy.random.random

    Nint = 500
    
    for ix in range(N_close):
        for iy in range(ix,N_close):
            x1 = dpix*R(Nint) + 1j*dpix*R(Nint)
            x2 = dpix*(R(Nint)+ix) + 1j*dpix*(R(Nint)+iy)
    
            vals = abs( x1[:,None]-x2[None,:] ).reshape(Nint*Nint)
    
            covs = xi_s(vals)
            
            C_close[ix,iy] = numpy.mean( covs )
            C_close[iy,ix] = numpy.mean( covs )
    print " - finished in %.2g sec" % (time()-t0)

    shot_noise = sigma**2/ngal

    t0 = time()
    print "constructing correlation matrix:"
    C = numpy.zeros( (Npix*Npix,
                      Npix*Npix) )

    #for efficiency, cycle through distances, not pixels
    # each (dx,dy) distance corresponds to up to eight orientations

    for di in range(Npix):
        for dj in range(di,Npix):
            if di==0 and dj==0:
                c = C_close[0,0] + shot_noise
            elif di<N_close and dj<N_close:
                c = C_close[di,dj]
            else:
                D = numpy.sqrt( (dpix*di)**2 + (dpix*dj)**2 )
                c = xi_s(D)
                
            for i1 in range(di,Npix):
                i2 = i1-di
                for j1 in range(dj,Npix):
                    j2 = j1-dj
                    C[i1*Npix+j1,i2*Npix+j2] = c
                    C[i2*Npix+j2,i1*Npix+j1] = c
                    C[i1*Npix+j2,i2*Npix+j1] = c
                    C[i2*Npix+j1,i1*Npix+j2] = c
                    C[j1*Npix+i1,j2*Npix+i2] = c
                    C[j2*Npix+i2,j1*Npix+i1] = c
                    C[j1*Npix+i2,j2*Npix+i1] = c
                    C[j2*Npix+i1,j1*Npix+i2] = c
                    
    print " - finished in %.2g sec" % (time()-t0)

    if whiten:
        C /= shot_noise

    return C

def compute_correlation_matrix_circ(xi_c,
                                    Npix,dpix,
                                    ngal,
                                    sigma = 0.3,
                                    N_evals = 1000,
                                    whiten=True):
    """
    Compute and return the correlation matrix of shear measurements.
    This uses an approximation to speed up calculation: uses xi_circ
    rather than xi_plus
    
    parameters:
      xi_c : the shear correlation through a circular pixel [ xi_circ ]
      Npix : number of pixels per side for square field
      dpix : width of pixels in arcmin
      ngal : avg galaxy count per pixel
      sigma: shear dispersion
      N_evals: number of evaluations of the correlation function.
               the rest will be interpolated on a b-spline.
      whiten: True|False - whiten the resulting covariance matrix
    """

    D_min = 0
    D_max = numpy.sqrt( (dpix*(Npix+1))**2 + (dpix*(Npix+1))**2 )

    t0 = time()
    print "sampling correlation funtion at %i points" % N_evals
    xi_s = Sample_xi(xi_c,D_min,D_max,N_evals)
    print " - finished in %.2g sec" % (time()-t0)

    shot_noise = sigma**2/ngal

    t0 = time()
    print "constructing correlation matrix:"
    C = numpy.zeros( (Npix*Npix,
                      Npix*Npix) )

    #for efficiency, cycle through distances, not pixels
    # each (dx,dy) distance corresponds to up to eight orientations

    for di in range(Npix):
        for dj in range(di,Npix):
            D = numpy.sqrt( (dpix*di)**2 + (dpix*dj)**2 )
            c = xi_s(D)
            if di==0 and dj==0:
                c += shot_noise
                
            for i1 in range(di,Npix):
                i2 = i1-di
                for j1 in range(dj,Npix):
                    j2 = j1-dj
                    C[i1*Npix+j1,i2*Npix+j2] = c
                    C[i2*Npix+j2,i1*Npix+j1] = c
                    C[i1*Npix+j2,i2*Npix+j1] = c
                    C[i2*Npix+j1,i1*Npix+j2] = c
                    C[j1*Npix+i1,j2*Npix+i2] = c
                    C[j2*Npix+i2,j1*Npix+i1] = c
                    C[j1*Npix+i2,j2*Npix+i1] = c
                    C[j2*Npix+i1,j1*Npix+i2] = c
                    
    print " - finished in %.2g sec" % (time()-t0)

    if whiten:
        C /= shot_noise

    return C


def compute_KL(C):
    """
    compute correlations for an Nx by Ny field
    """

    #----------------------------------------
    t0 = time()
    print "computing eigenvalues using eigh"
    evals,evecs = numpy.linalg.eigh(C)    
    isort = numpy.argsort(evals)[::-1]
    evals = evals[isort]
    evecs = evecs[:,isort]
    print " - finished in %.2g sec" % (time()-t0)
    #----------------------------------------

    return evals,evecs

if __name__ == '__main__':
    from shear_correlation import xi_plus
    import pylab

    Npix = 32
    dpix = 5.
    ngal = 40*dpix*dpix
    sigma = 0.3
    
    n = lambda z,z0=0.5: z**2 * numpy.exp(-(z/z0)**1.5)
    zlim = (0,3)

    xi = xi_plus(n,zlim,Nz=20,Nell=5E4,Or=0)
    C = compute_correlation_matrix(xi,
                                   Npix,dpix,
                                   ngal,
                                   sigma,
                                   whiten = True)

    xi_c = xi_circ(n,zlim,theta_R = dpix/numpy.sqrt(numpy.pi),
                   Nz=20,Nell=5E4,Or=0)
    C_c = compute_correlation_matrix_circ(xi_c,
                                          Npix,dpix,
                                          ngal,
                                          sigma,
                                          whiten = True)
    
    
    evals,evecs = compute_KL(C)
    pylab.figure()
    pylab.semilogy(evals)
    pylab.semilogy(evals-1)
    pylab.semilogy(numpy.ones(len(evals)))

    evals,evecs = compute_KL(C_c)
    pylab.figure()
    pylab.semilogy(evals)
    pylab.semilogy(evals-1)
    pylab.semilogy(numpy.ones(len(evals)))
    

    pylab.figure()
    pylab.imshow(numpy.log10(C),interpolation='nearest')
    pylab.colorbar().set_label('$log_{10}(C_{ij})$')

    pylab.figure()
    pylab.imshow(numpy.log10(C_c),interpolation='nearest')
    pylab.colorbar().set_label('$log_{10}(C_{ij})$')
    pylab.show()
    
