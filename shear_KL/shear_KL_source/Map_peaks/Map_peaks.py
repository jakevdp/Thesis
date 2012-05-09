"""
Implementation of the aperture mass peaks from Dietrich & Hartlap 2010
"""
import numpy

def Q_NFW(x,xc):
    r = x*1./xc
    return 1./( 1+ numpy.exp(6-150*x)+numpy.exp(-47+50*x) ) \
           * numpy.tanh(r) / r

def compute_Map(theta,pos,gamma,rmax,xc=0.15):
    """
    theta is an imaginary number giving the center of the filter:
    theta = theta_x + i*theta_y

    pos is an array length N of complex locations of shear estimators
    gamma is an array length N of complex shear values

    rmax is the angular radius of the filter, in the same units as theta
    and pos

    xc is the critical scale: empirically found to be 0.15 by D&H
    """
    sep = pos-theta

    r = abs(sep)

    i0 = numpy.where(r==0)

    x = r/rmax

    cos = numpy.real(sep)/r
    sin = numpy.imag(sep)/r

    cos2 = cos*cos-sin*sin
    sin2 = 2*cos*sin

    gamma_T = - numpy.real(gamma)*cos2 - numpy.imag(gamma)*sin2
    gamma_X = numpy.real(gamma)*sin2 - numpy.imag(gamma)*cos2

    Q = Q_NFW(x,xc)

    Q[i0] = 0.0
    gamma_T[i0] = 0.0
    gamma_X[i0] = 0.0

    norm = numpy.sum(Q)

    Map_E = numpy.sum( Q * gamma_T ) / norm
    Map_B = numpy.sum( Q * gamma_X ) / norm

    return Map_E, Map_B

def make_dense(gamma,dtheta,factor=3):
    N = gamma.shape[0]
    theta = dtheta * 1./factor * numpy.arange(factor*N)
    pos_new = theta[:,None] + 1j*theta[None,:]
    gamma_new = numpy.zeros( (factor*N,factor*N),
                             dtype = complex)
    for i in range(factor):
        for j in range(factor):
            gamma_new[i::factor,j::factor] = gamma
    return gamma_new,pos_new
    

if __name__ == '__main__':
    import sys
    import os
    import pylab
    sys.path.append(os.path.abspath('../distilled_code'))
    from DES_KL_tools import read_shear_out, whiskerplot
    from kappa_diff_maps import kappa_imshow

    gamma,dtheta,kappa = read_shear_out('../shear_fields/shear_out.dat',
                                        return_kappa = True)
    
    N = gamma.shape[0]
    assert N == gamma.shape[1]

    factor = 1

    gamma_new,pos_new = make_dense(gamma,dtheta,factor)
    
    theta = dtheta * numpy.arange(N)
    pos = theta[:,None] + 1j*theta[None,:]

    Map_E = numpy.zeros( (N,N) )
    Map_B = numpy.zeros( (N,N) )

    for i in range(N):
        for j in range(N):
            Map_E[i,j], Map_B[i,j] \
                        = compute_Map(pos[i,j],
                                      pos_new.reshape(pos_new.size),
                                      gamma_new.reshape(gamma_new.size),
                                      rmax = 5.6)
            
    pylab.figure()
    kappa_imshow(kappa,dtheta)

    pylab.figure()
    kappa_imshow(Map_E,dtheta)

    pylab.figure()
    kappa_imshow(Map_B,dtheta)

    pylab.figure()
    whiskerplot(gamma,dtheta,dtheta)

    pylab.show()
