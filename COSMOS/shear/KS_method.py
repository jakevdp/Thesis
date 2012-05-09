import numpy
from scipy import fftpack

@numpy.vectorize
def KS_kernel(ell):
    if abs(ell)==0:
        return 0.+0.j
    else:
        return ell*1./numpy.conj(ell)

def gamma_to_kappa(shear,dt1,dt2=None):
    """
    simple application of Kaiser-Squires (1995) kernel in fourier
    space to convert complex shear to complex convergence: imaginary
    part of convergence is B-mode.
    """
    if not dt2:
        dt2 = dt1
    N1,N2 = shear.shape

    #convert angles from arcminutes to radians
    dt1 = dt1 * numpy.pi / 180. / 60.
    dt2 = dt2 * numpy.pi / 180. / 60.

    #compute k values corresponding to field size
    dk1 = numpy.pi / N1 / dt1
    dk2 = numpy.pi / N2 / dt2

    k1 = fftpack.ifftshift( dk1 * (numpy.arange(2*N1)-N1) )
    k2 = fftpack.ifftshift( dk2 * (numpy.arange(2*N2)-N2) )

    ipart,rpart = numpy.meshgrid(k2,k1)
    k = rpart + 1j*ipart

    #compute (inverse) Kaiser-Squires kernel on this grid
    fourier_map = numpy.conj( KS_kernel(-k) )
    
    #compute Fourier transform of the shear
    gamma_fft = fftpack.fft2( shear, (2*N1,2*N2) )

    kappa_fft = fourier_map * gamma_fft

    kappa = fftpack.ifft2(kappa_fft)[:N1,:N2]

    return kappa
