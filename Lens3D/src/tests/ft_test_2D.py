import numpy
from scipy import fftpack
import pylab

arcmin_to_rad = lambda x: x / 60. * numpy.pi / 180.
rad_to_arcmin = lambda x: x * 180. / numpy.pi * 60.

############################################################
# Kaiser-Squires kernel

def KS_kernel_real(theta):
    """
    theta is a complex 2-D angle theta_x + i*theta_y
       (theta measured in radians)
    """
    theta = numpy.asarray(theta)
    i = numpy.where(abs(theta)==0)
    theta[i] = 1.0
    ret = - 1.0 / ( numpy.pi * theta.conj()**2 )
    ret[i] = 0.0
    theta[i] = 0.0
    return ret

def KS_kernel_fourier(ell):
    """
    ell is a complex 2-D wave number ell_x + i*ell_y
       (ell measured in inverse radians)
    """
    ell = numpy.asarray(ell)
    i = numpy.where(abs(ell)==0)
    ell[i] = 1.0
    ret = ell / ell.conj()
    ret[i] = 0.0
    ell[i] = 0.0
    return ret

############################################################
# 2D gaussian kernel

def gaussian_real(theta, a=arcmin_to_rad(1), theta0=0, ell0=0):
    """
    2D gaussian
    """
    theta0 = complex(theta0)
    ell0 = complex(ell0)
    return numpy.exp(-0.5*(numpy.abs(theta-theta0)*1./a)**2 \
                          + 1j*( theta.real*ell0.real \
                                     + theta.imag*ell0.imag) )

def gaussian_fourier(ell, a=arcmin_to_rad(1), theta0=0, ell0=0):
    """
    Fourier transform of 2D gaussian
    """
    theta0 = complex(theta0)
    ell0 = complex(ell0)
    ell_c = ell-ell0
    return 2*numpy.pi*a*a*numpy.exp(-0.5*(a*numpy.abs(ell_c))**2 \
                                         - 1j*(ell_c.real*theta0.real \
                                                   + ell_c.imag*theta0.imag) )

############################################################

def my_imshow(x,y=None,colorbar=True):
    """

    """
    #*extent*: [ None | scalars (left, right, bottom, top) ]
    #        Data limits for the axes.  The default assigns zero-based row,
    #        column indices to the *x*, *y* centers of the pixels.
    #
    if y is None:
        #no extent information
        pylab.imshow(x.T,origin='lower')
    else:
        extent = (x[0,0].real, x[-1,-1].real,
                  x[0,0].imag, x[-1,-1].imag )
        pylab.imshow(y.T,extent = extent, origin='lower')
        
    if colorbar:
        pylab.colorbar()

def construct_theta_array(theta_min,theta_max,N):
    theta = numpy.linspace(theta_min,theta_max,N)

    theta2D = numpy.zeros((N,N),dtype=complex)
    theta2D += theta.reshape([N,1])
    theta2D += 1j*theta

    return theta2D
    
    
def test_imshow():
    N = 2**6
    theta_min = arcmin_to_rad(-10.)
    theta_max = arcmin_to_rad(10.)

    theta2D = construct_theta_array(theta_min,theta_max,N)
    
    pylab.figure()
    my_imshow(theta2D,theta2D.real)
    pylab.title('real = x')

    pylab.figure()
    my_imshow(theta2D,theta2D.imag)
    pylab.title('imag = y')

    pylab.show()

def show_kernel():
    N = 2**8
    theta_min = arcmin_to_rad(-50)
    theta_max = arcmin_to_rad(50)

    dtheta = (theta_max-theta_min) / N
    dell = 2 * numpy.pi / N / dtheta

    ell = dell * ( numpy.arange(N)-(N/2) )
    
    theta2D = construct_theta_array(theta_min,theta_max,N)
    ell2D = construct_theta_array(ell[0],ell[-1],N)

    KSt = KS_kernel_real(theta2D)
    KSl = KS_kernel_fourier(ell2D)

    print KSt.shape
    print KSl.shape

    pylab.figure()
    my_imshow(theta2D,KSt.real)
    pylab.title(r'$\rm{Re}[KS(\theta)]$')

    pylab.xlabel(r'$\theta_x$')
    pylab.ylabel(r'$\theta_y$')

    pylab.figure()
    my_imshow(theta2D,KSt.imag)
    pylab.title(r'$\rm{Im}[KS(\theta)]$')

    pylab.xlabel(r'$\theta_x$')
    pylab.ylabel(r'$\theta_y$')

    pylab.figure()
    my_imshow(ell2D,KSl.real)
    pylab.title(r'$\rm{Re}[KS^*(\ell)]$')

    pylab.xlabel(r'$\ell_x$')
    pylab.ylabel(r'$\ell_y$')

    pylab.figure()
    my_imshow(ell2D,KSl.imag)
    pylab.title(r'$\rm{Im}[KS^*(\ell)]$')

    pylab.xlabel(r'$\ell_x$')
    pylab.ylabel(r'$\ell_y$')

    pylab.show()

def random_field(theta_min,theta_max,N,loc = 0):
    theta2D = construct_theta_array(theta_min,theta_max,N)
    
    F = numpy.exp(-abs(theta2D-loc)**2)
    return theta2D,F

    #F = numpy.zeros(theta2D.shape,dtype=complex)
    #F += numpy.random.random(theta2D.shape)#-0.5
    #F += 1j*(numpy.random.random(theta2D.shape))#-0.5)
    #return theta2D,F
    
def convolve2D_direct(theta,F):
    """
    perform a direct convolution of F, defined at points theta, 
    with the Kaiser-Squires kernel.
    """
    func = KS_kernel_real
    #func = gaussian_real

    assert theta.shape == F.shape

    output = numpy.empty(F.shape,dtype=complex)
    dtheta = theta[1,1]-theta[0,0]
    dA = dtheta.real * dtheta.imag
    
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            output[i,j] = numpy.sum(F*func(theta[i,j]-theta))

    return dA * output

def convolve2D_fft(theta,F):
    func = KS_kernel_real
    #func = gaussian_real

    assert theta.shape == F.shape

    N1,N2 = theta.shape
    
    dtheta1 = theta[1,1].real - theta[0,0].real
    theta1 = dtheta1*(numpy.arange(2*N1)-N1)
    theta1 = fftpack.ifftshift(theta1)
    
    dtheta2 = theta[1,1].imag - theta[0,0].imag
    theta2 = dtheta2*(numpy.arange(2*N2)-N2)
    theta2 = fftpack.ifftshift(theta2)
    
    theta_kernel = numpy.zeros((2*N1,2*N2),dtype=complex)
    theta_kernel += theta1.reshape((2*N1,1))
    theta_kernel += 1j*theta2

    kernel = func(theta_kernel)
    
    dA = dtheta1 * dtheta2

    F_fft = fftpack.fft2(F, (2*N1,2*N2) ) * dA
    F_fft *= fftpack.fft2(kernel,(2*N1,2*N2) ) 
    
    out = fftpack.ifft2(F_fft)
    
    return out[:N1,:N2]


def convolve2D_analytic(theta,F):
    func = KS_kernel_fourier
    #func = gaussian_fourier

    assert theta.shape == F.shape
    
    dtheta1 = theta[1,1].real - theta[0,0].real
    dtheta2 = theta[1,1].imag - theta[0,0].imag
    
    N1,N2 = theta.shape

    dell1 = numpy.pi / N1 / dtheta1
    dell2 = numpy.pi / N2 / dtheta2
    
    ell1 = fftpack.ifftshift( dell1 * (numpy.arange(2*N1)-N1) )
    ell2 = fftpack.ifftshift( dell2 * (numpy.arange(2*N2)-N2) )
    
    ell = numpy.zeros((2*N1,2*N2),dtype=complex)
    ell += ell1.reshape((2*N1,1))
    ell += 1j * ell2

    F_fft = fftpack.fft2(F,(2*N1,2*N2) )
    F_fft *= func( ell )

    out = fftpack.ifft2(F_fft)

    return out[:N1,:N2]
    

def test_2D_convolve(N=2**5,
                     test_direct = True,
                     test_analytic = True  ):
    numpy.random.seed(0)
    T_plot,F = random_field(-5,5,N)

    T = arcmin_to_rad(T_plot)
          
    if test_direct:
        out1 = convolve2D_direct(T,F)
    out2 = convolve2D_fft(T,F)
    if test_analytic:
        out3 = convolve2D_analytic(T,F)
    
    pylab.figure()
    if test_direct:
        pylab.subplot(221)
        my_imshow(T_plot,out1.real)
        pylab.title('direct (real)')
    
    pylab.subplot(222)
    my_imshow(T_plot,out2.real)
    pylab.title('discrete (real)')

    if test_analytic:
        pylab.subplot(223)
        my_imshow(T_plot,out3.real)
        pylab.title('analytic (real)')
    
    pylab.figure()
    if test_direct:
        pylab.subplot(221)
        my_imshow(T_plot,out1.imag)
        pylab.title('direct (imag)')
    
    pylab.subplot(222)
    my_imshow(T_plot,out2.imag)
    pylab.title('discrete (imag)')

    if test_analytic:
        pylab.subplot(223)
        my_imshow(T_plot,out3.imag)
        pylab.title('analytic (imag)')
    
    if test_direct:
        pylab.figure()
        my_imshow(T_plot,abs(out2-out1))
        pylab.title('discrete-direct: absolute error')
    
    if test_analytic:
        pylab.figure()
        my_imshow(T_plot,abs(out3-out2))
        pylab.title('analytic-discrete: absolute error')

    pylab.show()

if __name__ == '__main__':
    test_2D_convolve()

