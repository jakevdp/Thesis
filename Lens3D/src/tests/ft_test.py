import numpy
import pylab

from scipy import fftpack

"""
Testing out convolution with the convolution theorem,
  mixing discrete and continuous fourier transforms.

Convolution theorem:

given a convolution:
 h(t) = [f*g](t) = integral[dt' f(t')g(t-t') ]

 if F(k) = integral[dt f(t) e^(ikt) ] is the fourier transform of f, then

 H(k) = F(k) * G(k)
"""

########################################
# discrete convolution comparison

def padded_fft(x,pad_type='R'):
    """
    perform a fast fourier transform of x, zero-padding to twice the
    length of x.
    
    pad type is one of 'L','R','C'
    zeros to the left, the right, or even on both sides
    """
    N = len(x)
    xx = numpy.zeros(2*N,dtype = x.dtype)
    if pad_type == 'L':
        xx[N:] = x
    elif pad_type == 'R':
        xx[:N] = x
    elif pad_type == 'C':
        xx[N/2:3*N/2] = x
    else:
        raise ValueError, "pad_type not recognized"
        
    return fftpack.fft(xx,overwrite_x=True)
    

def convolve(x,y):
    assert x.ndim == 1
    assert y.ndim == 1
    xy = numpy.empty( len(x)+len(y) )

    for i in range(len(xy)):
        x_slice = slice( max(0   , i+1-len(x) ),
                         min(i+1 , len(x)     )  )
        y_slice = slice( max(0   , i+1-len(y) ),
                         min(i+1 , len(y)     )  )
        xy[i] = numpy.sum( x[x_slice] * y[y_slice][::-1] )

    return xy

def convolve_fft(x,y):
    x_fft = padded_fft(x,'R')
    y_fft = padded_fft(y,'R')
    return fftpack.ifft( x_fft * y_fft )

########################################
# analytic version of gaussian and its fourier transform

def gaussian_t(t,a,t0,k0):
    """
    a gaussian wave packet of width a, centered at t0, with momentum k0
    """ 
    #norm = (a*numpy.sqrt(numpy.pi))**(-0.5) 
    return numpy.exp(-0.5*((t-t0)*1./a)**2 + 1j*t*k0)

def gaussian_k(k,a,t0,k0):
    """
    fourier transform of gaussian_t, above
    convention is F(k) = integral[ f(t) exp(ikt)]
                  f(t) = 1/2pi integral[F(k) exp(-ikt)]
    in mathematica, this corresponts to FourierParameters->{1,1}
     fourier transform is sqrt(2pi) * [mathematica default]
    """
    #norm = (a*numpy.sqrt(numpy.pi))**(-0.5)
    return a * (2*numpy.pi)**0.5 * numpy.exp(-0.5*(a*(k-k0))**2 - 1j*(k-k0)*t0)

############################################################
# 3 different methods of convolving with a gaussian

def convolve_with_gaussian(x,t,a,t0,k0,method=0):
    """
    convolve x(t) with a gaussian kernel defined by a,t0,k0.
     method == 0: perform a (slow) direct convolution
     method == 1: use the binned discrete fourier transform
     method == 2: use the analytic fourier transform
    """
    dt = t[1]-t[0]
    N = len(t)

    if method == 0: #direct convolution
        x_out = numpy.zeros(N)
        for i in range(N):
            x_out[i] = numpy.sum(x*gaussian_t(t[i]-t,a,t0,k0) )
        return x_out*dt

    elif method == 1: #discrete fourier transform convolution
        t_c = dt * fftpack.ifftshift( ( numpy.arange(2*N) - N ) )
        y = gaussian_t(t_c,a,t0,k0)
        x_fft = fftpack.fft(x,2*N)
        x_fft *= fftpack.fft(y,2*N)
        x_out = fftpack.ifft(x_fft)
        x_out *= dt
        print "discrete leftovers: (%.2g,%.2g,%.2g)" % (x_out[N:].min(),
                                                        x_out[N:].max(),
                                                        abs(x_out[N:]).mean() )
        return x_out[:N]
    
    elif method == 2: #analytic fourier transform convolution
        dk = numpy.pi/N/dt
        k = fftpack.ifftshift( dk * ( numpy.arange(2*N)-N ) )
        
        x_fft = fftpack.fft(x,2*N)
        x_fft *= gaussian_k(k,a,t0,k0)
        x_out = fftpack.ifft(x_fft)
        print "analytic leftovers: (%.2g,%.2g,%.2g)" % (x_out[N:].min(),
                                                        x_out[N:].max(),
                                                        abs(x_out[N:]).mean() )
        return x_out[:N]

############################################################
# main program to test it out

def main():
    """
    compare direct convolution with
    discrete and analytic convolution
    """
    t = numpy.linspace(0,10,2**10)
    x = numpy.exp(-0.1*t)
    t0 = 2
    k0 = 2
    a = 1

    y = gaussian_t(t,a,t0,k0)

    xy1 = convolve_with_gaussian(x,t,a,t0,k0,0)
    xy2 = convolve_with_gaussian(x,t,a,t0,k0,1)    
    xy3 = convolve_with_gaussian(x,t,a,t0,k0,2)

    #plot the three together
    pylab.figure()
    pylab.subplot(311)
    pylab.plot(t,xy1,label='direct')
    pylab.plot(t,xy2,label='discrete')
    pylab.plot(t,xy3,label='analytic')
    pylab.legend()

    #plot differences
    pylab.subplot(312)
    pylab.plot(t,(xy2-xy1),label='discrete-direct')
    pylab.legend()

    pylab.subplot(313)
    pylab.plot(t,(xy3-xy1),label='analytic-direct')
    pylab.legend()
    pylab.show()

if __name__ == '__main__':
    main()
