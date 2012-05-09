import numpy
import pylab
from scipy import fftpack

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

N = 50
y = numpy.exp(-numpy.linspace(-2,2,2*N)**2)

M = numpy.zeros((N,N))
for i in range(N):
    M[i] = y[N-i:2*N-i]

pylab.figure()
pylab.imshow(M)
pylab.colorbar()

U,S,V = numpy.linalg.svd(M)

pylab.figure()
pylab.imshow(U)
pylab.colorbar()

pylab.figure()
pylab.imshow(V.T)
pylab.colorbar()

pylab.figure()
pylab.semilogy(S)

pylab.show()
