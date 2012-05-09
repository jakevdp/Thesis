import numpy
import pylab
from scipy import fftpack
from scipy.ndimage import filters

N1 = 32
N2 = 32
N3 = 5

def my_dot(x,y):
    return numpy.dot(x,y.transpose( (1,0,2) ) )

def los_multiply(A,weights):
    return my_dot(weights,A)

def angular_multiply(A,mat):
    return A*mat

def my_fft(x):
    N3,N1,N2 = x.shape
    return fftpack.fft2(x,(2*N1,2*N2))

def my_ifft(x_fft):
    N3,N1,N2 = x_fft.shape
    #return fftpack.ifft2(x_fft)[:,:N1/2,:N2/2]
    return fftpack.ifft2(x_fft,(N1/2,N2/2))[:,:N1/2,:N2/2]

numpy.random.seed(0)

weights = numpy.random.random( (N3,N3) )
x = numpy.random.normal(size=(N3,N1,N2))
for i in range(N3):
    x[i] = filters.gaussian_filter(x[i],3,mode='mirror')
A = numpy.random.random(size=(N3,2*N1,2*N2))
B = numpy.random.random(size=(N3,2*N1,2*N2))

x_fft = my_fft(x)
x_fft *= A

y1 = my_dot( weights,my_ifft(B*x_fft ) )
y2 = my_ifft( B*my_dot(weights,x_fft ) )

i = 2

kwargs = {"cmap":pylab.cm.gray,
          "interpolation":"nearest"}

#pylab.figure(figsize=(12,8))
#for j in range(4):
#    pylab.subplot(221+j)
#    pylab.imshow(abs(x[j]),**kwargs)
#    pylab.title('input')
#    pylab.colorbar()

pylab.figure(figsize = (12,8))
pylab.subplot(221)
pylab.imshow(abs(y1[i]),**kwargs)
pylab.colorbar()
pylab.title('abs(y1)')

pylab.subplot(222)
pylab.imshow(abs(y2[i]),**kwargs)
pylab.colorbar()
pylab.title('abs(y2)')

pylab.subplot(223)
pylab.imshow( y1[i].real-y2[i].real,**kwargs)
pylab.colorbar()
pylab.title('abs(Re[y1-y2])')

pylab.subplot(224)
pylab.imshow( y1[i].imag-y2[i].imag,**kwargs)
pylab.colorbar()
pylab.title('abs(Im[y1-y2])')

pylab.show()
exit()
