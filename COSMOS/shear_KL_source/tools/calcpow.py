import numpy
from scipy import fftpack
import pylab

def calcpow(rhoin,dx1,dx2=None,Nl=100, shape=None):
    if numpy.any(numpy.iscomplex(rhoin)):
        return calcpow_complex(rhoin,dx1,dx2=None,Nl=Nl, shape=shape)
    else:
        return calcpow_scalar(rhoin,dx1,dx2=None,Nl=Nl, shape=shape)

def calcpow_complex(gammain, dx1, dx2=None, Nl=100, shape=None):
    """
    compute the power spectrum of the 2D array gammain (real or complex)
    dx1 and dx2 give the pixel size in arcmin
    """
    if dx2 is None:
        dx2 = dx1

    #convert pixel size to radians
    dx1 = dx1*numpy.pi/180./60.
    dx2 = dx2*numpy.pi/180./60.
        
    N1,N2 = gammain.shape

    cellarea = dx1*dx2
    area = N1*N2*cellarea

    #determing ell array
    ell_min = 2 * numpy.pi / max(N1*dx1,N2*dx2)
    ell_max = numpy.pi * numpy.sqrt(1. / dx1 / dx1 + 1. / dx2 / dx2)
    ell = numpy.linspace(ell_min, ell_max + 1, Nl)

    #create power array
    num = numpy.zeros(Nl)
    pow_ee = numpy.zeros(Nl)
    pow_bb = numpy.zeros(Nl)
    pow_eb = numpy.zeros(Nl)

    #fourier transform array
    g1 = fftpack.fft2(gammain.real, shape=shape)
    g2 = fftpack.fft2(gammain.imag, shape=shape)

    for j in range(N2):
        if (j > N2/2): j = j-N2
        for i in range(N1/2):
            lx = i*2*numpy.pi/N1/dx1
            ly = j*2*numpy.pi/N2/dx2
            labs = numpy.sqrt(lx*lx+ly*ly)
            pbl = labs*labs/2./numpy.pi

            if labs<ell_min: continue
            if labs>=ell_max: continue
          
            cos1=lx/labs
            sin1=ly/labs
            cos2=cos1*cos1-sin1*sin1
            sin2=2.*sin1*cos1

            #g1r = g1[i,j].real
            #g1i = g1[i,j].imag
            #g2r = g2[i,j].real
            #g2i = g2[i,j].imag

            #elr =  g1r*cos2 + g2r*sin2
            #eli =  g1i*cos2 + g2i*sin2
            #blr = -g1r*sin2 + g2r*cos2
            #bli = -g1i*sin2 + g2i*cos2
          
            #powee=elr*elr+eli*eli
            #powbb=blr*blr+bli*bli
            #poweb=elr*blr+eli*bli

            el = g1*cos2 + g2*sin2
            bl = -g1*sin2 + g2*cos2

            powee = (el*el.conj()).real
            powbb = (bl*bl.conj()).real
            poweb = (el*bl.conj()).real

            np = numpy.searchsorted( ell,labs )

            if i==0:
                num[np] += 1
                pow_ee[np] += powee*pbl
                pow_bb[np] += powbb*pbl
                pow_eb[np] += poweb*pbl
            else:
                num[np] += 2
                pow_ee[np] += 2*powee*pbl
                pow_bb[np] += 2*powbb*pbl
                pow_eb[np] += 2*poweb*pbl
            ##
        ##
    ##
    
    i = numpy.where(num==0)
    
    num[i] = 1
    pow_ee /= num
    pow_ee /= area
    pow_bb /= num
    pow_bb /= area
    pow_eb /= num
    pow_eb /= area

    return ell, pow_ee, pow_bb, pow_eb

def calcpow_EB_ratio(gammain,dx1,dx2=None,Nl=100, shape=None):
    """
    compute the power spectrum of the 2D array gammain (real or complex)
    dx1 and dx2 give the pixel size in arcmin
    """
    if dx2 is None:
        dx2 = dx1

    #convert pixel size to radians
    dx1 = dx1*numpy.pi/180./60.
    dx2 = dx2*numpy.pi/180./60.
        
    N1,N2 = gammain.shape

    cellarea = dx1*dx2
    area = N1*N2*cellarea

    #fourier transform array
    g1 = fftpack.fft2(gammain.real, shape=shape)
    g2 = fftpack.fft2(gammain.imag, shape=shape)

    #array of e^(2 i phi)
    lx = numpy.arange(N1)*2*numpy.pi/N1/dx1
    ly = numpy.arange(N2)*2*numpy.pi/N2/dx2
    ll = lx[:,None] + 1j*ly[None,:]
    e2iphi = ll / ll.conj()
    e2iphi[0,0] = 0.0 #this is a NaN: 0/0

    #array of labs2
    labs2 = lx[:,None]**2 + ly[None,:]**2

    #compute E and B fourier coefficients
    Ek = g1*e2iphi.real + g2*e2iphi.imag
    Bk = -g1*e2iphi.imag + g2*e2iphi.real

    Pee = abs(Ek)**2 * labs2/2*numpy.pi
    Pbb = abs(Bk)**2 * labs2/2*numpy.pi

    return numpy.sum(Pee)/numpy.sum(Pbb)
    

def calcpow_scalar(rhoin,dx1,dx2=None,Nl=100, shape=None):
    """
    compute the power spectrum of the 2D array rhoin (real or complex)
    dx1 and dx2 give the pixel size in arcmin
    """
    if dx2 is None:
        dx2 = dx1

    #convert pixel size to radians
    dx1 = dx1*numpy.pi/180./60.
    dx2 = dx2*numpy.pi/180./60.
        
    N1,N2 = rhoin.shape

    cellarea = dx1*dx2
    area = N1*N2*cellarea

    #determing ell array
    ell_min = 2 * numpy.pi / max(N1*dx1,N2*dx2)
    ell_max = numpy.pi * numpy.sqrt(1. / dx1 / dx1 + 1. / dx2 / dx2)
    ell = numpy.linspace(ell_min, ell_max + 1, Nl)

    #create power array
    num = numpy.zeros(Nl)
    power = numpy.zeros(Nl)

    #fourier transform array
    rho = fftpack.fft2(rhoin, shape=shape)

    for j in range(N2):
        if (j > N2/2): j = j-N2
        for i in range(N1/2):
            lx = i*2*numpy.pi/N1/dx1
            ly = j*2*numpy.pi/N2/dx2
            labs = numpy.sqrt(lx*lx+ly*ly)
            pbl = labs*labs/2./numpy.pi

            if labs<ell_min: continue
            if labs>=ell_max: continue

            powrho = numpy.abs(rho[i,j])**2

            np = numpy.searchsorted( ell,labs )

            if i==0:
                num[np] += 1
                power[np] += powrho*pbl
            else:
                num[np] += 2
                power[np] += 2*powrho*pbl
            ##
        ##
    ##
    
    i = numpy.where(num==0)
    
    num[i] = 1
    power /= num
    power /= area

    return ell, power
    

def test_calcpow():
    N1 = 128
    N2 = 128
    t1 = numpy.arange(N1)
    t2 = numpy.arange(N2)
    y1 = numpy.sin(t1*16.*numpy.pi/N1) + numpy.cos(t1*64.*numpy.pi/N1)
    y2 = numpy.sin(t2*16.*numpy.pi/N2) + numpy.sin(t2*32.*numpy.pi/N1)
    
    x = y1[:,None]*y2[None,:]
    x += 0.1*numpy.random.normal(size=(N1,N2))
    
    dt = 2.0

    ell,Pl = calcpow(x,dt,Nl=100)

    pylab.figure()
    pylab.imshow(x)
    pylab.colorbar()

    pylab.figure()
    pylab.semilogy(ell,Pl)
    
    i = numpy.argmax(Pl)
    print "scale of Pmax: %.3g arcmin" % (180.*60./ell[i])
    
    pylab.show()
    

def test_calcpow_complex():
    N1 = 128
    N2 = 128

    k1 = numpy.linspace(0,numpy.pi,N1)
    k2 = numpy.linspace(0,numpy.pi,N2)

    x = numpy.sin(k1)[:,None] * numpy.sin(k2)[None,:]

    #x *= (1.+1j)
    
    dt = 2.0

    ell,pee,pbb,peb = calcpow_complex(x,dt,Nl=20)

    pylab.figure()
    pylab.imshow( abs(x) )
    pylab.colorbar()

    pylab.figure()
    pylab.semilogy(ell,pee)
    pylab.semilogy(ell,pbb)
    pylab.semilogy(ell,abs(peb))
    
    i = numpy.argmax(pee)
    print "scale of Pmax: %.3g arcmin" % (180.*60./ell[i])

    EB = calcpow_EB_ratio(x,dt,Nl=20)

    print EB
    print numpy.sum(pee)/numpy.sum(pbb)
    
    pylab.show()


if __name__ == '__main__':
    test_calcpow_complex()
