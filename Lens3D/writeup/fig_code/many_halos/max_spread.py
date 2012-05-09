import numpy
from scipy.interpolate import splrep,splev

def compute_max_spread(z,delta,interpolate=True):
    """
    returns the central wavelength, and the FWHM spread
    """
    if interpolate:
        tck = splrep(z,delta)
        z = numpy.linspace(z[0],z[-1],10*len(z))
        delta = splev(z,tck)

    #correct for things sloping up to the boundaries
    i_start = 0
    while delta[i_start]>delta[i_start+1]:
        i_start += 1

    i_stop = len(delta)-1
    while delta[i_stop]>delta[i_stop-1]:
        i_stop -= 1

    i_max = numpy.argmax(delta[i_start:i_stop+1])+i_start
    
    v_max = delta[i_max]
    #find min:
    i0=i_max
    if i_max == 0 or i_max==len(delta)-1:
        return z[i_max],0

    while i0>0 and delta[i0]>0.5*v_max:
        i0-=1
    z0 = z[i0] + (z[i0+1]-z[i0]) \
         * (0.5*v_max-delta[i0])/(delta[i0+1]-delta[i0])

    i1 = i_max
    while i1<len(delta)-1 and delta[i1]>0.5*v_max:
        i1+=1
    z1 = z[i1-1] + (z[i1]-z[i1-1]) \
         * (0.5*v_max-delta[i1])/(delta[i1-1]-delta[i1])

    return z[i_max], z1-z0

if __name__ == '__main__':
    import pylab
    x = range(10)
    y = [0.1, 0.2, 0.4, 0.5, 0.8, 0.9, 0.5, 0.3, 0.1, 0.1]
    
    pylab.plot(x,y)
    print compute_max_spread(x,y)
    pylab.show()
        
