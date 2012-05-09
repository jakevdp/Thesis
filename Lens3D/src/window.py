import numpy

"""
Module containing various analytic fourier transforms of window functions
"""


def pixel(k,r1,r2,w):
    """
    returns f(k), the analytic radial fourier transform of a windown function
    which describes a square pixel with angular width w (in radians)
    from r1 to r2 (in Mpc).
    Note that setting width = pi will (by construction) evaluate over all
    4pi radians on the sky - the square pixel approximation only works
    for width << 1
    
    inputs:
     k    - radial wave number in Mpc^-1
     r1   - distance to start of the bin in Mpc
     r2   - distance to the end of the bin in Mpc
     w    - width in radians of the square pixels

    return value:
     f(k) - fourier transform of pixel in the radial direction, evaluated at k
    
    """
    #X1 = lambda k: r1 * k * numpy.sin(width/2)
    #X2 = lambda k: r2 * k * numpy.sin(width/2)

    S = numpy.sin(w/2)

    return 3/( k**3 * S**3 * (r2**3 - r1**3)) * (  S*r1*k*numpy.cos(S*r1*k)
                                                   - S*r2*k*numpy.cos(S*r2*k)
                                                   - numpy.sin(S*r1*k)
                                                   + numpy.sin(S*r2*k)       )
#end pixel

def tophat(kR):
    """
    returns the fourier transform of a 3-d tophat window function.
    
    kR is the radial wave number, in units of the radius R of the window.
    """
    return 3.0 * ( numpy.sin(kR) - kR*numpy.cos(kR) ) / (kR)**3
#end tophat

def gaussian(kR):
    """
    returns the fourier transform of a 3-d gaussian window function.
    
    kR is the radial wave number, in units of the radius R of the window.
    """
    return numpy.exp(-0.5*(kR)**2)
#end gaussian
