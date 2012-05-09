"""
explore how the model covariance matrix changes for parameter estimation.
"""

import os
import sys
import numpy
import pylab
from matplotlib import ticker

sys.path.append(os.path.abspath('../../'))
from shear_KL_source.shear_correlation import xi_plus
from shear_KL_source.shear_KL import compute_correlation_matrix, compute_KL

def calc_C_matrices(Npix = 32,         #pixels
                    field_width = 120., #arcmin
                    z0 = 0.5,
                    nz_a = 2,
                    nz_b = 1.5,
                    zlim = (0,3),
                    ngal_arcmin=20,
                    compute=True,
                    filename='C_calc.npz'):
    dpix = field_width/Npix
    
    n_z =  lambda z: z**nz_a * numpy.exp(-(z/z0)**nz_b)
    
    ngal = ngal_arcmin*dpix**2
    
    if compute:
        Om_range = (0.1,0.3,0.5)
        s8_range = (0.6,0.8,1.0)
        C = numpy.zeros((3,3,Npix**2,Npix**2))
        
        for i,Om in enumerate(Om_range):
            for j,s8 in enumerate(s8_range):
                xi = xi_plus(n_z,
                             zlim,
                             Nz=20,
                             Or=0,
                             Om=Om,
                             sigma8=s8)
            
                C[i,j] = compute_correlation_matrix(xi,
                                                    Npix = Npix,
                                                    dpix = dpix,
                                                    ngal = ngal,
                                                    sigma = sigma,
                                                    whiten=True)
        evals,evecs = compute_KL(C[1,1])
        
        numpy.savez(filename,
                    C=C,
                    Om_range=Om_range,
                    s8_range=s8_range,
                    evals=evals,
                    evecs=evecs)
    else:
        X = numpy.load(filename)
        C = X['C']
        Om_range = X['Om_range']
        s8_range = X['s8_range']
        evals = X['evals']
        evecs = X['evecs']

    return C,Om_range,s8_range,evals,evecs

if __name__ == '__main__':
    C,Om_range,s8_range,evals,evecs = calc_C_matrices(compute=False)

    nmodes = 500
    evecs_n = evecs[:,:nmodes]

    I = [[None for i in range(3)] for i in range(3)]

    pylab.figure(figsize=(10,10))
    for i,Om in enumerate(Om_range):
        for j,s8 in enumerate(s8_range):
            pylab.subplot(331+(3*i+j))
            M = numpy.dot(evecs_n.T,numpy.dot(C[i,j],evecs_n))
            
            s,logdet = numpy.linalg.slogdet(M)
            approx_det = numpy.prod(M.diagonal())
            approx_logdet = numpy.log(abs(approx_det))

            print "---------------------------------"
            print Om,s8
            print s,logdet
            print numpy.sign(approx_det), approx_logdet
            print "d|M|/|M| = %.2g" % abs(1 - numpy.exp(logdet-approx_logdet))
            
            M[numpy.where(M==0)]=1E-16
            I[i][j] = pylab.imshow(numpy.log10(abs(M)),
                                   origin='lower',
                                   interpolation='nearest')
            pylab.gca().xaxis.set_major_formatter(ticker.NullFormatter())
            pylab.gca().yaxis.set_major_formatter(ticker.NullFormatter())
            
            pylab.title(r'$\Omega_M=%.1f,\ \sigma_8=%.1f$' % (Om,s8))
    ax = pylab.axes([0.92,0.1,0.02,0.8])

    clim = [0,-10]
    for i in range(3):
        for j in range(3):
            Ic = I[i][j].get_clim()
            if Ic[0]<clim[0]: clim[0] = Ic[0]
            if Ic[1]>clim[1]: clim[1] = Ic[1]

    clim[0] = -6
    
    for i in range(3):
        for j in range(3):
            I[i][j].set_clim(clim)
    pylab.colorbar(cax=ax)
    
    pylab.show()
