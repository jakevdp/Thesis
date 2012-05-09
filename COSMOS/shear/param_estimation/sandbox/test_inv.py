"""
explore how the model covariance inverse can be approximated
"""
import numpy
import pylab

from covariance_matrix import calc_C_matrices

C,Om_range,s8_range,evals,evecs = calc_C_matrices(compute=False)

nmodes = 40
evecs_n = evecs[:,:nmodes]

I = [[None for i in range(3)] for i in range(3)]
    
pylab.figure(figsize=(10,10))
for i,Om in enumerate(Om_range):
    print i
    for j,s8 in enumerate(s8_range):
        print '',j
        M = numpy.dot(evecs_n.T,numpy.dot(C[i,j],evecs_n))

        MI = numpy.linalg.inv(M)
        MI_approx = numpy.zeros(M.shape)
        MI_approx.flat[::1+M.shape[1]] = 1./M.diagonal()

        MI -= numpy.diag(1./evals[:nmodes])

        pylab.subplot(331+(3*i+j))
        #I[i][j] = pylab.imshow(numpy.log10(abs(MI-MI_approx)),
        #                       origin='lower',interpolation='nearest')
        I[i][j] = pylab.imshow(numpy.log10(abs(MI)),
                               origin='lower',interpolation='nearest')
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
