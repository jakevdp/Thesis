#-append the correct paths for importing Lens3D modules--------------
import sys, os
Lens3D_path = os.environ.get('LENS3D_PATH')
if Lens3D_path is None:
    raise ValueError, 'must set environment variable LENS3D_PATH'
sys.path.append(Lens3D_path)
sys.path.append(os.path.join(Lens3D_path,'3D_shear_generation'))
#--------------------------------------------------------------------

import pylab
from Reconstructions import *

N = 64
border_size = N/16
border_noise = 1E3

z = 0.59
Mvir = 5E14
x_coord = N/2
y_coord = N/2

z_gamma = numpy.linspace(0.08,2.0,25)
z_delta = numpy.linspace(0.1,2.0,20)

Ngal = 70
z0 = 0.57
sig = 0.3

offset = 0.0+0.0j #because evaluating at r=0 produces errors

#construct theta arrays
theta_min = 0
theta_max = N-1

theta1 = numpy.linspace(theta_min,theta_max,N)
theta2 = numpy.linspace(theta_min,theta_max,N)

PS =  ProfileSet( NFW(z, x_coord + 1j*y_coord + offset, 
                      Mvir = Mvir,
                      rvir = 1.0, alpha = 1, c = 5.0) )

gamma = PS.get_gamma_vector(theta1,theta2,z_gamma)

#Signal
delta,N_cut,sig_cut = calculate_delta_SVD(gamma,
                                          z_delta = z_delta,
                                          z_gamma = z_gamma,
                                          v_cut = 0.1,
                                          theta_min = theta_min,
                                          theta_max = theta_max,
                                          N = N,
                                          border_size = border_size,
                                          border_noise = border_noise,
                                          Ngal = Ngal,
                                          z0 = z0,
                                          sig = sig)

#Noise
P_kd = construct_P_kd(N,N,z_gamma,z_delta)
U,S,VT = numpy.linalg.svd(P_kd.data_,full_matrices=0)

Ndd = S[0]**-2 * numpy.outer(VT[0],VT[0])
#Ndd = numpy.dot(VT[:-1].T,numpy.dot(numpy.diag(S[:-1]**-2),VT[:-1]))

i = numpy.searchsorted(z_delta,z0)

j = range(i-1,i+2)
print "signal:",
print delta.data[j,N/2,N/2].real

print "noise:",
print numpy.sqrt( Ndd[j,j].real )

print "signal-to-noise:",
print delta.data[j,N/2,N/2].real/numpy.sqrt( Ndd[i-1,i-1].real )

#delta.imshow_lens_plane('all')
#pylab.show()
