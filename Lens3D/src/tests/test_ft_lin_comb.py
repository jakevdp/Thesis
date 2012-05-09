import numpy
import pylab

#append the correct path for importing-----------------------------
import sys, os
pypath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if pypath: sys.path.append(pypath)
sys.path.append(os.path.join(pypath,'3D_shear_generation'))
#-------------------------------------------------------------------

from Simon_Taylor_method import *
from thin_lens import *
from scipy.sparse.linalg import cg, LinearOperator

theta1_min = 0
theta1_max = 10
N1 = 32
theta1 = numpy.linspace(theta1_min,theta1_max,N1)

theta2_min = 0
theta2_max = 10
N2 = 32
theta2 = numpy.linspace(theta2_min,theta2_max,N2)

z_kappa = [0.1,0.15,0.2]
z_gamma = z_kappa
z_delta = [0.05,0.1,0.15]
z_Sigma = z_delta

border_size = 4

PS = ProfileSet(NFW(0.05, 4.2+7.1j, Mvir = 1E13,
                    rvir = 1.0, alpha = 1, c = 10.0),
                NFW(0.15, 6.4+7.1j, Mvir = 1E13,
                    rvir = 1.0, alpha = 1, c = 10.0) )

gamma = PS.get_gamma_vector(theta1,theta2,z_gamma)
kappa = PS.get_kappa_vector(theta1,theta2,z_kappa)
Sigma = PS.get_Sigma_vector(theta1,theta2,z_Sigma)

P_gk = construct_P_gk( theta1_min, theta1_max, N1,
                       theta2_min, theta2_max, N2,
                       z_kappa )

P_kd = construct_P_kd(N1,N2,z_kappa,z_delta)

S_dd = construct_angular_S_dd( theta1_min, theta1_max, N1,
                               theta2_min, theta2_max, N2,
                               z_delta, cosmo=Cosmology() )

P_gk_cross = P_gk.conj_transpose()
P_kd_T = P_kd.transpose()

v = numpy.random.normal(size=P_gk_cross.shape[1])
v = P_gk_cross.view_as_Lens3D_vec(v)

v2 = v
v2 = P_gk_cross.matvec(v2)
v2.move_to_real()
v2.move_to_fourier()
v2 = P_kd_T.matvec( v2 )
v2 = S_dd.matvec( v2 )
v2.move_to_real()
v2.move_to_fourier()
v2 = P_kd.matvec( v2 )
v2 = P_gk.matvec( v2 )

v3 = v
v3 = P_gk_cross.matvec(v3)
v3.move_to_real()
v3 = P_kd_T.matvec( v3 )
v3 = S_dd.matvec( v3 )
v3.move_to_real()
v3 = P_kd.matvec( v3 )
v3 = P_gk.matvec( v3 )

v4 = v
v4 = P_gk_cross.matvec(v4)
v4 = P_kd_T.matvec( v4 )
v4 = S_dd.matvec( v4 )
v4 = P_kd.matvec( v4 )
v4 = P_gk.matvec( v4 )

pylab.figure(figsize=(12,8))
pylab.subplot(221)
v2.imshow_lens_plane(0,'a')
pylab.subplot(222)
v3.imshow_lens_plane(0,'a')
pylab.subplot(223)
v4.imshow_lens_plane(0,'a')

pylab.show()
