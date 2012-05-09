import numpy
from scipy.sparse.linalg import *

#append the correct path for importing-----------------------------
import sys, os
pypath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if pypath: sys.path.append(pypath)
sys.path.append(os.path.join(pypath,'3D_shear_generation'))
#-------------------------------------------------------------------

from Simon_Taylor_method import construct_P_gk
from Lens3D import get_mat_rep

theta1_min = 0
theta1_max = 31
N1 = 64
theta1 = numpy.linspace(theta1_min,theta1_max,N1)

theta2_min = 0
theta2_max = 31
N2 = 64
theta2 = numpy.linspace(theta2_min,theta2_max,N2)

z_kappa = [0.1]
z_gamma = z_kappa

P_gk = construct_P_gk( theta1_min, theta1_max, N1,
                       theta2_min, theta2_max, N2,
                       z_kappa )

P_gk_H = P_gk.conj_transpose()

P_gk_I = P_gk.inverse()
P_gk_I_H = P_gk_I.conj_transpose()

P = LinearOperator(P_gk.shape,dtype=P_gk.dtype,
                   matvec = P_gk.matvec,
                   rmatvec = P_gk_H.matvec)

PI = LinearOperator(P_gk_I.shape,dtype=P_gk_I.dtype,
                    matvec = P_gk_I.matvec,
                    rmatvec = P_gk_I_H.matvec)

P_gk_rep = get_mat_rep(P_gk)

v = numpy.random.random(P_gk.shape[1])

def callback(*args):
    callback.N += 1
callback.N = 0

from time import time
#t0 = time()
#print numpy.dot( numpy.linalg.inv(P_gk_rep), v )[:5]
#print time()-t0

t0 = time()
R,err = bicg(P,v,
             M = PI,
             callback=callback)
print callback.N,err
print R[:5]
print time()-t0
