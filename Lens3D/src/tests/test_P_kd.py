
#append the correct path for importing-----------------------------
import sys, os
pypath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if pypath: sys.path.append(pypath)
sys.path.append(os.path.join(pypath,'3D_shear_generation'))
#-------------------------------------------------------------------

from Lens3D import *
from cosmo_tools import Cosmology
import numpy
import pylab

def construct_P_kd(N1,N2,z_kappa,z_Delta,
                   cosmo=None,**kwargs):
    """
    construct the P_kd (kappa-Delta) matrix such that
    kappa = P_kd * Delta
    
    equivalent to equation 31 & 32 in Simon 2009,
    using Delta = delta/a as in Hu and Keeton 2003
    """
    if cosmo==None:
        cosmo = Cosmology(**kwargs)
    Nj = len(z_kappa)
    Nk = len(z_Delta)

    if max(z_Delta) > max(z_kappa):
        print "-------"
        print "WARNING: construct_P_kd: singular matrix [ min(z_kappa) < min(z_Delta) ]"
        print "-------"
    
    P = numpy.zeros([Nj,Nk])

    #array to hold the comoving distance to each z in z_Delta
    Dk = numpy.zeros(Nk+1)

    #for ease of calculation below,
    # make z_Delta[-1] = 0
    z_Delta = numpy.concatenate([z_Delta,[0]])

    for k in range(Nk):
        Dk[k] = cosmo.Dc(z_Delta[k])

    for j in range(Nj):
        Dj = cosmo.Dc(z_kappa[j])
        for k in range(Nk):
            if Dj < Dk[k]:
                P[j,k] = 0
            else:
                #P[j,k] = (Dj-Dk[k])*Dk[k]/Dj \
                #         * (z_Delta[k]-z_Delta[k-1]) / cosmo.H(z_kappa[j])
                P[j,k] = (Dk[k]-Dk[k-1]) * (Dj-Dk[k])*Dk[k]/Dj*(1.+z_Delta[k])

    #P *= ( 1.5 * cosmo.c*cosmo.Om*(cosmo.H0)**2 )
    P *= ( 1.5 * cosmo.Om*(cosmo.H0 / cosmo.c)**2 )

    print P.shape
    
    for i in range(P.shape[0]):
        pylab.plot(z_delta,P[i])
    pylab.show()
    exit()

    return Lens3D_los_mat(Nk,N1,N2,Nj,data=P)


if __name__ == '__main__':
    z_delta = numpy.linspace(0.1,1.5,15)
    z_kappa = numpy.linspace(0.05,1.5,30)

    construct_P_kd(2,2,z_kappa,z_delta)
