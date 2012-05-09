import numpy
import pylab

#append the correct path for importing-----------------------------
import sys, os
pypath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if pypath: sys.path.append(pypath)
#-------------------------------------------------------------------

from Simon_Taylor_method import *
from S_DD import P_2D
from cosmo_tools import Cosmology

theta1_min = 0
theta1_max = 10
N1 = 16
theta1 = numpy.linspace(theta1_min,theta1_max,N1)

theta2_min = 0
theta2_max = 10
N2 = 16
theta2 = numpy.linspace(theta2_min,theta2_max,N2)

z_delta = [0.1]
    

#test P_2D inversion
if False:
    print "testing P2D"
    P2D = P_2D(zmin = 0.1,
               zmax = 0.2,
               Nz = 4,
               cosmo = Cosmology(),
               use_ST_form=True)

    P2DI = P2D.I

    x = numpy.linspace(0,10,100)

    pylab.subplot(311)
    pylab.plot(x,P2D(x))
    pylab.subplot(312)
    pylab.plot(x,P2DI(x))
    pylab.subplot(313)
    pylab.plot(x,P2D(x)*P2DI(x))
    pylab.show()
    
    exit()

#test S_DD
if False:
    #construct signal (angular) covariance
    print "construct S_DD"
    S_DD = construct_angular_S_dd( theta1_min, theta1_max, N1,
                                   theta2_min, theta2_max, N2,
                                   z_delta, cosmo=Cosmology() )

    S_DD_I = S_DD.inverse(True)
    S_DD_I2 = S_DD.inverse(False)
    
    print "computing full matrices:"
    S_DD_r = S_DD.full_matrix
    S_DD_I_r = S_DD_I.full_matrix
    S_DD_I_r_2 = S_DD_I2.full_matrix

    ID = numpy.dot(S_DD_I_r,S_DD_r)
    ID2 = numpy.dot(S_DD_I_r_2,S_DD_r)

    pylab.figure()
    pylab.imshow(abs(S_DD_r),interpolation='nearest')
    pylab.colorbar()
    
    pylab.figure()
    pylab.imshow(abs(S_DD_I_r),interpolation='nearest')
    pylab.title('analytic inverse')
    pylab.colorbar()
    
    
    pylab.figure()
    pylab.imshow(abs(ID),interpolation='nearest')
    pylab.title('analytic identity')
    pylab.colorbar()
    
    pylab.figure()
    pylab.imshow(abs(S_DD_I_r_2),interpolation='nearest')
    pylab.title('numerical inverse')
    pylab.colorbar()
    
    pylab.figure()
    pylab.imshow(abs(ID2),interpolation='nearest')
    pylab.title('numerical identity')
    pylab.colorbar()
    
    pylab.show()
    exit()

#test S_DD in fourier space
if False:
    print "testing S_DD analytic inversion"
    S_DD = construct_angular_S_dd( theta1_min, theta1_max, N1,
                                   theta2_min, theta2_max, N2,
                                   z_delta, cosmo=Cosmology() )

    S_DDI = S_DD.I
    
    v = S_DD.view_as_Lens3D_vec(numpy.random.random(S_DD.shape[1])-0.5)

    v2 = S_DD.matvec(v)
    v2ft1 = v2.lens_plane_fft(0)
    
    pylab.figure()
    pylab.imshow(abs(v2ft1))
    pylab.colorbar()
    
    v2.del_lens_plane_fft(0)
    v2ft2 = v2.lens_plane_fft(0)
    
    pylab.figure()
    pylab.imshow(abs(v2ft2))
    pylab.colorbar()
    
    pylab.figure()
    pylab.imshow(abs(v2ft2-v2ft1))
    pylab.colorbar()
    
    v3 = S_DDI.matvec(v2)

    pylab.figure()
    pylab.plot(v.vec,v3.vec,'.k')
    pylab.show()

    exit()


#test P_gk
if True:
    print "testing P_gk analytic inversion"
    P_gk = construct_P_gk( theta1_min, theta1_max, N1,
                           theta2_min, theta2_max, N2,
                           z_delta )
    P_gkI = P_gk.inverse(True)
    P_gkI2 = P_gk.inverse(False)

    P_gk_r = P_gk.as_Lens3D_lensplane().full_matrix
    #P_gk_r = P_gk.full_matrix
    P_gkI_r = P_gkI.full_matrix
    P_gkI2_r = P_gkI2.full_matrix

    ID = numpy.dot(P_gkI_r,P_gk_r)
    ID2 = numpy.dot(P_gkI2_r,P_gk_r)

    pylab.figure()
    pylab.imshow(P_gk_r.real,interpolation='nearest')
    pylab.colorbar()

    pylab.figure()
    pylab.imshow(P_gkI_r.real,interpolation='nearest')
    pylab.title('analytic inverse')
    pylab.colorbar()

    pylab.figure()
    pylab.imshow(ID.real,interpolation='nearest')
    pylab.title('analytic identity')
    pylab.colorbar()

    pylab.figure()
    pylab.imshow(P_gkI2_r.real,interpolation='nearest')
    pylab.title('numerical inverse')
    pylab.colorbar()

    pylab.figure()
    pylab.imshow(ID2.real,interpolation='nearest')
    pylab.title('numerical identity')
    pylab.colorbar()
    
    pylab.show()
