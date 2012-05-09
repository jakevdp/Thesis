
#append the correct path for importing-----------------------------
import sys, os
pypath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if pypath: sys.path.append(pypath)
sys.path.append(os.path.join(pypath,'3D_shear_generation'))
#-------------------------------------------------------------------

from Simon_Taylor_method import *
from generate_3D_shear import create_random_profiles
from thin_lens import *

from scipy.sparse.linalg import cg, LinearOperator\

theta1_min = 0
theta1_max = 15
N1 = 16
theta1 = numpy.linspace(theta1_min,theta1_max,N1)

theta2_min = 0
theta2_max = 15
N2 = 16
theta2 = numpy.linspace(theta2_min,theta2_max,N2)

#z_kappa = [0.1,0.15,0.2]
#z_kappa = [0.5,1.0,1.5]
z_kappa = [0.3,0.6,0.9,1.2]
z_gamma = z_kappa
#z_delta = [0.05,0.1,0.15]
#z_delta = [0.4,0.9,1.4]
z_delta = [0.4,0.8,1.2]
z_Sigma = z_delta

print "Nz_in =",len(z_kappa)
print "Nz_out = ",len(z_delta)

border_size = 2

"""

N_halos = 10
sig_min = 500
sig_max = 1000

generate_3D_shear('tmp.dat',
                  N_halos,
                  zmin,zmax,
                  sig_min,sig_max,
                  theta1_min,theta1_max,
                  theta2_min,theta2_max,
                  z_kappa_range,
                  theta1range,
                  theta2range,
                  rseed = 1)

z,theta1,theta2,L = read_shear_file('tmp.dat')
"""


#PS = create_random_profiles(N_halos,
#                            zmin,zmax,
#                            sig_min,sig_max,
#                            theta1_min,theta1_max,
#                            theta2_min,theta2_max,
#                            rseed = 3)

#PS = SIS(0.15,200,5.+5j)
#PS = NFW(0.15, 5+5j, Mvir = 1E13, rvir = 1.0, alpha = 1, c = 10.0)
#PS = SIS(0.15,200,4.2+7.1j)
PS = ProfileSet(NFW(0.05, 4.2+7.1j, Mvir = 1E13,
                    rvir = 1.0, alpha = 1, c = 10.0),
                NFW(0.15, 6.4+7.1j, Mvir = 1E13,
                    rvir = 1.0, alpha = 1, c = 10.0) )

gamma = PS.get_gamma_vector(theta1,theta2,z_gamma)
kappa = PS.get_kappa_vector(theta1,theta2,z_kappa)
Sigma = PS.get_Sigma_vector(theta1,theta2,z_Sigma)

print "construct_P_gk"
P_gk = construct_P_gk( theta1_min, theta1_max, N1,
                       theta2_min, theta2_max, N2,
                       z_kappa )

P_kd = construct_P_kd(N1,N2,z_kappa,z_delta)

print "gamma1 min/max:",numpy.min(gamma.vec.real),numpy.max(gamma.vec.real)
print "gamma2 min/max:",numpy.min(gamma.vec.imag),numpy.max(gamma.vec.imag)

#add shear noise
Nx = N1
Ny = N2
Nz = len(z_gamma)

Ngal = 20
sig  = 0.3
noise = sig**2 * 1./numpy.sqrt(Ngal)

print "noise level:",noise

#normally distributed noise, with a random phase
gamma.vec += ( noise \
                   * numpy.random.normal(size = Nx*Ny*Nz) \
                   * numpy.exp(2j*numpy.pi*numpy.random.random(Nx*Ny*Nz)) )

#test Sigma->kappa
if False:
    #note to self, 4-7-10.  Should use this section to test relative
    # Sigma->delta scaling.
    print "testing Sigma->kappa"
    kappa_test = P_kd.matvec(Sigma)
    
    P_kdI = P_kd.I
    Sigma_test = P_kdI.matvec(kappa)

    for i in range(gamma.Nz):
        pylab.figure( figsize=(12,8) )
        pylab.subplot(221)
        kappa.contourf_lens_plane(i,loglevels=False)
        pylab.title(r'$\kappa_{\rm{true}}$')
        
        pylab.subplot(222)
        kappa_test.contourf_lens_plane(i,'r',loglevels=False)
        pylab.title(r'$\kappa_{\rm{test}}$')

        pylab.subplot(223)
        Sigma.contourf_lens_plane(i,loglevels=False)
        pylab.title(r'$\Sigma_{\rm{true}}$')
        
        pylab.subplot(224)
        Sigma_test.contourf_lens_plane(i,'r',loglevels=False)
        pylab.title(r'$\Sigma_{\rm{test}}$')
        
    pylab.show()
    exit()

#test kappa->gamma
if False:
    print "testing kappa->gamma"
    gamma_test = P_gk.matvec(kappa)
    P_gkI = P_gk.I
    kappa_test = P_gkI.matvec(gamma)

    for i in range(gamma.Nz):
        pylab.figure( figsize=(12,8) )
        pylab.subplot(221)
        kappa.imshow_lens_plane(i,'r',loglevels=False)
        gamma.fieldplot_lens_plane(i)
        pylab.title(r'$\gamma,\kappa\ \rm{true}$')
        
        pylab.subplot(222)
        gamma_test.imshow_lens_plane(i,'n')
        pylab.title(r'$|\gamma_{\rm{test}}|$')

        pylab.subplot(223)
        kappa_test.imshow_lens_plane(i,'r',loglevels=False)
        gamma_test.fieldplot_lens_plane(i)
        pylab.title(r'$\gamma,\kappa\ \rm{test}$')

        pylab.subplot(224)
        pylab.imshow(abs(gamma.lens_plane(i)\
                             -gamma_test.lens_plane(i)).T,
                     interpolation='nearest',cmap=pylab.cm.gray,
                     origin = 'lower')
        pylab.title(r'$|\gamma_{\rm{test}} - \gamma_{\rm{true}}|$')
        pylab.colorbar()

    pylab.show()
    exit()

#test Sigma->gamma
if False:
    print "testing Sigma->gamma"
    def matvec(v):
        v1 = P_kd.matvec(v)
        return P_gk.matvec(v1)

    M = LinearOperator(P_kd.shape,matvec=matvec,dtype=complex)
    
    kappa_test = P_kd.matvec(Sigma)
    gamma_test = M.matvec(Sigma.vec)
    gamma_test = P_kd.view_as_Lens3D_vec(gamma_test)
    
    for i in range(gamma.Nz):
        pylab.figure( figsize=(6,8) )
        pylab.subplot(211)
        kappa.imshow_lens_plane(i)
        gamma.fieldplot_lens_plane(i)
        pylab.title(r"$\kappa,\gamma\ \rm{true}\ (z=%.2f)$" % z_gamma[i])
        pylab.subplot(212)
        kappa_test.imshow_lens_plane(i)
        gamma_test.fieldplot_lens_plane(i)
        pylab.title(r"$\kappa,\gamma\ \rm{test}\ (z=%.2f)$" % z_gamma[i])
    pylab.show()
    exit()

#test gamma->delta without filtering
if False:
    print "testing gamma->delta without filtering"
    P_gkI = P_gk.inverse(False)
    P_kdI = P_kd.I

    kappa_test = P_gkI.matvec(gamma)
    delta = P_kdI.matvec(kappa_test)

    #try back-computing gamma from the result
    if False:
        kappa_back = P_kd.matvec(delta)
        gamma_back = P_gk.matvec(kappa_back)
        
        gamma_back = P_gk.view_as_Lens3D_vec(gamma_back)
        for i in range(gamma.Nz):
            pylab.figure(figsize=(12,8))
            pylab.subplot(221)
            kappa.imshow_lens_plane(i)
            gamma.fieldplot_lens_plane(i)
            pylab.title(r"$\gamma,\kappa\ \rm{true}\ (z=%.2f)$" % z_gamma[i])
            
            pylab.subplot(222)
            gamma.contourf_lens_plane(i,'n')
            pylab.title(r"$\gamma\ \rm{true}\ (z=%.2f)$" % z_gamma[i])
            
            pylab.subplot(223)
            kappa.imshow_lens_plane(i)
            gamma_back.fieldplot_lens_plane(i)
            pylab.title(r"$\kappa\ \rm{true},\ \gamma\ \rm{back-computed}\ (z=%.2f)$" % z_gamma[i])
            
            pylab.subplot(224)
            gamma_back.contourf_lens_plane(i,'n')
            pylab.title(r"$\gamma\ \rm{back-computed}\ (z=%.2f)$" % z_gamma[i])
    #otherwise, plot the result
    else:
        for i in range(gamma.Nz):
            pylab.figure(figsize=(6,8))
            pylab.subplot(211)
            #Sigma.contourf_lens_plane(i,'n')
            Sigma.imshow_lens_plane(i,'n')
            pylab.title(r"$\Sigma\ \rm{true}\ (z=%.2f)$" % z_gamma[i])
            
            pylab.subplot(212)
            #delta.contourf_lens_plane(i,'n')
            delta.imshow_lens_plane(i,'n')
            pylab.title(r"$\delta\ \rm{unfiltered}\ (z=%.2f)$" % z_gamma[i])
    pylab.show()
    exit()
    
    
#construct signal (angular) covariance
print "construct S_dd"
S_dd = construct_angular_S_dd( theta1_min, theta1_max, N1,
                               theta2_min, theta2_max, N2,
                               z_delta, cosmo=Cosmology() )

#construct noise vectors
N = Lens3D_diag(Nz,Nx,Ny,noise**2 * numpy.ones(Nx*Ny*Nz))
#N.set_border( border_size,1E3 )

#choose alpha (filtering level)
alpha = 1.0

#test condition number
estimate_condition_number(P_kd,
                          P_gk,
                          S_dd,
                          N,
                          alpha,
                          compute_exact = True)    

exit()

#compute delta from gamma
delta = calculate_delta_simple(gamma,P_kd,P_gk,S_dd,N,alpha)

delta.set_border(border_size,0)

for i in range(len(z_delta)):
    pylab.figure(figsize=(10,8))

    pylab.subplot(221)
    kappa.imshow_lens_plane(i)
    gamma.fieldplot_lens_plane(i)
    pylab.title(r'$\gamma_{in}/\kappa_{in}\  (z=%.2f)$' % z_gamma[i])
    
    pylab.subplot(223)
    Sigma.imshow_lens_plane(i,'r',loglevels=False)
    pylab.title(r'$\Sigma_{in}\ (z=%.2f)$' % z_delta[i])
    
    pylab.subplot(222)
    delta.imshow_lens_plane(i,'r',loglevels=False)#,gaussian_filter=1.0)
    pylab.title(r'$Re[\delta_{out}]\ (z=%.2f)$' % z_delta[i])
    
    pylab.subplot(224)
    delta.imshow_lens_plane(i,'i',loglevels=False)#,gaussian_filter=1.0)
    pylab.title(r'$Im[\delta_{out}]\ (z=%.2f)$' % z_delta[i])

pylab.show()
