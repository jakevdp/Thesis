from Simon_Taylor_method import *
import sys, os
dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dirname,'3D_shear_generation'))
from generate_3D_shear import create_random_profiles
from thin_lens import *

from scipy.sparse.linalg import cg, LinearOperator
from Lens3D_display_results import *

def Lens3D_separation(sep_x,
                      sep_y,
                      M1 = 1E15,
                      M2 = 1E15,
                      z1 = 0.31,
                      z2 = 0.81,
                      Npix = 64,
                      z_kappa = numpy.linspace(0.08,2.0,25),
                      z_delta = numpy.linspace(0.1,2.0,20),
                      alpha = 0.9,
                      Ngal = 20,
                      sig = 0.3,
                      border_size = 8,
                      border_noise = 1E4,
                      rseed = 0):
    numpy.random.seed(0)

    theta_min = 0
    theta_max = Npix-1
    theta1 = numpy.linspace(theta_min,theta_max,Npix)
    theta2 = numpy.linspace(theta_min,theta_max,Npix)

    x_coord_1 = Npix/2
    y_coord_1 = Npix/2
    
    offset = 0.5+0.5j #because evaluating at r=0 produces errors

    x_coord_2 = x_coord_1 + sep_x
    y_coord_2 = y_coord_1 + sep_y

    PS =  ProfileSet(NFW(z1, x_coord_1 + 1j*y_coord_1 + offset, 
                         Mvir = M1,
                         rvir = 1.0, alpha = 1, c = 10.0),
                     NFW(z2, x_coord_2 + 1j*y_coord_2 + offset, 
                         Mvir = M1,
                         rvir = 1.0, alpha = 1, c = 10.0)   )
    
    #determine noise level
    noise = sig * 1.0 / numpy.sqrt(Ngal)
    print "noise level: %.2g" % noise
    print "alpha:       %.2g" % alpha
    
    #extract gamma,kappa,Sigma in the correct bins
    gamma = PS.get_gamma_vector(theta1,theta2,z_kappa)
    kappa = PS.get_kappa_vector(theta1,theta2,z_kappa)
    Sigma = PS.get_Sigma_vector(theta1,theta2,z_delta)

    tot_size = Npix*Npix*len(z_kappa)

    #add shear noise:
    # normally distributed noise, with a random phase
    gamma.vec += ( noise * numpy.random.normal(size = tot_size) \
                       * numpy.exp(2j*numpy.pi*numpy.random.random(tot_size)) )
    
    #construct transformation matrices
    P_gk = construct_P_gk( theta_min, theta_max, Npix,
                           theta_min, theta_max, Npix,
                           z_kappa )
    
    P_kd = construct_P_kd(Npix,Npix,z_kappa,z_delta)

    #construct signal (angular) covariance
    S_DD = construct_angular_S_dd( theta_min, theta_max, Npix,
                                   theta_min, theta_max, Npix,
                                   z_delta, cosmo=Cosmology() )
    
    #construct (diagonal) noise matrix
    N = Lens3D_diag(len(z_kappa),Npix,Npix,noise**2 * numpy.ones(tot_size))
    N.set_border( border_size,border_noise )

    #compute delta from gamma
    delta = calculate_delta_simple(gamma,P_kd,P_gk,S_DD,N,alpha)
    

    #set the border to zero: this part was deweighted by the algorithm
    delta.set_border(border_size,0)

    #determine plot color ranges
    dmin = numpy.min(delta.data.real)
    dmax = numpy.max(delta.data.real)
    Smin = numpy.min(Sigma.data.real)
    Smax = numpy.max(Sigma.data.real)
    
    #do not plot border
    xlim = (border_size-0.5,Npix-border_size-0.5)
    ylim = (border_size-0.5,Npix-border_size-0.5)
    
    L3D_disp = Lens3D_display_results(z_kappa,
                                      z_delta,
                                      theta1[1]-theta1[0],
                                      theta2[1]-theta2[0],
                                      delta,
                                      gamma,
                                      kappa,
                                      Sigma)
    fig1 = L3D_disp.plot_los(x_coord_1,y_coord_1,
                             numpy.searchsorted(z_delta,z1),
                             border = border_size)
    fig2 = L3D_disp.plot_los(x_coord_2,y_coord_2,
                             numpy.searchsorted(z_delta,z2),
                             border = border_size)

    return fig1,fig2


if __name__ == '__main__':
    
    for sep_x in [0,1,2,3,5,10,15]:
        fig1,fig2 = Lens3D_separation(sep_x=sep_x,
                                      sep_y=0,
                                      M1 = 1E14,
                                      M2 = 1E14,
                                      z1 = 0.29,
                                      z2 = 0.79,
                                      Npix = 64,
                                      z_kappa = numpy.linspace(0.08,2.0,25),
                                      z_delta = numpy.linspace(0.1,2.0,20),
                                      alpha = 0.1,
                                      Ngal = 20,
                                      sig = 0.3,
                                      border_size = 8,
                                      rseed = 0)
    pylab.show()
