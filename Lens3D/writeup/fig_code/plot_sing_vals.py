"""
Plot the singular values for the lensing tomography transformation
"""

#-append the correct paths for importing Lens3D modules--------------
import sys, os
Lens3D_path = os.environ.get('LENS3D_PATH')
if Lens3D_path is None:
    raise ValueError, 'must set environment variable LENS3D_PATH'
sys.path.append(Lens3D_path)
sys.path.append(os.path.join(Lens3D_path,'3D_shear_generation'))
#--------------------------------------------------------------------

from Reconstructions import *
from matplotlib.ticker import MultipleLocator

def plot_sing_vals(Npix = 16,
                   z_gamma = numpy.linspace(0.08,2.0,25),
                   z_delta = numpy.linspace(0.1,2.0,20),
                   Ngal = 10,
                   z0 = 0.57,
                   sig = 0.3,
                   border_size = 2,
                   border_noise = 1E4):
    Nx = Ny = Npix
    Nzg = len(z_gamma)
    Nzd = len(z_delta)

    N_border_pix = 4*border_size*(Npix - border_size)
    
    tot_size = Nx*Ny*Nzg

    #construct theta arrays
    theta_min = 0
    theta_max = Npix-1
    
    theta1 = numpy.linspace(theta_min,theta_max,Nx)
    theta2 = numpy.linspace(theta_min,theta_max,Ny)
    
    theta = theta_comp_to_grid(theta1,theta2).ravel()
    P_gk = construct_P_gk( 0,Nx-1,Nx,
                           0,Ny-1,Ny,
                           z_gamma)
    P_kd = construct_P_kd(Nx,Ny,z_gamma,z_delta)
    N_angular = compute_N_angular(Nx,border_size,border_noise)
    N_los = compute_N_los(z_gamma,z0,sig,Ngal)
    
    U,S,V = compute_SVD(P_gk,
                        P_kd,
                        N_angular,
                        N_los)

    sig_QP = S.data_
    sig_QP.sort()
    sig_QP = sig_QP[::-1]

    var = sig_QP**2
    var /= var.sum()
    sumvar = numpy.cumsum(var)

    pylab.figure()
    pylab.semilogy(numpy.arange(1,len(sig_QP)+1),
                   sig_QP,'-k')
    pylab.xlabel(r'$n$',fontsize=18)
    pylab.ylabel(r'$\sigma_n$',fontsize=18)
    ylim = pylab.ylim()
    
    vars = ('99','99.9','99.99')
    for i in range(len(vars)):
        var = vars[i]
        n = numpy.searchsorted(sumvar,0.01*float(var))
        pylab.plot([n+1,n+1],ylim,':k')
        pylab.text(n+1000,10*ylim[0],var+"%",fontsize=14,rotation=90)
    pylab.ylim(ylim)

    pylab.title(r'$M_{\gamma\delta}\ \rm{Signular\ Values}$')
    
    if Npix==64:
        pylab.gca().xaxis.set_major_locator(MultipleLocator(15000))
        pylab.xlim(0,90000)

    

if __name__ == '__main__':
    N = 64
    border_size = N/16

    z_gamma = numpy.linspace(0.08,2.0,25)
    z_delta = numpy.linspace(0.1,2.0,20)

    plot_sing_vals(Npix = N,
                   z_gamma = z_gamma,
                   z_delta = z_delta,
                   Ngal = 10,
                   sig = 0.3,
                   border_size = border_size,
                   border_noise = 1E3 )

    pylab.savefig('sing_vals.eps')
    pylab.savefig('sing_vals.pdf')

    if '-show' in sys.argv:
        pylab.show()
