#-append the correct paths for importing Lens3D modules--------------
import sys, os
Lens3D_path = os.environ.get('LENS3D_PATH')
if Lens3D_path is None:
    raise ValueError, 'must set environment variable LENS3D_PATH'
sys.path.append(Lens3D_path)
sys.path.append(os.path.join(Lens3D_path,'3D_shear_generation'))
#--------------------------------------------------------------------

from Reconstructions import *

def plot_three(PS,
               i_x,i_y,i_z,
               Npix = 16,
               z_gamma = numpy.linspace(0.08,2.0,25),
               z_delta = numpy.linspace(0.1,2.0,20),
               Ngal = 100,
               z0 = 0.57,
               sig = 0.3,
               border_size = 2,
               border_noise = 1E3,
               rseed = 0,
               v_cuts = [1E-1,1E-2,1E-3,1E-4] ):
    """
    plot_three : plot ST method results with los
    """
    numpy.random.seed(rseed)
    Nx = Ny = Npix
    Nzg = len(z_gamma)
    Nzd = len(z_delta)

    print Nx*Ny*Nzd, 'total pixels'
    
    tot_size = Nx*Ny*Nzg

    #construct theta arrays
    theta_min = 0
    theta_max = Npix-1
    
    theta1 = numpy.linspace(theta_min,theta_max,Nx)
    theta2 = numpy.linspace(theta_min,theta_max,Ny)
    
    #extract gamma,kappa,Sigma in the correct bins
    gamma = PS.get_gamma_vector(theta1,theta2,z_gamma)
    add_noise_to_gamma(gamma,
                       z_gamma = z_gamma,
                       z0 = z0,
                       sig = sig,
                       Ngal = Ngal)
    Sigma = PS.get_Sigma_vector(theta1,theta2,z_delta,1)

    pylab.figure(figsize=(8,10))
    #3x2 plots: left will be lens-plane images, right will be line-of-sight
    
    delta_in = Sigma_to_delta(Sigma,z_delta)
    los_in =delta_in.line_of_sight(i_x,i_y).real
    
    for i in range(3):
        delta,N_cut,sig_cut = calculate_delta_SVD(gamma,
                                                  z_delta = z_delta,
                                                  z_gamma = z_gamma,
                                                  v_cut = v_cuts[i],
                                                  theta_min = theta_min,
                                                  theta_max = theta_max,
                                                  N = Npix,
                                                  border_size = border_size,
                                                  border_noise = border_noise,
                                                  Ngal = Ngal,
                                                  z0 = z0,
                                                  sig = sig)
        los = delta.line_of_sight(i_x,i_y)
        
        pylab.subplot(322+2*i)
        pylab.fill(z_delta,los_in,
                   fc = '#AAAAAA',ec='#AAAAAA' )
        pylab.xlabel('z')
        
        #pylab.ylabel(r'$\delta(z,\theta)$')
        pylab.text(-0.17,0.5,r'$\delta(z,\theta)$',
                   transform = pylab.gca().transAxes,
                   rotation=90,va='center')

        l = pylab.plot(z_delta,los.real,
                       '-')
        pylab.plot(z_delta,los.imag,
                   l[0].get_color()+'--')
        pylab.text(0.6,0.85,r'$n=%i$' % (tot_size-N_cut),
                   transform = pylab.gca().transAxes)
        pylab.text(0.6,0.75,
                   r'$v_{cut}=%.1g$' % v_cuts[i],
                   transform = pylab.gca().transAxes)
        
        pylab.subplot(321+2*i)
        cb = delta.imshow_lens_plane(i_z,
                                     gaussian_filter=0.5,
                                     cmap=pylab.cm.binary)

        cmax = cb.get_clim()[1]
        
        pylab.text(0.7,0.9,'z=%.2f' % z_delta[i_z], 
                   transform = pylab.gca().transAxes,
                   bbox = dict(facecolor='w',edgecolor='w') )
        pylab.xlabel(r'$\theta_x\ {\rm (arcmin)}$')
        pylab.ylabel(r'$\theta_y\ {\rm (arcmin)}$')


    

if __name__ == '__main__':
    N = 64
    border_size = N/16
    
    v_cuts = [1E-1,1E-2,5E-3]

    z = 0.59
    Mvir = 1E15
    x_coord = N/2
    y_coord = N/2

    z_gamma = numpy.linspace(0.08,2.0,25)
    z_delta = numpy.linspace(0.1,2.0,20)
    
    offset = 0.5+0.5j #because evaluating at r=0 produces errors
    
    PS =  ProfileSet( NFW(z, x_coord + 1j*y_coord + offset, 
                          Mvir = Mvir,
                          rvir = 1.0, alpha = 1, c = 5.0) )

    plot_three(PS,    
               Npix = N,
               z_gamma = z_gamma,
               z_delta = z_delta,
               i_x = x_coord,
               i_y = y_coord,
               i_z = numpy.searchsorted(z_delta,z),
               Ngal = 70,
               z0 = 0.57,
               sig = 0.3,
               border_size = border_size,
               border_noise = 1E3,
               rseed = 0,
               v_cuts = v_cuts)

    pylab.savefig('three_lines.eps')
    pylab.savefig('three_lines.pdf')

    if '-show' in sys.argv:
        pylab.show()
