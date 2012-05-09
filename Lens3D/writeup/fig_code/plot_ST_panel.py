#-append the correct paths for importing Lens3D modules--------------
import sys, os
Lens3D_path = os.environ.get('LENS3D_PATH')
if Lens3D_path is None:
    raise ValueError, 'must set environment variable LENS3D_PATH'
sys.path.append(Lens3D_path)
sys.path.append(os.path.join(Lens3D_path,'3D_shear_generation'))
#--------------------------------------------------------------------
from Reconstructions import *

from matplotlib.ticker import MultipleLocator, FuncFormatter

def plot_los(PS,
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
             alpha = 1 ):
    """
    plot_together: just like plot_three, but for two different profiles
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

    #compute delta
    delta_in = Sigma_to_delta(Sigma,z_delta)
    
    delta_rad = calculate_delta_rad(gamma,
                                    z_delta = z_delta,
                                    z_gamma = z_gamma,
                                    alpha = alpha,
                                    theta_min = theta_min,
                                    theta_max = theta_max,
                                    N = Npix,
                                    border_size = border_size,
                                    border_noise = border_noise,
                                    Ngal = Ngal,
                                    z0 = z0,
                                    sig = sig)
    
    delta_trans = calculate_delta_trans(gamma,
                                        z_delta = z_delta,
                                        z_gamma = z_gamma,
                                        alpha = alpha,
                                        theta_min = theta_min,
                                        theta_max = theta_max,
                                        N = Npix,
                                        border_size = border_size,
                                        border_noise = border_noise,
                                        Ngal = Ngal,
                                        z0 = z0,
                                        sig = sig)

    pylab.figure(figsize=(8,6.3))
    
    los_in =delta_in.line_of_sight(i_x,i_y).real
    los_trans = delta_trans.line_of_sight(i_x,i_y).copy()
    los_rad = delta_rad.line_of_sight(i_x,i_y).copy()

    #--------------------------------------------------------------------------
    pylab.subplot(221)
    #pylab.axes([0.08,0.57,0.45,0.38])
    delta_trans.vec *= 1E6
    cb = delta_trans.imshow_lens_plane(i_z,
                                 gaussian_filter=0,
                                 cmap=pylab.cm.binary)
    pylab.text(1.05,1.02,r'$\times 10^{-6}$',
               transform = pylab.gca().transAxes,
               fontsize = 14)

    pylab.text(0.75,0.9,'z=%.1f' % z_delta[i_z],
               transform = pylab.gca().transAxes,
               bbox = dict(facecolor='w',edgecolor='w') )
    pylab.xlabel(r'$\theta_x\ {\rm (arcmin)}$')
    pylab.ylabel(r'$\theta_y\ {\rm (arcmin)}$')
    
    #--------------------------------------------------------------------------
    pylab.subplot(222)
    #pylab.axes([0.6,0.57,0.35,0.38])
    pylab.fill(z_delta,los_in,
               fc = '#AAAAAA',ec='#AAAAAA' )
    l = pylab.plot(z_delta,los_trans.real,'-')
    pylab.plot(z_delta,los_trans.imag,
               l[0].get_color()+'--')
    
    pylab.xlabel('$z$')
    
    pylab.text(-0.15,0.5,r'$\delta(z,\theta)$',
               transform = pylab.gca().transAxes,
               rotation=90,va='center')
        
    pylab.text(0.3,0.9,
               r'$\rm{Transverse\ WF:}\ \alpha=%.2f$' % alpha,
               transform = pylab.gca().transAxes)
    
    #plot inset
    pylab.axes([0.70,0.70,0.18,0.13])

    los_in_copy = los_in.copy()
    i = numpy.argmax(los_in_copy)
    los_in_copy[i] = 3E-4
    
    pylab.fill(z_delta,los_in_copy,
               fc = '#AAAAAA',ec='#AAAAAA' )
    l = pylab.plot(z_delta,los_trans.real,'-')
    pylab.plot(z_delta,los_trans.imag,
               l[0].get_color()+'--')
    pylab.ylim(-5E-7,3E-6)
    pylab.xlim(0,1.6)
    pylab.gca().xaxis.set_major_locator(MultipleLocator(0.4))
    
    pylab.gca().yaxis.set_major_locator(MultipleLocator(1E-6))
    pylab.gca().yaxis.set_major_formatter(FuncFormatter(\
                lambda x,pos: '%i'%(x*1E6)))
    pylab.text(-0.05,1.02,r'$\times 10^{-6}$',
               transform = pylab.gca().transAxes)
    
    #--------------------------------------------------------------------------
    pylab.subplot(223)
    #pylab.axes([0.08,0.1,0.45,0.38])
    cb = delta_rad.imshow_lens_plane(i_z,
                                 gaussian_filter=0,
                                 cmap=pylab.cm.binary)
    
    pylab.text(0.75,0.9,'z=%.1f' % z_delta[i_z], 
               transform = pylab.gca().transAxes,
               bbox = dict(facecolor='w',edgecolor='w') )
    pylab.xlabel(r'$\theta_x\ {\rm (arcmin)}$')
    pylab.ylabel(r'$\theta_y\ {\rm (arcmin)}$')
    
    #--------------------------------------------------------------------------
    pylab.subplot(224)
    #pylab.axes([0.6,0.1,0.35,0.38])
    pylab.fill(z_delta,los_in,
               fc = '#AAAAAA',ec='#AAAAAA' )
    l = pylab.plot(z_delta,los_rad.real,'-')
    pylab.plot(z_delta,los_rad.imag,
               l[0].get_color()+'--')
    
    pylab.xlabel('$z$')
    
    pylab.text(-0.15,0.5,r'$\delta(z,\theta)$',
               transform = pylab.gca().transAxes,
               rotation=90,va='center')
        
    pylab.text(0.4,0.9,
               r'$\rm{Radial\ WF:}\ \alpha=%.2f$' % alpha,
               transform = pylab.gca().transAxes)


    

if __name__ == '__main__':
    N = 64
    border_size = N/16
    
    alpha = 0.05

    Mvir = 1E15
    x_coord = N/2
    y_coord = N/2

    z_gamma = numpy.linspace(0.08,2.0,25)
    z_delta = numpy.linspace(0.1,2.0,20)
    
    offset = 0.5+0.5j #because evaluating at r=0 produces errors

    PS =  ProfileSet( NFW(0.59, x_coord + 1j*y_coord + offset, 
                          Mvir = Mvir,
                          rvir = 1.0, alpha = 1, c = 5.0) )

    plot_los(PS,
             Npix = N,
             z_gamma = z_gamma,
             z_delta = z_delta,
             i_x = x_coord,
             i_y = y_coord,
             i_z = numpy.searchsorted(z_delta,0.59),
             Ngal = 70,
             z0 = 0.57,
             sig = 0.3,
             border_size = border_size,
             border_noise = 1E3,
             rseed = 0,
             alpha = alpha)

    pylab.savefig('los_ST.pdf')
    pylab.savefig('los_ST.eps')

    if '-show' in sys.argv:
        pylab.show()
