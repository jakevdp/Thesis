#-append the correct paths for importing Lens3D modules--------------
import sys, os
Lens3D_path = os.environ.get('LENS3D_PATH')
if Lens3D_path is None:
    raise ValueError, 'must set environment variable LENS3D_PATH'
sys.path.append(Lens3D_path)
sys.path.append(os.path.join(Lens3D_path,'3D_shear_generation'))
#--------------------------------------------------------------------

from Simon_Taylor_method import *
from generate_3D_shear import create_random_profiles
from thin_lens import *
from Lens3D_display_results import *
from Lens3D import *
from scipy.sparse.linalg import LinearOperator
from SVD_method import *
from scipy import fftpack

numpy.random.seed(0)

N = 128

dx = dy = 1.0

border_size = N/16

zrange = (0.2,1.0)
xrange = (0,N*dx)
yrange = (0,N*dy)

#select halo masses
use_discrete = False
if use_discrete:
    possible_masses = [8E14, 4E14, 2E14]
    N_heaviest = 1
    
    halo_masses = []
    M0 = possible_masses[0]
    for M in possible_masses:
        N_halos = int( numpy.ceil( N_heaviest * (M0/M)**2 ) )
        for i in range(N_halos):
            halo_masses.append(M)
            
    N_halos = len(halo_masses)
else:
    mass_range = [2E14,8E14]
    Nbins = 100
    N_halos = 22

    Mrange = numpy.linspace(min(mass_range),max(mass_range),Nbins)
    dM = Mrange[1]-Mrange[0]
    
    weight = ( 1./Mrange )**2

    weight_cuml = numpy.cumsum(weight)
    weight_cuml /= weight_cuml[-1]

    halo_masses = numpy.zeros(N_halos)

    for i in range(N_halos):
        r = numpy.random.random()
        ind = numpy.searchsorted(weight_cuml,r)
        masses[i] = Mrange[ind] + dM*(numpy.random.random()-0.5)
    

print "creating halo distribution with %i halos" % N_halos

x_coords = xrange[0] + numpy.random.random(N_halos) * (xrange[1]-xrange[0])
y_coords = yrange[0] + numpy.random.random(N_halos) * (yrange[1]-yrange[0])
z_vals   = zrange[0] + numpy.random.random(N_halos) * (zrange[1]-zrange[0])

PS = ProfileSet( *(NFW(z_vals[i],
                       x_coords[i] + 1j*y_coords[i],
                       halo_masses[i],
                       rvir = 1.0,
                       alpha = 1,
                       c = 10.0) for i in range(N_halos) ) )


z_gamma = numpy.linspace(0.08,2.0,25)
z_delta = numpy.linspace(0.1,2.0,20)

Nx = Ny = N
Nzg = len(z_gamma)
Nzd = len(z_delta)

#construct theta arrays
theta_min_x = 0
theta_max_x = dx*(Nx-1)
theta_min_y = 0
theta_max_y = dy*(Ny-1)
theta1 = numpy.linspace(theta_min_x,theta_max_x,Nx)
theta2 = numpy.linspace(theta_min_y,theta_max_y,Ny)
    
#determine noise level
Ngal = 70
z0 = 0.57
sig = 0.3

if z0 == 0:
    N_per_bin = numpy.ones(Nzg)
    N_per_bin /= Nzg
    N_per_bin *= Ngal
else:
    N_per_bin = z_gamma**2 * numpy.exp( -(z_gamma/z0)**1.5 )
    N_per_bin /= N_per_bin.sum()
    N_per_bin *= Ngal
noise = sig / numpy.sqrt(N_per_bin)
    
#extract gamma,kappa,Sigma in the correct bins
print "extracting gamma and sigma"
gamma = PS.get_gamma_vector(theta1,theta2,z_gamma)
Sigma = PS.get_Sigma_vector(theta1,theta2,z_delta,1)

#add shear noise:
# normally distributed noise, with a random phase
#print "adding shear noise"
#for i in range(Nzg):
#    R1 = numpy.random.normal(size = (Nx,Ny))
#    R2 = numpy.random.random((Nx,Ny))
#    sp = gamma.source_plane(i)
#    sp += ( noise[i] * R1 \
#            * numpy.exp(2j*numpy.pi*R2) )
            
#construct transformation matrices
#print "constructing transformation matrices"
#P_gk = construct_P_gk( theta_min_x, theta_max_x, Nx,
#                       theta_min_y, theta_max_y, Ny,
#                       z_gamma )
#P_kd = construct_P_kd(Nx,Ny,z_gamma,z_delta)
#    
#construct noise distribution
#N_los = noise**2
#N_angular = numpy.ones(Nx*Ny)

pylab.figure()
Sigma.imshow_lens_plane('all')
gamma.fieldplot_lens_plane(Nzg-1)

pylab.show()
