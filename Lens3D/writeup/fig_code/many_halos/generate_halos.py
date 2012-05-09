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

from params import *

pylab.ioff()

def select_redshifts(func,zmin,zmax,N,
                     dz = 0.01):
    z = numpy.arange(zmin,zmax,dz)
    f = func(z)
    f_cuml = numpy.cumsum(f)
    f_cuml /= f_cuml[-1]

    R_ind = [numpy.searchsorted(f_cuml,r) for r in numpy.random.random(N)]
    return z[R_ind] + dz * numpy.random.random(N)

def z_dist(z,z0=0.57):
    return z**2 * numpy.exp( -(z/z0)**1.5 )

numpy.random.seed(2)

N = 128

dx = dy = 1.0

border_size = N/16

zrange = (0.1,0.8)
xrange = ( 0, dx*N )
yrange = ( 0, dx*N )

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
        halo_masses[i] = Mrange[ind] + dM*(numpy.random.random()-0.5)

print "creating halo distribution with %i halos" % N_halos

x_coords = xrange[0] + numpy.random.random(N_halos) * (xrange[1]-xrange[0])
y_coords = yrange[0] + numpy.random.random(N_halos) * (yrange[1]-yrange[0])
z_vals   = select_redshifts(z_dist,zrange[0],zrange[1],N_halos)

PS = ProfileSet( *(NFW(z_vals[i],
                       x_coords[i] + 1j*y_coords[i],
                       halo_masses[i],
                       rvir = 1.0,
                       alpha = 1,
                       c = 10.0) for i in range(N_halos) ) )

OF = open('halo_coordinates.dat','w')
OF.write('#x_coord y_coord z M\n')
for i in range(len(z_vals)):
    OF.write('%.4g %.4g %.4g %.4g\n' % (x_coords[i],y_coords[i],
                                        z_vals[i],halo_masses[i]))
OF.close()
    
    

    
#extract gamma,kappa,Sigma in the correct bins
print "extracting gamma and sigma"
gamma = PS.get_gamma_vector(theta1,theta2,z_gamma)
Sigma = PS.get_Sigma_vector(theta1,theta2,z_delta,1)

gamma.write('gamma%i.dat' % N)
Sigma.write('Sigma%i.dat' % N)

pylab.figure()
Sigma.imshow_lens_plane('all')
gamma.fieldplot_lens_plane(Nzg-1,n_bars=30)

if '-show' in sys.argv:
    pylab.show()
