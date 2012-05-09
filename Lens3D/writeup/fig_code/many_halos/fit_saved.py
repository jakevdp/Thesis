#-append the correct paths for importing Lens3D modules--------------
import sys, os
Lens3D_path = os.environ.get('LENS3D_PATH')
if Lens3D_path is None:
    raise ValueError, 'must set environment variable LENS3D_PATH'
sys.path.append(Lens3D_path)
sys.path.append(os.path.join(Lens3D_path,'3D_shear_generation'))
#--------------------------------------------------------------------

from Reconstructions import *
from params import *

numpy.random.seed(4)

#define filters and filtering levels
FILTS = {'svd' : [1E-1,5E-2,1E-2,5E-3],
         'rad' : [5E-2,1E-2],
         'trans' : [5E-2,1E-2]}

FUNCS = {'svd':calculate_delta_SVD,
         'rad':calculate_delta_rad,
         'trans':calculate_delta_trans}

#extract shear information
gamma = Lens3D_vec_from_file('gamma%i.dat' % N)
add_noise_to_gamma(gamma,
                   z_gamma = z_gamma,
                   z0 = z0,
                   sig = sig,
                   Ngal = Ngal)

for filt in FILTS:
    filt_levels = FILTS[filt]
    filt_func = FUNCS[filt]
    
    for filt_level in filt_levels:
        delta = filt_func(gamma,
                          z_delta,
                          z_gamma,
                          filt_level,
                          theta_min = theta_min_x,
                          theta_max = theta_max_x,
                          N = Nx,
                          border_size = border_size,
                          border_noise = border_noise,
                          Ngal = Ngal,
                          z0 = z0,
                          sig = sig)
        if filt=='svd':
            delta = delta[0]

        delta.write('delta%i_%.1g_%s.dat' % (N,filt_level,filt))

        
    
