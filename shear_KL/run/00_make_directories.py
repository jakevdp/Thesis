"""
run this code to create the directories
"""

import os
import sys
sys.path.append(os.path.abspath('../'))
from shear_KL_source import params
params.load('base_params.dat')

for param_file in ('base_params.dat',
                   'params_f0.5.dat',
                   'params_sig0.1.dat',
                   'params_sig0.1_f0.5.dat',
                   'params_sig0.1_f0.35.dat'):
    params.load(param_file)
    for p in ['scratch_dir', 'shear_in_dir', 'shear_recons_dir', 'Map_dir',
              'kappa_dir', 'mask_outdir', 'condorlog']:
        F = params[p]
        if not os.path.exists(F):
            print "mkdir %s" % F
            os.system('mkdir %s' % F)
