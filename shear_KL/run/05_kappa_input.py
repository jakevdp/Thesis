"""
run this code to compute the convergence map from the input shear
"""

import os
import sys
sys.path.append(os.path.abspath('../'))
from shear_KL_source import params
params.load('base_params.dat')

from shear_KL_source.DES_tile.mass_maps import compute_kappa_input

compute_kappa_input(noisy=True,   #use noisy shear
                    perfect=True, #use noiseless shear
                    true=True,    #use true kappa
                    masked=True ) #use masked kappa
