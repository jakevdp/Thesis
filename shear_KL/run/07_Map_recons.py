"""
run this code to compute the aperture mass map from the KL-reconstructed shear
"""

import os
import sys
sys.path.append(os.path.abspath('../'))
from shear_KL_source import params
params.load('base_params.dat')

from shear_KL_source.DES_tile.mass_maps import compute_Map_recons

compute_Map_recons('900_n_a0.15y')
compute_Map_recons('900_y_a0.15y')
compute_Map_recons('900_n_a0.15y_no')
compute_Map_recons('900_y_a0.15y_no')