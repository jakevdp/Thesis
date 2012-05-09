"""
run this code to compute the shear eigenbasis

It will be saved in the scratch directory specified in base_params
"""

import os
import sys
sys.path.append(os.path.abspath('../'))
from shear_KL_source import params
params.load('base_params.dat')

from shear_KL_source.DES_tile.setup_utils import compute_eigenbasis

compute_eigenbasis()
