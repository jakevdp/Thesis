"""
run this code to parse the shear catalog

it will be saved in the directories specified in base_params.dat
"""

import os
import sys
sys.path.append(os.path.abspath('../'))
from shear_KL_source import params
params.load('base_params.dat')

from shear_KL_source.DES_tile.setup_utils import parse_DES_shear

parse_DES_shear( )
