"""
run this to create the mask tiles for the condor runs
"""

import os
import sys
sys.path.append(os.path.abspath('../'))
from shear_KL_source import params
params.load('base_params.dat')

from shear_KL_source.DES_tile.setup_utils import create_mask_tiles

import numpy
numpy.random.seed(3)

create_mask_tiles( params.mask_outdir,
                   fmask = params.fmask )
