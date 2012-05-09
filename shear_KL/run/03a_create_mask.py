"""
run this to create the mask tiles for the condor runs

This creates masks with different mask fractions
"""

import os
import sys
sys.path.append(os.path.abspath('../'))
from shear_KL_source import params

from shear_KL_source.DES_tile.setup_utils import create_mask_tiles

import numpy
params.load('base_params.dat')

params.load('params_f0.5.dat')
numpy.random.seed(3)
create_mask_tiles( params.mask_outdir,
                   fmask = params.fmask )

params.load('params_f0.35.dat')
numpy.random.seed(3)
create_mask_tiles( params.mask_outdir,
                   fmask = params.fmask )
