"""
run this code to create the cfg file for the condor runs
"""

import os
import sys
sys.path.append(os.path.abspath('../'))
from shear_KL_source import params
params.load('base_params.dat')

from shear_KL_source.DES_tile.setup_utils import create_cfg_file

create_cfg_file( cfg_file = 'DES_tile.cfg',
                 alphas = (0.15,),
                 NMODES = (900,),
                 use_noise = (True,),
                 use_mask = (True,False),
                 noise_only = False,
                 compute_shear_noise = False,
                 compute_Map = True,
                 compute_Map_noise = True,
                 )

create_cfg_file( cfg_file = 'DES_tile_noiseonly.cfg',
                 alphas = (0.15,),
                 NMODES = (900,),
                 use_noise = (True,),
                 use_mask = (True,False),
                 noise_only = True,
                 compute_shear_noise = False,
                 compute_Map = True,
                 compute_Map_noise = True,
                 )


#----------------------------------------------------------------------
#  extra files with varying mask fractions and varying noise
RAlim = (11.5,12.5)
DEClim = (36.0,37.0)

create_cfg_file( cfg_file = 'DES_tile_xtra.cfg',
                 append_file = False,
                 alphas = (0,0.15,),
                 NMODES = (100,900,2000,),
                 use_noise = (True,),
                 use_mask = (True,False),
                 compute_shear_noise = False,
                 compute_Map = False,
                 compute_Map_noise = False,
                 RAlim = RAlim,
                 DEClim = DEClim,
                 )

params.load('params_f0.5.dat')

create_cfg_file( cfg_file = 'DES_tile_xtra.cfg',
                 append_file = True,
                 alphas = (0,0.15,),
                 NMODES = (100,900,2000,),
                 use_noise = (True,),
                 use_mask = (True,False),
                 compute_shear_noise = False,
                 compute_Map = False,
                 compute_Map_noise = False,
                 RAlim = RAlim,
                 DEClim = DEClim,
                 )

params.load('params_sig0.1.dat')

create_cfg_file( cfg_file = 'DES_tile_xtra.cfg',
                 append_file = True,
                 alphas = (0.15,),
                 NMODES = (900,),
                 use_noise = (True,),
                 use_mask = (True,False),
                 compute_shear_noise = False,
                 compute_Map = False,
                 compute_Map_noise = False,
                 RAlim = RAlim,
                 DEClim = DEClim,
                 )

params.load('params_sig0.1_f0.5.dat')

create_cfg_file( cfg_file = 'DES_tile_xtra.cfg',
                 append_file = True,
                 alphas = (0.15,),
                 NMODES = (900,),
                 use_noise = (True,),
                 use_mask = (True,False),
                 compute_shear_noise = False,
                 compute_Map = False,
                 compute_Map_noise = False,
                 RAlim = RAlim,
                 DEClim = DEClim,
                 )

params.load('params_sig0.1_f0.35.dat')

create_cfg_file( cfg_file = 'DES_tile_xtra.cfg',
                 append_file = True,
                 alphas = (0.15,),
                 NMODES = (900,),
                 use_noise = (True,),
                 use_mask = (True,False),
                 compute_shear_noise = False,
                 compute_Map = False,
                 compute_Map_noise = False,
                 RAlim = RAlim,
                 DEClim = DEClim,
                 )
