"""
run this code to compute the aperture mass map from the input shear
"""

import os
import sys
sys.path.append(os.path.abspath('../'))
from shear_KL_source import params
params.load('base_params.dat')

from shear_KL_source.DES_tile.mass_maps import compute_Map_input

for usemask in ['y','n']:
    for normed in [True,False]:
        if usemask=='n' and normed: continue
        for add_signal in (True, False):
            for add_noise in (True, False):
                if add_signal == False and add_noise == False:
                    continue
                compute_Map_input(add_signal = add_signal,
                                  add_noise = add_noise,
                                  usemask = usemask,
                                  normed = normed)
