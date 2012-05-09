import os
import sys

sys.path.append( os.path.abspath('../'))
from shear_KL_source import params

params.load('../run/base_params.dat')

print params.ngal
