import os
import sys

import numpy

sys.path.append(os.path.abspath('../'))
from shear_KL_source import params
params.load('base_params.dat')

dir = params.condorlog

print "clearing %s" % dir

os.system('rm -r %s' % dir)
os.system('mkdir %s' % dir)

        
        
