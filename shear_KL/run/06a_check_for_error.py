import os
import sys

import numpy

sys.path.append(os.path.abspath('../'))
from shear_KL_source import params
params.load('base_params.dat')

dir = params.condorlog

print "searching %s" % dir

for F in os.listdir(dir):
    if F.endswith('err'):
        F = os.popen('wc %s' % (os.path.join(dir, F))).readlines()
        line = F[0].split()
        if line[0] == '0' or line[0] == '1024':
            pass
        else:
            print F
        
        
