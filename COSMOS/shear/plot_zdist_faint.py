import os

import numpy as np

import pylab

from params import COSMOS_DIR

if 0:
    zdist_file = os.path.join(COSMOS_DIR,
                              'z_cos30', 'cosmos_zdist_faint_w0.asc')
    X = np.loadtxt(zdist_file)
    pylab.plot(X[:,0], X[:,1])
    
    zdist_file = os.path.join(COSMOS_DIR,
                              'z_cos30', 'cosmos_zdist_faint_w1.asc')
    X = np.loadtxt(zdist_file)
    pylab.plot(X[:,0], X[:,1])

else:
    zdist_file = os.path.join(COSMOS_DIR,
                              'zdist_bright_plus_faint_w0_zprob0.asc')
    X = np.loadtxt(zdist_file)
    pylab.plot(X[:,0], X[:,1])
    
    zdist_file = os.path.join(COSMOS_DIR,
                              'zdist_bright_plus_faint_w0.asc')
    X = np.loadtxt(zdist_file)
    pylab.plot(X[:,0], X[:,1])
    
pylab.show()
