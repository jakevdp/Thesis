#-append the correct paths for importing Lens3D modules--------------
import sys, os
Lens3D_path = os.environ.get('LENS3D_PATH')
if Lens3D_path is None:
    raise ValueError, 'must set environment variable LENS3D_PATH'
sys.path.append(Lens3D_path)
sys.path.append(os.path.join(Lens3D_path,'3D_shear_generation'))
#--------------------------------------------------------------------

import numpy
from Simon_Taylor_method import *
from Reconstructions import *

N=128
#z_gamma = numpy.linspace(0.08,2.0,25)
#z_delta = numpy.linspace(0.1,2.0,20)
#z_gamma = numpy.linspace(0.04,2.0,50)
#z_delta = numpy.linspace(0.05,2.0,40)
z_gamma = numpy.linspace(0.02,2.0,100)
z_delta = numpy.linspace(0.025,2.0,80)

P_kd = construct_P_kd(N,N,z_gamma,z_delta)

U,S,VT = numpy.linalg.svd(P_kd.data_,full_matrices=0)

S2 = S**2
S2 /= S2.sum()

#pylab.figure(figsize=(11,4))
#pylab.axes((0.1,0.15,0.38,0.77))
pylab.figure(figsize=(6,4.5))
pylab.axes([0.15,0.15,0.75,0.75])

styles = ['-k','-b','-r','-g']

for i in range(4):
    vi = VT[i]*numpy.sign(VT[i,-2])
    pylab.plot(z_delta,vi,
               styles[i],
               lw=4-i,label = r'$n_%i \propto %.1f$' % (i+1,S[0]/S[i]) )
pylab.legend(loc=4)
pylab.xlabel('z')
pylab.ylabel(r'$\delta_{out}$')

pylab.savefig('../radial_modes.pdf')
pylab.savefig('../radial_modes.eps')

pylab.figure(figsize=(6,4.5))
pylab.axes([0.15,0.15,0.75,0.75])

irange = numpy.arange(1.,1.*len(S))
pylab.loglog(irange,S[:-1]/S[0],'-k')
#pylab.semilogy(irange,irange**-2,':k')
pylab.xlabel(r'$\rm{mode\ number\ }i$')
pylab.ylabel(r'$\sigma_i$')

pylab.savefig('../radial_sigma.pdf')
pylab.savefig('../radial_sigma.eps')

pylab.show()
