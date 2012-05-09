from thin_lens import *
import numpy
import pylab
import sys
sys.path.append('../')
from Lens3D import Lens3D_vector, theta_comp_to_grid, theta_grid_to_comp

N_clusters = 5

zmin = 0.1
zmax = 0.3

theta_min = -3
theta_max = 3

sig_min = 200
sig_max = 500

#numpy.random.seed(2)
z = numpy.random.random(N_clusters)*(zmax-zmin)+zmin
theta = numpy.random.random((2,N_clusters))*(theta_max-theta_min)+theta_min
theta = theta[0] + 1j*theta[1]
sig = numpy.random.random(N_clusters)*(sig_max-sig_min)+sig_min

print theta

PS = ProfileSet()
for i in range(N_clusters):
    PS.add( SIS(z[i],sig[i],theta[i]) )

z_source = 0.3
theta_1 = numpy.linspace(-3,3,30)
theta_2 = numpy.linspace(-3,3,30)
theta = theta_comp_to_grid(theta_1,theta_2)

pylab.figure()
PS.get_kappa_vector(theta_1,theta_2,(0.1,0.3) ).imshow_lens_plane(1,cmap=pylab.cm.gray,extent = theta_extent(theta))
PS.get_gamma_vector(theta_1,theta_2,(0.1,0.3) ).fieldplot_lens_plane(1,extent=(theta_1[0],theta_1[-1],theta_2[0],theta_2[-1]) )
pylab.figure()
PS.plot_kappa(theta,z_source)
pylab.figure()
PS.plot_gammakappa(theta,0.3,normalize = True)
pylab.show()
