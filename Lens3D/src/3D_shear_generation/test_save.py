from thin_lens import *
import numpy
import sys
import pylab

filename = 'tmp.dat'

N_clusters = 10

zmin = 0.1
zmax = 0.3

theta_min = -3
theta_max = 3

sig_min = 200
sig_max = 500

z = numpy.random.random(N_clusters)*(zmax-zmin)+zmin
theta = numpy.random.random((2,N_clusters))*(theta_max-theta_min)+theta_min
theta = theta[0] + 1j*theta[1]
sig = numpy.random.random(N_clusters)*(sig_max-sig_min)+sig_min

print theta

PS = ProfileSet()
for i in range(N_clusters):
    PS.add( SIS(z[i],sig[i],theta[i]) )

z_out = [0.2,0.3,0.4]
theta_1 = numpy.linspace(-3,3,30)
theta_2 = numpy.linspace(-3,3,30)

PS.write_to_file(filename,
                 z_out,theta_1,theta_2)
z,theta1,theta2,Sigma,kappa,gamma = read_density_file(filename)

gamma_1 = PS.get_gamma_vector(theta1,theta2,z_out)
kappa_1 = PS.get_kappa_vector(theta1,theta2,z_out)

for i in range(3):
    pylab.figure()
    kappa.imshow_lens_plane(i)
    gamma.fieldplot_lens_plane(i)
    pylab.figure()
    kappa_1.imshow_lens_plane(i)
    gamma_1.fieldplot_lens_plane(i)

pylab.show()
