import numpy
import pylab

possible_masses = [8E14, 4E14, 2E14]
N_heaviest = 1

halo_masses = []
M0 = possible_masses[0]
for M in possible_masses:
    N_halos = int( numpy.ceil( N_heaviest * (M0/M)**2 ) )
    for i in range(N_halos):
        halo_masses.append(M)

halo_masses = numpy.asarray( halo_masses )

pylab.figure()
pylab.hist(halo_masses)

mass_range = [2E14,8E14]
Nbins = 1000
N_halos = 22

Mrange = numpy.linspace(min(mass_range),max(mass_range),Nbins)
dM = Mrange[1]-Mrange[0]


weight = ( 1./Mrange )**2

weight_cuml = numpy.cumsum(weight)
weight_cuml /= weight_cuml[-1]

masses = numpy.zeros(N_halos)

for i in range(N_halos):
    r = numpy.random.random()
    ind = numpy.searchsorted(weight_cuml,r)
    masses[i] = Mrange[ind] + dM*(numpy.random.random()-0.5)

pylab.figure()
pylab.hist(masses)

pylab.show()
