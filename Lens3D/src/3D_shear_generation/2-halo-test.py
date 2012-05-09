"""
Mimicking figure 4 in Vitelli, Jain, Kapmien '09
"""

from thin_lens import *
import numpy
import pylab

theta_1 = numpy.linspace(0,4.2,60)
theta_2 = numpy.linspace(0,4.2,60)
theta = theta_comp_to_grid(theta_1,theta_2)

PS = ProfileSet( SIS(0.1,400,0.5+3.0j) ,
                 SIS(0.15, 250, 3.5+3.5j),
                 SIS(0.15, 250, 3.5+3.4j),
                 SIS(0.15, 250, 3.5+3.3j),
                 SIS(0.2, 200, 2.7+2.0j),  
                 SIS(0.2, 200, 4.3+2.0j),  
                 SIS(0.2, 100, 1.1+0.1j)   )

PS = NFW(0.1,0.5+3.0j,
         Mvir = 1E15, rvir = 200.,
         alpha = 1, c = 1.0)
PS.plot_gammakappa(theta,0.3,normalize=True,n_bars = 15)
pylab.figure()
PS.plot_Sigma(theta,0,10,numlevels = 15)
pylab.show()
