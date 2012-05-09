import numpy
import pylab

#survey size
N = 128
border_size = N/16
border_noise = 1E3

dx = dy = 1.0

#redshift binning
z_gamma = numpy.linspace(0.08,2.0,25)
z_delta = numpy.linspace(0.1,2.0,20)

Nx = Ny = N
Nzg = len(z_gamma)
Nzd = len(z_delta)

#construct theta arrays
theta_min_x = 0
theta_max_x = dx*(Nx-1)
theta_min_y = 0
theta_max_y = dy*(Ny-1)
theta1 = numpy.linspace(theta_min_x,theta_max_x,Nx)
theta2 = numpy.linspace(theta_min_y,theta_max_y,Ny)
    
#determine noise level
Ngal = 70
z0 = 0.57
sig = 0.3

if z0 == 0:
    N_per_bin = numpy.ones(Nzg)
    N_per_bin /= Nzg
    N_per_bin *= Ngal
else:
    N_per_bin = z_gamma**2 * numpy.exp( -(z_gamma/z0)**1.5 )
    N_per_bin /= N_per_bin.sum()
    N_per_bin *= Ngal
noise = sig / numpy.sqrt(N_per_bin)


def plot_bounds(color = ':w'):
    xlims = [border_size,Nx-border_size,Nx-border_size,border_size,border_size]
    ylims = [border_size,border_size,Ny-border_size,Ny-border_size,border_size]

    A = pylab.axis()
    pylab.plot(xlims,ylims,color)
    pylab.axis(A)
