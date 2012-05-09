#-append the correct paths for importing Lens3D modules--------------
import sys, os
Lens3D_path = os.environ.get('LENS3D_PATH')
if Lens3D_path is None:
    raise ValueError, 'must set environment variable LENS3D_PATH'
sys.path.append(Lens3D_path)
sys.path.append(os.path.join(Lens3D_path,'3D_shear_generation'))
#--------------------------------------------------------------------
from Lens3D import *
from params import *
from Simon_Taylor_method import Sigma_to_delta

from scipy.ndimage import filters
from matplotlib.ticker import NullFormatter

alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def sort_by_mass(halo_array):
    i = numpy.argsort(halo_array[:,3])
    return halo_array[i[::-1],:]

def scatter_halos(halo_file='halo_coordinates.dat',
                  min_circ = 300,
                  max_circ = 900):
    x,y,z,M = sort_by_mass( numpy.loadtxt(halo_file) ).T
    
    M_scaled = M.copy()
    M_scaled -= min(M_scaled)
    M_scaled /= max(M_scaled)

    M_scaled *= (max_circ-min_circ)
    M_scaled += min_circ

    i = numpy.where((x > border_size*dx) & (x < (N-border_size)*dx) \
                    & (y > border_size*dy) & (y < (N-border_size)*dy) )[0]

    ax = pylab.axis()
    pylab.scatter(x[i],y[i],s=M_scaled[i],
                  edgecolor='r',facecolor='')

    offset = 5
    
    for j in range(len(x[i])):
        pylab.text(x[i][j]+offset,
                   y[i][j]-offset,
                   alphabet[j],
                   fontsize=14,
                   color='r')
    
    pylab.axis(ax)

delta_vec = Lens3D_vec_from_file('delta128_0.05_rad.dat')

delta = filters.gaussian_filter(delta_vec.data.real,(0,0.5,0.5), mode='mirror')

Nz,Nx,Ny = delta.shape

pylab.figure()
proj_1 = delta.sum(0)
pylab.imshow(proj_1.T,
             origin='lower',
             aspect = proj_1.shape[0]*1./proj_1.shape[1],
             cmap = pylab.cm.binary)
pylab.colorbar()

scatter_halos()

xlim = pylab.xlim()
ylim = pylab.ylim()


######################################################################

pylab.figure(figsize=(10,6))

ax_x0 = 0.1 #0.05
ax_y0 = 0.1 #0.05

ax_xstep = 0.2 #0.3
ax_ystep = 0.2 #0.3

ax_Nx = 4 #3
ax_Ny = 4 #3

ax_list = []

coords = sort_by_mass( numpy.loadtxt('halo_coordinates.dat') )
i=0

delta_in = Sigma_to_delta(Lens3D_vec_from_file('Sigma128.dat'),
                          z_delta).data

for iy in range(ax_Ny):
    for ix in range(ax_Nx):
        ax = pylab.axes( (ax_x0+ix*ax_xstep,ax_y0+(ax_Ny-iy-1)*ax_ystep,
                                    ax_xstep,ax_ystep) )
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())

        x,y,z,M = coords[i]
        x = int(x+0.5)
        y = int(y+0.5)

        z_out = delta[:,x-1:x+2,y-1:y+2].sum(1).sum(1)
        z_true = delta_in[:,x-1:x+2,y-1:y+2].sum(1).sum(1)

        #i_z = numpy.searchsorted(z_delta,z)
        #z_true = numpy.zeros(Nz)
        #z_true[i_z] = numpy.max(z_out)

        pylab.plot(z_delta,z_out,'-k')
        pylab.fill(z_delta,z_true,'-',
                   color='#AAAAAA',fc='#AAAAAA')

        minn,maxx = min(z_out),max(z_out)

        pylab.ylim(minn-0.3*(maxx-minn),
                   2*maxx-minn)

        pylab.text(0.9,0.8,alphabet[i],
                   transform=ax.transAxes)
        
        i+=1
        


pylab.show()
