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

from scipy.ndimage import filters

def read_halo_params():
    x = []
    y = []
    z = []
    M = []
    for line in open('halo_coordinates.dat'):
        line = line.strip()
        if line.startswith('#'):
            continue
        line = map(float,line.split())
        if len(line)==0:
            continue
        x.append(line[0])
        y.append(line[1])
        z.append(line[2])
        M.append(line[3])
    return map(numpy.asarray,(x,y,z,M))

x,y,z,M = read_halo_params()

def indices_in_field():
    return numpy.where((x > border_size*dx) & (x < (N-border_size)*dx) \
                       & (y > border_size*dy) & (y < (N-border_size)*dy) )[0]

def scatter_halos(min_circ = 300,
                  max_circ = 900):
    M_scaled = M.copy()
    M_scaled -= min(M_scaled)
    M_scaled /= max(M_scaled)

    M_scaled *= (max_circ-min_circ)
    M_scaled += min_circ

    i = indices_in_field()

    ax = pylab.axis()
    pylab.scatter(x[i],y[i],s=M_scaled[i],
                  edgecolor='r',facecolor='')
    pylab.axis(ax)

pylab.figure(figsize=(4,10))

#plot the SVD method on top
pylab.subplot(311)
delta = Lens3D_vec_from_file('delta128_0.05_svd.dat')
sig_cut = 0.05
color_min = -8#0

delta_proj = Lens3D_vector(1,Nx,Ny,delta.data.sum(0))
i_zero = numpy.where(delta_proj.vec < color_min)
delta_proj.vec[i_zero] = color_min
cb = delta_proj.imshow_lens_plane(0,gaussian_filter=1,
                                  loglevels = False,
                                  #cbargs = {'extend':'min'},
                                  cmap = pylab.cm.binary)
scatter_halos(200,600)
pylab.text(0.95,0.05,r'$v_{cut} = %.2g$' % sig_cut,
           transform=pylab.gca().transAxes,
           fontsize = 14, bbox = dict(facecolor='w',edgecolor='w'),
           horizontalalignment='right')
#pylab.xlabel(r'$\theta_x\rm{\ (arcmin)}$')
pylab.ylabel(r'$\theta_y\rm{\ (arcmin)}$')
cb.set_label(r'$\delta\rm{\ (projected,\ 0<z<2)}$')

plot_bounds(':k')
pylab.title('SVD filter',fontsize=12)

#------------------------------------------------------------
#plot transverse WF in the middle
pylab.subplot(312)
delta = Lens3D_vec_from_file('delta128_0.05_trans.dat')
alpha = 0.05
color_min = -2E-6#0

delta_proj = Lens3D_vector(1,Nx,Ny,delta.data.sum(0))
i_zero = numpy.where(delta_proj.vec < color_min)
delta_proj.vec[i_zero] = color_min
delta_proj.vec *= 1E5
cb = delta_proj.imshow_lens_plane(0,gaussian_filter=1,
                                  loglevels = False,
                                  #cbargs = {'extend':'min'},
                                  cmap = pylab.cm.binary)
pylab.text(1.05,1.0,r'$\times 10^{-5}$',
           transform = pylab.gca().transAxes,
           fontsize = 14)
scatter_halos(200,600)
pylab.text(0.95,0.05,r'$\alpha = %.2g$' % alpha,
           transform=pylab.gca().transAxes,
           fontsize = 14, bbox = dict(facecolor='w',edgecolor='w'),
           horizontalalignment='right')
pylab.ylabel(r'$\theta_y\rm{\ (arcmin)}$')
cb.set_label(r'$\delta\rm{\ (projected,\ 0<z<2)}$')

plot_bounds(':k')
pylab.title('Transverse WF',fontsize=12)

#------------------------------------------------------------
#plot radial WF on the bottom
pylab.subplot(313)
delta = Lens3D_vec_from_file('delta128_0.05_rad.dat')
alpha = 0.05
color_min = -8#0

delta_proj = Lens3D_vector(1,Nx,Ny,delta.data.sum(0))
i_zero = numpy.where(delta_proj.vec < color_min)
delta_proj.vec[i_zero] = color_min
delta_proj.vec
cb = delta_proj.imshow_lens_plane(0,gaussian_filter=1,
                                  loglevels = False,
                                  #cbargs = {'extend':'min'},
                                  cmap = pylab.cm.binary)
scatter_halos(200,600)
pylab.text(0.95,0.05,r'$\alpha = %.2g$' % alpha,
           transform=pylab.gca().transAxes,
           fontsize = 14, bbox = dict(facecolor='w',edgecolor='w'),
           horizontalalignment='right')
pylab.xlabel(r'$\theta_x\rm{\ (arcmin)}$')
pylab.ylabel(r'$\theta_y\rm{\ (arcmin)}$')
cb.set_label(r'$\delta\rm{\ (projected,\ 0<z<2)}$')

plot_bounds(':k')
pylab.title('Radial WF',fontsize=12)

############################################################

#pylab.savefig('../many_halos_compare.eps')
#pylab.savefig('../many_halos_compare.pdf')

if '-show' in sys.argv:
    pylab.show()
