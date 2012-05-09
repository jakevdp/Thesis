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
from MultiAxes import MultiAxesIterator
from matplotlib import ticker

alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

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

    offset = 10
    
    for j in range(len(x[i])):
        pylab.text(x[i][j]+offset,
                   y[i][j],#-offset,
                   alphabet[j],
                   fontsize=12,
                   ha='center',va='center',
                   color='k')
    
    pylab.axis(ax)

def indices_in_field():
    return numpy.where((x > border_size*dx) & (x < (N-border_size)*dx) \
                       & (y > border_size*dy) & (y < (N-border_size)*dy) )[0]

def plot_lines_of_sight(delta_in,delta_out,z_delta,coords,
                        xmin,xmax,Nx_ax,
                        ymin,ymax,Ny_ax,
                        label_xaxis=False,
                        title=None):
    i=0

    AxesGrid = MultiAxesIterator(xmin,xmax,Nx_ax,ymin,ymax,Ny_ax)
    
    for ax in AxesGrid:
        if AxesGrid.j==Ny_ax-1:
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            for label in ax.get_xticklabels():
                label.set_rotation(0)
                label.set_fontsize(10)
            if label_xaxis:
                pylab.xlabel('z')

        if title and AxesGrid.j==0 and AxesGrid.i==1:
            pylab.title(title,fontsize=12)

                
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.yaxis.set_major_locator(ticker.NullLocator())
        
        if AxesGrid.i==Nx_ax-1:
            pylab.xlim(0,1.5)
        else:
            pylab.xlim(0,1.49)
            
        x,y,z,M = coords[i]
        x = int(x+0.5)
        y = int(y+0.5)

        z_out = delta_out[:,x-1:x+2,y-1:y+2].sum(1).sum(1)
        z_in  = delta_in[:,x-1:x+2,y-1:y+2].sum(1).sum(1)

        if max(z_in) > 100*max(abs(z_out)):
            z_in[numpy.where(z_in>0)] = 100*max(abs(z_out))

        pylab.plot(z_delta-0.5*z_delta[0],z_out,'-k')
        pylab.fill(z_delta-0.5*z_delta[0],z_in,'-',
                   color='#AAAAAA',fc='#AAAAAA')

        minn,maxx = min(z_out),max(z_out)

        pylab.ylim(minn-0.3*(maxx-minn),
                   2*maxx-minn)

        pylab.text(0.8,0.7,alphabet[i],
                   transform=ax.transAxes)

        i+=1

def output_halo_table(coords,
                      outfile='../../halo_table.tex'):
    x,y,z,M = coords.T
    
    i = numpy.where((x > border_size*dx) & (x < (N-border_size)*dx) \
                    & (y > border_size*dy) & (y < (N-border_size)*dy) )[0]

    coords = coords[i,:]
    
    OF = open(outfile,'w')
    OF.write('\\begin{tabular}{lllll}\n')
    OF.write('\\hline\n')
    OF.write('& $\\theta_x$ & $\\theta_y$ & $z$ & $M/M_\odot$ \\\\\n')
    OF.write('\\hline\n')
    for i in range(coords.shape[0]):
        x,y,z,M = coords[i]

        Mstr = '%.2g' % M
        M_N,M_E = Mstr.split('e+')
        
        OF.write('  %s & %.1f & %.1f & %.2f & $%s\\times 10^{%s}$ \\\\\n' \
                 % (alphabet[i],x,y,z,M_N,M_E) )
    OF.write('\\hline\n')
    OF.write('\\end{tabular}\n')
    OF.close()


def sort_by_mass(halo_array):
    i = numpy.argsort(halo_array[:,3])
    return halo_array[i[::-1],:]
        
coords = sort_by_mass( numpy.loadtxt('halo_coordinates.dat') )

output_halo_table(coords)

exit()

delta_in = Sigma_to_delta(Lens3D_vec_from_file('Sigma128.dat'),
                          z_delta).data

Nx_ax=3
Ny_ax=4

xmin = 0.55
xmax = 0.93

ymin1=0.66
ymax1=0.90

ymin2=0.38
ymax2=0.62

ymin3=0.1
ymax3=0.34

#--------------------------------------------------

pylab.figure(figsize=(8,10))

#plot the SVD method on top
pylab.subplot(321)
delta = Lens3D_vec_from_file('delta128_0.05_svd.dat')
sig_cut = 0.05
color_min = -8

delta_proj = Lens3D_vector(1,Nx,Ny,delta.data.sum(0))
i_zero = numpy.where(delta_proj.vec < color_min)
delta_proj.vec[i_zero] = color_min
cb = delta_proj.imshow_lens_plane(0,gaussian_filter=1,
                                  loglevels = False,
                                  #cbargs = {'extend':'min'},
                                  cmap = pylab.cm.binary)
scatter_halos('halo_coordinates.dat',200,200)#600)
pylab.text(0.95,0.05,r'$v_{cut} = %.2g$' % sig_cut,
           transform=pylab.gca().transAxes,
           fontsize = 14, bbox = dict(facecolor='w',edgecolor='w'),
           horizontalalignment='right')
#pylab.xlabel(r'$\theta_x\rm{\ (arcmin)}$')
pylab.ylabel(r'$\theta_y\rm{\ (arcmin)}$')
cb.set_label(r'$\delta\rm{\ (projected,\ 0<z<2)}$')

plot_bounds(':k')
pylab.title('SVD filter',fontsize=12)

plot_lines_of_sight(delta_in,delta.data,z_delta,coords,
                    xmin,xmax,Nx_ax,
                    ymin1,ymax1,Ny_ax,
                    title='SVD filter')

#------------------------------------------------------------
#plot transverse WF in the middle
pylab.subplot(323)
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
scatter_halos('halo_coordinates.dat',200,200)#600)
pylab.text(0.95,0.05,r'$\alpha = %.2g$' % alpha,
           transform=pylab.gca().transAxes,
           fontsize = 14, bbox = dict(facecolor='w',edgecolor='w'),
           horizontalalignment='right')
pylab.ylabel(r'$\theta_y\rm{\ (arcmin)}$')
cb.set_label(r'$\delta\rm{\ (projected,\ 0<z<2)}$')

plot_bounds(':k')
pylab.title('Transverse WF',fontsize=12)

plot_lines_of_sight(delta_in,delta.data,z_delta,coords,
                    xmin,xmax,Nx_ax,
                    ymin2,ymax2,Ny_ax,
                    title='Transverse WF')

#------------------------------------------------------------
#plot radial WF on the bottom
pylab.subplot(325)
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
scatter_halos('halo_coordinates.dat',200,200)#600)
pylab.text(0.95,0.05,r'$\alpha = %.2g$' % alpha,
           transform=pylab.gca().transAxes,
           fontsize = 14, bbox = dict(facecolor='w',edgecolor='w'),
           horizontalalignment='right')
pylab.xlabel(r'$\theta_x\rm{\ (arcmin)}$')
pylab.ylabel(r'$\theta_y\rm{\ (arcmin)}$')
cb.set_label(r'$\delta\rm{\ (projected,\ 0<z<2)}$')

plot_bounds(':k')
pylab.title('Radial WF',fontsize=12)

plot_lines_of_sight(delta_in,delta.data,z_delta,coords,
                    xmin,xmax,Nx_ax,
                    ymin3,ymax3,Ny_ax,
                    label_xaxis=True,
                    title = 'Radial WF')






############################################################

pylab.savefig('../many_halos_los.eps')
pylab.savefig('../many_halos_los.pdf')

if '-show' in sys.argv:
    pylab.show()
