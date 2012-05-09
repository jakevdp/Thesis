import pylab
import numpy
from matplotlib.ticker import NullFormatter,NullLocator
from matplotlib.colors import LinearSegmentedColormap
import sys
import os
import matplotlib

from scipy.ndimage import filters

sys.path.append(os.path.abspath('../'))

from shear_KL_source.DES_tile.tools import get_from_parsed, get_random_field
from shear_KL_source.tools import gamma_to_kappa
from shear_KL_source import params
params.load('../run/base_params.dat')

#define color maps
BkBuW = LinearSegmentedColormap('BkBuW',
                                {'red':   [(0.0,  0.0, 0.0),
                                           (0.5,  0.0, 0.0),
                                           (1.0,  1.0, 1.0)],
                                 
                                 'green': [(0.0,  0.0, 0.0),
                                           (0.25, 0.0, 0.0),
                                           (0.75, 1.0, 1.0),
                                           (1.0,  1.0, 1.0)],
                                 
                                 'blue':  [(0.0,  0.0, 0.0),
                                           (1.0,  1.0, 1.0),
                                           (1.0,  1.0, 1.0)]} )
BkRdW = LinearSegmentedColormap('BkRdW',
                                {'blue':   [(0.0,  0.0, 0.0),
                                            (0.5,  0.0, 0.0),
                                            (1.0,  1.0, 1.0)],
                                 
                                 'green': [(0.0,  0.0, 0.0),
                                           (0.25, 0.0, 0.0),
                                           (0.75, 1.0, 1.0),
                                           (1.0,  1.0, 1.0)],
                                 
                                 'red':  [(0.0,  0.0, 0.0),
                                          (0.5,  1.0, 1.0),
                                          (1.0,  1.0, 1.0)]} )

class multizoom(object):
    def __init__(self,figsize = (8,12)):
        self.figsize = figsize
        self.figaxes = None

        self.tuples = ( (0,0,1,1),
                        (0.07,0.719,0.35,0.233333),
                        (0.345,0.755,0.35,0.233333),
                        (0.62,0.686,0.35,0.233333),
                        (0.07,0.386,0.35,0.233333),
                        (0.345,0.422,0.35,0.233333),
                        (0.62,0.353,0.35,0.233333),
                        (0.07,0.053,0.35,0.233333),
                        (0.345,0.087,0.35,0.233333),
                        (0.62,0.02,0.35,0.233333) )

        self.Nax = len(self.tuples)
        
        self.ax = [None for i in range(self.Nax)]

        self.fig_lines = []

        self.draw()

    def draw(self):
        self.fig = pylab.figure(figsize=self.figsize)

        for i in range(1,len(self.tuples)):
            self.ax[i] = pylab.axes(self.tuples[i])

        for i in range(1,self.Nax):
            if (i-1)%3:
                self.ax[i].xaxis.set_major_formatter(NullFormatter())
                self.ax[i].yaxis.set_major_formatter(NullFormatter())
        
        self.ax[0] = pylab.axes(self.tuples[0], axisbg=(1,1,1,0))
        self.ax[0].xaxis.set_major_locator(NullLocator())
        self.ax[0].yaxis.set_major_locator(NullLocator())

    def __get_fig_coord(self,i,xval,yval):
        """
        for axex number i=1,6 determine the figure coordinate at xval,yval
        """
        compTrans = self.ax[i].transData + self.fig.transFigure.inverted()
        return compTrans.transform([xval,yval])

    def draw_fig_line(self,i1,x1,y1,
                      i2,corner,
                      *args,**kwargs):
        if 'R' in corner.upper():
            x2 = self.tuples[i2][0] + self.tuples[i2][2]
        elif 'L' in corner.upper():
            x2 = self.tuples[i2][0]
        else:
            raise ValueError

        if 'B' in corner.upper():
            y2 = self.tuples[i2][1]
        elif 'U' in corner.upper():
            y2 = self.tuples[i2][1]+self.tuples[i2][3]
        
        x1,y1 = self.__get_fig_coord(i1,x1,y1)

        self.ax[0].plot( (x1,x2), (y1,y2), *args, **kwargs )
        self.ax[0].axis([0,1,0,1])

    def draw_fig_line_cutoff(self,i1,x1,y1,
                             i2,corner,
                             hard_xcut,
                             *args,**kwargs):
        if 'R' in corner.upper():
            x2 = self.tuples[i2][0] + self.tuples[i2][2]
        elif 'L' in corner.upper():
            x2 = self.tuples[i2][0]
        else:
            raise ValueError

        if 'B' in corner.upper():
            y2 = self.tuples[i2][1]
        elif 'U' in corner.upper():
            y2 = self.tuples[i2][1]+self.tuples[i2][3]
        
        x1,y1 = self.__get_fig_coord(i1,x1,y1)

        if y1<y2 and 'B' in corner.upper(): #no cutoff
            pass
        elif y1>y2 and 'U' in corner.upper(): #no cutoff
            pass
        else:
            dx = x2-x1
            dy = y2-y1
            
            x2 = self.tuples[i2][0]
            y2 = y1 + dy * (x2-x1) * 1./dx

        if (hard_xcut is not None) and (x2 > hard_xcut):
            dx = x2-x1
            dy = y2-y1
            
            x2 = hard_xcut
            y2 = y1 + dy * (x2-x1) * 1./dx
            
        self.ax[0].plot( (x1,x2), (y1,y2), *args, **kwargs )
        self.ax[0].axis([0,1,0,1])

    def draw_box(self,i,tup,hard_xcut=None,*args,**kwargs):
        x = [ tup[0],
              tup[0]+tup[2],
              tup[0]+tup[2],
              tup[0],
              tup[0] ]
        y = [ tup[1],
              tup[1],
              tup[1]+tup[3],
              tup[1]+tup[3],
              tup[1] ]

        xlim = self.ax[i].get_xlim()
        ylim = self.ax[i].get_ylim()
        
        self.ax[i].plot(x,y,*args,**kwargs)

        self.ax[i].set_xlim(xlim)
        self.ax[i].set_ylim(ylim)

        self.draw_fig_line(i,tup[0],tup[1]+tup[3],i+1,
                           'UL',
                           *args,**kwargs)
        self.draw_fig_line(i,tup[0],tup[1],i+1,
                           'BL',
                           *args,**kwargs)
        self.draw_fig_line_cutoff(i,tup[0]+tup[2],tup[1],i+1,
                                  'BR',hard_xcut,
                                  *args,**kwargs)
        self.draw_fig_line_cutoff(i,tup[0]+tup[2],tup[1]+tup[3],i+1,
                                  'UR',hard_xcut,
                                  *args,**kwargs)

def imshow_kappa_in_bounds(ax,kappa,bounds,filter=None,cmap=BkRdW,
                           use_full_clim = True):
    """
    bounds is (RAmin,DECmin,RAmax,DECmax)
    """
    if filter:
        kappa = filters.gaussian_filter(kappa,filter/params.dtheta)

    clim = (numpy.min(kappa),numpy.max(kappa))

    RAlim = params.RAlim
    DEClim = params.DEClim

    ixmin = int( numpy.floor( (bounds[0]-params.RAlim[0])\
                              *60./params.dtheta ) )
    ixmax = int( numpy.ceil( (bounds[0]+bounds[2]-params.RAlim[0])\
                             *60./params.dtheta ) )

    iymin = int( numpy.floor( (bounds[1]-params.DEClim[0])\
                              *60./params.dtheta ) )
    iymax = int( numpy.ceil( (bounds[1]+bounds[3]-params.DEClim[0])\
                             *60./params.dtheta ) )

    if True:
        extent = [RAlim[0]+ixmin*params.dtheta/60.,
                  RAlim[0]+ixmax*params.dtheta/60.,
                  DEClim[0]+iymin*params.dtheta/60.,
                  DEClim[0]+iymax*params.dtheta/60.]
        
        kappa = kappa[ixmin:ixmax,
                      iymin:iymax]
    else:
        extent = RAlim+DEClim

    IM = ax.imshow(kappa.T,
                   origin='lower',
                   aspect='auto',
                   extent=extent,
                   cmap=cmap)
    ax.set_xlim(bounds[0],bounds[0]+bounds[2])
    ax.set_ylim(bounds[1],bounds[1]+bounds[3])
    if use_full_clim:
        IM.set_clim(clim)

    return IM


def plot_kappa_multires(kappa_perfect,
                        kappa_noisy,
                        kappa_KL,
                        bounds1,bounds2,bounds3,
                        filters=[4,2,1],
                        common_clim = False,
                        use_full_clim = False):
    MZ = multizoom()

    kappas = [kappa_perfect,
              kappa_noisy,
              kappa_KL]

    bounds = [bounds1,
              bounds2,
              bounds3]

    IM = []

    for i in range(3):
        for j in range(3):
            i_ax = 1+3*i+j
            ax = MZ.ax[i_ax]
            im = imshow_kappa_in_bounds( ax,
                                         kappas[i],
                                         bounds[j],
                                         filter=filters[j],
                                         use_full_clim = use_full_clim)
            if filters[j]>0:
                ftxt = r'$f=%i^\prime$' % filters[j]
            else:
                ftxt = '(no filter)'
            ax.text(0.02,0.98,ftxt,
                    bbox = dict(facecolor='w',edgecolor='w',alpha=0.75),
                    ha = 'left', va='top',
                    transform = ax.transAxes)
                

            IM.append(im)

        MZ.ax[3*i+1].set_xlabel('RA (deg)')
        MZ.ax[3*i+1].set_ylabel('DEC (deg)')

        MZ.draw_box(3*i+1,
                    bounds[1],
                    hard_xcut=MZ.tuples[3*i+3][0],
                    c='#999999')
        MZ.draw_box(3*i+2,
                    bounds[2],
                    hard_xcut=MZ.tuples[3*i+3][0],
                    c='#999999')

    if common_clim:
        IM[1].set_clim(IM[0].get_clim())
        IM[2].set_clim(IM[0].get_clim())

    for i in (1,2):
        for j in range(3):
            IM[3*i+j].set_clim(IM[j].get_clim())

    pylab.figtext(MZ.tuples[1][0]+0.03,
                  MZ.tuples[1][1]+MZ.tuples[1][3]+0.01,
                  r"$\rm{Noiseless\ Shear}$",
                  fontsize = 14)

    pylab.figtext(MZ.tuples[4][0]+0.03,
                  MZ.tuples[4][1]+MZ.tuples[1][3]+0.01,
                  r"$\rm{Noisy\ Shear}$",
                  fontsize = 14)

    pylab.figtext(MZ.tuples[7][0]+0.03,
                  MZ.tuples[7][1]+MZ.tuples[1][3]+0.01,
                  r"$\rm{KL}\ (n=900,\ \alpha=0.15)$",
                  fontsize = 14)

    for i in range(1,MZ.Nax):
        for child in MZ.ax[i].get_children():
            if isinstance(child, matplotlib.spines.Spine):
                child.set_color('#999999') 
    for child in MZ.ax[0].get_children():
        if isinstance(child, matplotlib.spines.Spine):
            child.set_color('w')


def main_1():
    nmodes = 900
    alpha = 0.15
    
    kappa_perfect = numpy.load(os.path.join(params.shear_recons_dir,
                                            'kappa_perfect.npz') )['kappa']
    kappa_KL = numpy.load(os.path.join(params.shear_recons_dir,
                                       'kappa_%i_y_a%.2fy.npz' % \
                                       (nmodes,alpha)) )['kappa']
    kappa_noisy = numpy.load(os.path.join(params.shear_recons_dir,
                                            'kappa_noisy.npz') )['kappa']
    kappa_masked = numpy.load(os.path.join(params.shear_recons_dir,
                                            'kappa_masked.npz') )['kappa']
    plot_kappa_multires(kappa_perfect.real,
                        kappa_masked.real,
                        kappa_KL.real,
                        (10,35,15,15),
                        (13,36,4,4),
                        (13.5,37.25,1,1),
                        (4,2,1) )
if __name__ == '__main__':
    main_1()
    pylab.savefig('figs/fig06_kappa_multires2.pdf')
    pylab.savefig('figs/fig06_kappa_multires2.eps')
    pylab.show()
