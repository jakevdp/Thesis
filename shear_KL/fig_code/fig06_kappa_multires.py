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
    def __init__(self,figsize = (10,10)):
        self.figsize = figsize
        self.ax = [None for i in range(7)]
        self.figaxes = None

        self.tuples = ( (0,0,1,1),
                        (0.07,0.58,0.35,0.35),
                        (0.345,0.63,0.35,0.35),
                        (0.62,0.53,0.35,0.35),
                        (0.07,0.08,0.35,0.35),
                        (0.345,0.13,0.35,0.35),
                        (0.62,0.03,0.35,0.35) )

        self.fig_lines = []

        self.draw()

    def draw(self):
        self.fig = pylab.figure(figsize=self.figsize)

        for i in range(1,7):
            self.ax[i] = pylab.axes(self.tuples[i])

        for i in (2,3,5,6):
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

def imshow_kappa_in_bounds(ax,kappa,bounds,filter=None,cmap=BkRdW):
    """
    bounds is (RAmin,DECmin,RAmax,DECmax)
    """
    if filter:
        kappa = filters.gaussian_filter(kappa,filter/params.dtheta)

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

    return IM


def plot_kappa_multires(kappa_perfect,kappa_KL,
                        bounds1,bounds2,bounds3,
                        filters=[4,2,1]):
    MZ = multizoom()
    
    IM1 = imshow_kappa_in_bounds( MZ.ax[1],
                                  kappa_perfect,
                                  bounds1,
                                  filter=filters[0])
    MZ.ax[1].text(0.02,0.98,r'$f=%i^\prime$' % filters[0],
                  bbox = dict(facecolor='w',edgecolor='w',alpha=0.75),
                  ha = 'left', va='top',
                  transform = MZ.ax[1].transAxes)
    MZ.ax[1].set_xlabel('RA (deg)')
    MZ.ax[1].set_ylabel('DEC (deg)')
                 
    
    IM2 = imshow_kappa_in_bounds( MZ.ax[2],
                                  kappa_perfect,
                                  bounds2,
                                  filter=filters[1])
    MZ.draw_box(1,bounds2,hard_xcut=MZ.tuples[3][0],c='#999999')
    MZ.ax[2].text(0.02,0.98,r'$f=%i^\prime$' % filters[1],
                  bbox = dict(facecolor='w',edgecolor='w',alpha=0.75),
                  ha = 'left', va='top',
                  transform = MZ.ax[2].transAxes)
    
    IM3 = imshow_kappa_in_bounds( MZ.ax[3],
                                  kappa_perfect,
                                  bounds3,
                                  filter=filters[2])
    MZ.draw_box(2,bounds3,c='#999999')
    MZ.ax[3].text(0.02,0.98,r'$f=%i^\prime$' % filters[2],
                  bbox = dict(facecolor='w',edgecolor='w',alpha=0.75),
                  ha = 'left', va='top',
                  transform = MZ.ax[3].transAxes)
    
    IM4 = imshow_kappa_in_bounds( MZ.ax[4],
                                  kappa_KL,
                                  bounds1,
                                  filter=filters[0])
    MZ.ax[4].text(0.02,0.98,r'$f=%i^\prime$' % filters[0],
                  bbox = dict(facecolor='w',edgecolor='w',alpha=0.75),
                  ha = 'left', va='top',
                  transform = MZ.ax[4].transAxes)
    MZ.ax[4].set_xlabel('RA (deg)')
    MZ.ax[4].set_ylabel('DEC (deg)')
    
    IM5 = imshow_kappa_in_bounds( MZ.ax[5],
                                  kappa_KL,
                                  bounds2,
                                  filter=filters[1])
    MZ.draw_box(4,bounds2,hard_xcut=MZ.tuples[3][0],c='#999999')
    MZ.ax[5].text(0.02,0.98,r'$f=%i^\prime$' % filters[1],
                  bbox = dict(facecolor='w',edgecolor='w',alpha=0.75),
                  ha = 'left', va='top',
                  transform = MZ.ax[5].transAxes)
    
    IM6 = imshow_kappa_in_bounds( MZ.ax[6],
                                  kappa_KL,
                                  bounds3,
                                  filter=filters[2])
    MZ.draw_box(5,bounds3,c='#999999')
    MZ.ax[6].text(0.02,0.98,r'$f=%i^\prime$' % filters[2],
                  bbox = dict(facecolor='w',edgecolor='w',alpha=0.75),
                  ha = 'left', va='top',
                  transform = MZ.ax[6].transAxes)

    IM4.set_clim(IM1.get_clim())
    IM5.set_clim(IM2.get_clim())
    IM6.set_clim(IM3.get_clim())

    pylab.figtext(MZ.tuples[1][0]+0.03,
                  MZ.tuples[1][1]+MZ.tuples[1][3]+0.01,
                  r"$\rm{Noiseless\ Shear}$",
                  fontsize = 14)

    pylab.figtext(MZ.tuples[4][0]+0.03,
                  MZ.tuples[4][1]+MZ.tuples[1][3]+0.01,
                  r"$\rm{KL}\ (n=900,\ \alpha=0.15)$",
                  fontsize = 14)

    for i in range(1,7):
        for child in MZ.ax[i].get_children():
            if isinstance(child, matplotlib.spines.Spine):
                child.set_color('#999999') 
    for child in MZ.ax[0].get_children():
        if isinstance(child, matplotlib.spines.Spine):
            child.set_color('w') 

if __name__ == '__main__':

    nmodes = 900
    alpha = 0.15
    
    kappa_perfect = numpy.load(os.path.join(params.shear_recons_dir,
                                            'kappa_perfect.npz') )['kappa']
    kappa_KL = numpy.load(os.path.join(params.shear_recons_dir,
                                       'kappa_%i_y_a%.2fy.npz' % \
                                       (nmodes,alpha)) )['kappa']
    kappa_noisy = numpy.load(os.path.join(params.shear_recons_dir,
                                            'kappa_noisy.npz') )['kappa']
    plot_kappa_multires(kappa_perfect.real,
                        kappa_KL.real,
                        (10,35,15,15),
                        (13,36,4,4),
                        (13.5,37.25,1,1),
                        (4,2,1) )

    pylab.savefig('figs/fig06_kappa_multires.pdf')
    pylab.savefig('figs/fig06_kappa_multires.eps')
    
    pylab.show()
