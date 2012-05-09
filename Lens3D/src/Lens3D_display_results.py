import numpy
import pylab
from Lens3D import *
from scipy import interpolate

def plot_vertical(x,*args,**kwargs):
    xlim = pylab.xlim()
    ylim = pylab.ylim()
    pylab.plot([x,x],ylim,*args,**kwargs)

    x_offset = 0.05*(xlim[1]-xlim[0])
    y_offset = 0.05*(ylim[1]-ylim[0])

    pylab.text(x+x_offset,ylim[0]+y_offset,"x = %.2f" % x)
    pylab.ylim(ylim)

def plot_horizontal(y,*args,**kwargs):
    xlim = pylab.xlim()
    pylab.plot(xlim,[y,y],*args,**kwargs)
    pylab.xlim(xlim)

def find_spline_max(x,y,dx=0.01):
    """
    use cubic spline interpolation to find the value of x where y is
    maximized
    """
    i = numpy.argmax(y)
    tck = interpolate.splrep(x,y)
    try:
        xf = numpy.arange(x[i-1],x[i+1],dx)
        yf = interpolate.splev(xf,tck)
        return xf[ numpy.argmax(yf) ]
    except:
        return x[i]

class Lens3D_display_results:
    def __init__(self,
                 z_gamma,
                 z_delta,
                 dtheta1,
                 dtheta2,
                 delta_out = None,
                 gamma_in = None,
                 kappa_in = None,
                 Sigma_in = None):
        self.z_gamma = z_gamma
        self.z_delta = z_delta
        self.dtheta1 = dtheta1
        self.dtheta2 = dtheta2
        self.delta_out = delta_out
        self.gamma_in = gamma_in
        self.kappa_in = kappa_in
        self.Sigma_in = Sigma_in

    def plot_los(self,i_x,i_y,i_z,
                 border=0,gaussian_filter=None):
        dmin = numpy.min(self.delta_out.data.real)
        dmax = numpy.max(self.delta_out.data.real)
        Smin = numpy.min(self.Sigma_in.data.real)
        Smax = numpy.max(self.Sigma_in.data.real)

        dnorm = matplotlib.colors.Normalize(dmin,dmax)
        Snorm = matplotlib.colors.Normalize(Smin,Smax)

        xlim = (border-0.5,self.delta_out.Nx-border-0.5)
        ylim = (border-0.5,self.delta_out.Ny-border-0.5)

        rect_x = numpy.array([i_x,i_x+1,i_x+1,i_x,i_x])#-0.5 
        rect_y = numpy.array([i_y,i_y,i_y+1,i_y+1,i_y])#-0.5
        
        fig = pylab.figure( figsize=(10,8) )
        #first row is Re[delta]
        pylab.subplot(222)
        self.delta_out.imshow_lens_plane(i_z,'r',loglevels=False,
                                         norm=dnorm,
                                         gaussian_filter=gaussian_filter)
        pylab.plot(rect_x,rect_y,'-r')
        pylab.xlim(xlim)
        pylab.ylim(ylim)
        pylab.title(r'$Re[\delta_{out}]\ (z=%.2f)$' % self.z_delta[i_z])

        pylab.subplot(221)
        P = self.delta_out.plot_los(i_x,i_y,'r',z_array=self.z_delta)
        pylab.ylim(dmin,dmax)
        plot_vertical( find_spline_max(P[0].get_xdata(),P[0].get_ydata() ),
                       'r')
        pylab.title(r'$Re[\delta_{out}]\ \theta=(%.1f,%.1f)$' %
                    (i_x*self.dtheta1,i_y*self.dtheta2) )

        #second row is Sigma
        pylab.subplot(224)
        self.Sigma_in.imshow_lens_plane(i_z,'r',loglevels=False,
                                norm=Snorm)
        pylab.plot(rect_x,rect_y,'-r')
        pylab.xlim(xlim)
        pylab.ylim(ylim)
        pylab.title(r'$\Sigma_{in}\ (z=%.2f)\ [M_\odot /\rm{Mpc}]$' \
                        % self.z_delta[i_z])

        pylab.subplot(223)
        P = self.Sigma_in.plot_los(i_x,i_y,'r',z_array=self.z_delta)
        pylab.ylim(Smin,Smax)
        pylab.title(r'$\Sigma_{in}\ \theta=(%.1f,%.1f)$' %
                    (i_x*self.dtheta1,i_y*self.dtheta2) )

        return fig

    def plot_los_withB(self,i_x,i_y,i_z,
                       border=0,gaussian_filter=None):
        dmin = numpy.min(self.delta_out.data.real)
        dmax = numpy.max(self.delta_out.data.real)
        Smin = numpy.min(self.Sigma_in.data.real)
        Smax = numpy.max(self.Sigma_in.data.real)

        dnorm = matplotlib.colors.Normalize(dmin,dmax)
        Snorm = matplotlib.colors.Normalize(Smin,Smax)

        xlim = (border-0.5,self.delta_out.Nx-border-0.5)
        ylim = (border-0.5,self.delta_out.Ny-border-0.5)

        rect_x = [i_x,i_x+1,i_x+1,i_x,i_x] 
        rect_y = [i_y,i_y,i_y+1,i_y+1,i_y]
        
        fig = pylab.figure( figsize=(10,8) )
        #first row is Re[delta]
        pylab.subplot(322)
        self.delta_out.imshow_lens_plane(i_z,'r',loglevels=False,
                                         norm=dnorm,
                                         gaussian_filter=gaussian_filter)
        pylab.plot(rect_x,rect_y,'-r')
        pylab.xlim(xlim)
        pylab.ylim(ylim)
        pylab.title(r'$Re[\delta_{out}]\ (z=%.2f)$' % self.z_delta[i_z])

        pylab.subplot(321)
        self.delta_out.plot_los(i_x,i_y,'r',z_array=self.z_delta)
        pylab.ylim(dmin,dmax)
        pylab.title(r'$Re[\delta_{out}]\ \theta=(%.1f,%.1f)$' %
                    (i_x*self.dtheta1,i_y*self.dtheta2) )

        #second row is Im[delta]
        pylab.subplot(324)
        self.delta_out.imshow_lens_plane(i_z,'i',loglevels=False,
                                         norm=dnorm,
                                         gaussian_filter=gaussian_filter)
        pylab.plot(rect_x,rect_y,'-r')
        pylab.xlim(xlim)
        pylab.ylim(ylim)
        pylab.title(r'$Im[\delta_{out}]\ (z=%.2f)$' % self.z_delta[i_z])

        pylab.subplot(323)
        self.delta_out.plot_los(i_x,i_y,'i',z_array=self.z_delta)
        pylab.ylim(dmin,dmax)
        pylab.title(r'$Im[\delta_{out}]\ \theta=(%.1f,%.1f)$' %
                    (i_x*self.dtheta1,i_y*self.dtheta2) )

        #third row is Sigma
        pylab.subplot(326)
        self.Sigma_in.imshow_lens_plane(i_z,'r',loglevels=False,
                                norm=Snorm)
        pylab.plot(rect_x,rect_y,'-r')
        pylab.xlim(xlim)
        pylab.ylim(ylim)
        pylab.title(r'$\Sigma_{in}\ (z=%.2f)\ [M_\odot /\rm{Mpc}]$' \
                        % self.z_delta[i_z])

        pylab.subplot(325)
        self.Sigma_in.plot_los(i_x,i_y,'r',z_array=self.z_delta)
        pylab.ylim(Smin,Smax)
        pylab.title(r'$\Sigma_{in}\ \theta=(%.1f,%.1f)$' %
                    (i_x*self.dtheta1,i_y*self.dtheta2) )

        return fig
