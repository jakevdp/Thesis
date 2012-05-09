import pylab
import matplotlib
import numpy
import scipy
import scipy.linalg
from scipy import fftpack
from scipy.ndimage import filters
from scipy.sparse.linalg import LinearOperator, aslinearoperator


############################################################
# Utility routines for working with grids of angles.
#
# In general, we will express location on the sky as a
#  complex number theta = theta_x + i*theta_y, where
#  theta_x progresses from left to right, and theta_y
#  progresses from bottom to top.
#
# When working with an array, we want the elements to be
#  accessed in the most intuitive way, e.g. A[j,k]
#  gives the element corresponding to the j^th x-value and
#  the k^th y-value.
#
# Because of this it is necessary to lay out the grids in
#  a particular way.  This is implemented in the utility
#  functions below, where
#    theta1 <=> theta_x
#    theta2 <=> theta_y
#
# When using pylab.imshow to plot a field layed out in this
#  way, one must be careful.  For an array A which fits
#  the above criteria ( A[j,k] gives the value at the j^th
#  x-location and the k^th y-location ), the correct use is
#     pylab.imshow(A.T,origin='lower')
#  this is implemented below, in imshow_Lens3D
#
############################################################

def theta_comp_to_grid(theta1,theta2):
    """
    takes two 1D arrays of positions
    returns a 2D grid of imaginary angles
    """
    ipart,rpart = numpy.meshgrid(theta2,theta1)
    return rpart + 1j * ipart

def theta_minmax_to_grid(theta1_min,theta1_max,N1,
                         theta2_min,theta2_max,N2):
    theta1 = numpy.linspace(theta1_min,theta1_max,N1)
    theta2 = numpy.linspace(theta2_min,theta2_max,N2)
    return theta_comp_to_grid(theta1,theta2)

def theta_grid_to_comp(theta):
    """
    takes a 2D grid of imaginary angles
    returns two 1D arrays of positions
    """
    theta1 = theta.real[:,0]
    theta2 = theta.imag[0,:]
    return (theta1,theta2)

def theta_grid_to_minmax(theta):
    theta1,theta2 = theta_grid_to_comp(theta)
    return ( theta1[0],theta1[-1],len(theta1),
             theta2[0],theta2[-1],len(theta2) )

def imshow_Lens3D(x,y=None,colorbar=True,
                  cbargs = {},
                  **kwargs):
    """
    show an image of the array.
    """
    if 'origin' in kwargs and kwargs['origin'] != 'lower':
        print "warning: imshow_Lens3D: origin keyword should be 'lower'"
    kwargs['origin'] = 'lower'
    
    if y is None:
        pylab.imshow(x.T,**kwargs)
    else:
        assert x.shape==y.shape
        kwargs['extent'] = (x[0,0].real, x[-1,0].real,
                            x[0,0].imag, x[0,-1].imag )
        pylab.imshow(y.T,**kwargs)

    if colorbar:
        pylab.colorbar(**cbargs)

############################################################
# ZeroProtectFunction: wrapper for certain fourier kernels
class ZeroProtectFunction(object):
    """
    class which protects a function from evaluating at zero.  Used as
    a decorator for various fourier-space kernels and lensing 
    convolution kernels.
    """
    def __init__(self,func,func_zero = 0.0):
        self.func = func
        self.func_zero = func_zero
        self.__doc__ = func.__doc__

    def __call__(self,x):
        #make x an array
        x = numpy.asarray(x)
        x_is_scalar = (len(x.shape)==0)
        if x_is_scalar:
            x.resize(1)

        #put in a dummy value where x is zero
        i = numpy.where( abs(x)==0 )
        x[i] = 1.0

        #evaluate the function
        ret = self.func(x)
        ret[i] = self.func_zero
        x[i] = 0
        
        if x_is_scalar:
            return ret[0]
        else:
            return ret

    def inverse(self):
        ret = ZeroProtectFunction(None,None)
        ret.func = lambda x: 1./self.func(x)
        if self.func_zero == 0:
            ret.func_zero = 0.0
        else:
            ret.func_zero = 1./self.func_zero
        return ret

    I = property(lambda self: self.inverse() )


class Lens3D_vector(object):
    """Lens3D_vector class
    An object to hold vector data for weak lensing.  This is a 3D array,
     with dimensions (Nx,Ny,Nz).  Nx and Ny refer to the number of pixels
     in the image.  Nz refers to the number of redshift bins, either
     lens-plane or source-plane.  In the formalism developed in Simon et al
     2009 (arXiv:0907.0016), this 3D array is treated as a vector.  This
     wrapper object provides convenient views of each lens/source-plane and
     each line-of-sight.

    To speed repeat calculations via Lens3D_lp_conv (see below),
    this class stores the fft of each lens plane when it is calculated.
    Also, if a vector is calculated via an fft, the inverse fft is not
    calculated until needed.
    """
    def __init__(self,Nz,Nx,Ny,
                 data = None, 
                 data_fft = None, 
                 **kwargs):
        """Lens3D_vector : __init__(self,Nx,Ny,Nz,data = None)
        initialize a Lens3D_vector object.
        inputs:
          Nz    : number of source/lensing planes
          Nx,Ny : number of pixels in the x,y direction.
          data  : initial vector (defaults to all zero)
                  note that data is assumed to be a 3-d array in the form
                  (Nz,Nx,Ny).  Any other shape with the correct number of
                  entries is valid, but will produce a warning.
          kwargs : passed to numpy array initialization
        """
        self.Nz_ = Nz
        self.Nx_ = Nx
        self.Ny_ = Ny
        
        #storage type: 
        #  0 : real space
        #  1 : fourier space
        self.storage_type_ = None
        self.data_ = None

        #order of data must be 'C' so that data.ravel() will not return a copy.
        if 'order' in kwargs and kwargs['order'] != 'C':
            print "warning: Lens3D_vector: overriding kwarg order"
        kwargs['order'] = 'C'

        #dtype should be complex if not otherwise specified
        if 'dtype' not in kwargs:
            kwargs['dtype'] = complex

        if (data is not None) and (data_fft is not None):
            raise ValueError, "Lens3D_vector : both data and data_fft supplied"
        elif (data is None) and (data_fft is None):
            self.data = numpy.zeros((Nz,Nx,Ny),**kwargs)
        elif (data is not None):
            self.data = \
                numpy.asarray(data,**kwargs).reshape( (Nz,Nx,Ny) )
        elif (data_fft is not None):
            self.data_fft = \
                numpy.asarray(data_fft,**kwargs).reshape((Nz,2*Nx,2*Ny))
        #----
    #----

    def copy(self):
        return self.__class__(self.Nz,self.Nx,self.Ny,
                              data = self.vec.copy())

    def raw_line_of_sight_(self,i_x,i_y):
        """
        return a view of the memory corresponding to lens_plane(i_z)
        without attempting to reconstruct from fft info
        """
        if self.storage_type_ == 0:
            Nx = self.Nx
            Ny = self.Ny
        else:
            Nx = 2*self.Nx
            Ny = 2*self.Ny

        if(i_x>=Nx or i_x<0 or i_y>=Ny or i_y<0):
            raise IndexError, "line_of_sight : index out of range"

        strides = numpy.divide( self.data_.strides, self.data_.itemsize )
        
        return self.data_.ravel()[i_x*strides[1]+i_y*strides[2]::strides[0]]
    #----

    def raw_lens_plane_(self,i_z):
        """
        return a view of the memory corresponding to lens_plane(i_z)
        without attempting to reconstruct from fft info
        """
        if(i_z>=self.Nz or i_z<0):
            raise IndexError, "lens_plane : index out of range"

        strides = numpy.divide( self.data_.strides, self.data_.itemsize )

        if self.storage_type_ == 0:
            shape = (self.Nx,self.Ny)
        else:
            shape = (2*self.Nx,2*self.Ny)
            
        return self.data_.ravel()[i_z*strides[0]:
                                      (i_z+1)*strides[0]].reshape(shape)
    #----

    def check_data_(self,data,s=""):
        """
        private function used to check data updates.
        """
        if data.size != self.Nz*self.Nx*self.Ny:
            raise ValueError, \
                  "Lens3D_vector: %s: data is not the correct size" % s
        elif data.shape in ( (self.Nz*self.Nx*self.Ny,),
                             (self.Nx,self.Ny),
                             (self.Nz,self.Nx,self.Ny) ):
            pass
        else:
            print data.shape
            print "warning: %s: Lens3D data may be mis-shaped." % s
            print "  data assumed to be shaped (Nz,Nx,Ny)"
        #----
    #----

    def check_data_fft_(self,data_fft,s=""):
        """
        private function used to check data updates.
        """
        if data_fft.size != 4*self.Nz*self.Nx*self.Ny:
            raise ValueError, \
                  "Lens3D_vector: %s: data_fft is not the correct size" % s
        elif data_fft.shape in ( (4*self.Nz*self.Nx*self.Ny,),
                             (2*self.Nx,2*self.Ny),
                             (self.Nz,2*self.Nx,2*self.Ny) ):
            pass
        else:
            print data_fft.shape
            print "warning: %s: Lens3D data may be mis-shaped." % s
            print "  data assumed to be shaped (Nz,Nx,Ny)"
        #----
    #----
    
    def line_of_sight(self,i_x,i_y, fourier=False):
        """Lens3D_vector : line_of_sight(self, i_x, i_y)
        return a view of the data along a particular line-of-sight.
        inputs:
          i_x,i_y : index of the pixel for which the line of sight is returned.
        returns:
          L : a length Nz view of the data vector corresponding to the
              specified line-of-sight
        """
        if fourier:
            self.move_to_fourier()
        else:
            self.move_to_real()

        return self.raw_line_of_sight_(i_x,i_y)
    #----
    
    def lens_plane(self,i_z, fourier=False):
        """Lens3D_vector : lens_plane(self, i_z)
              : source_plane(self,i_z)
        return a view of the data in a particular lens/source plane.
        inputs:
          i_z : index of the desired redshift bin.
        returns:
          P : a size=(Nx,Ny) view of the data vector corresponding to the
              specified redshift bin.
        """
        if fourier:
            self.move_to_fourier()
        else:
            self.move_to_real()

        return self.raw_lens_plane_(i_z)
    
    #----
    source_plane = lens_plane
    #----

    @property
    def size(self):
        return self.Nx*self.Ny*self.Nz

    @property
    def shape(self):
        return (self.size,)

    @property
    def dtype(self):
        return self.data_.dtype
    
    def __len__(self):
        return self.size

    #------------------------------------------------------------
    # methods to move back and forth between real & fourier space
    
    def move_to_fourier(self):
        if self.storage_type_ == 0:
            self.data_fft = fftpack.fft2( self.data_,(2*self.Nx,2*self.Ny) )
    #----

    def move_to_real(self):
        if self.storage_type_ == 1:
            self.data = fftpack.ifft2(self.data_)[:,:self.Nx,:self.Ny]
    #----

    def is_fourier(self):
        return self.storage_type_ == 1
    #----

    def is_real(self):
        return self.storage_type_ == 0
    #----

    #-------------------------------------------------------------
    # methods to get/set data and fft of data
    def set_data_fft(self,data_fft):
        self.storage_type_ = 1
        self.check_data_fft_(data_fft,"set_data_fft")
        self.data_ = numpy.asarray(data_fft,order='C').reshape( (self.Nz,2*self.Nx,2*self.Ny) )
    #----

    def get_data_fft(self):
        self.move_to_fourier()
        return self.data_
    #----
    
    def set_data(self,data):
        self.storage_type_ = 0
        self.check_data_(data,"set_data")
        self.data_ = numpy.asarray(data,order='C').reshape( (self.Nz,self.Nx,self.Ny) )
    #----

    def get_data(self):
        self.move_to_real()
        return self.data_
    #----
    
    data = property(get_data,
                    set_data)
    data_fft = property(get_data_fft,
                        set_data_fft)
    vec = property(lambda self: self.data.ravel(),
                   set_data)
            
    #access to dimensions: these cannot be changed.
    Nz = property(lambda self: self.Nz_)
    Nx = property(lambda self: self.Nx_)
    Ny = property(lambda self: self.Ny_)

    #------------------------------------------------------------
    #data plotting functions
    def contour_lens_plane(self,i_z,part='r',nlevels=50,loglevels=True,
                           extent=None,fill=False,label_contours=None,
                           gaussian_filter = False,
                           **kwargs):
        """
        produce a filled contour plot of the given lens plane.
        part is one of 'r','i','n': real, imaginary, or norm
        """
        lens_plane = self.lens_plane(i_z)

        if label_contours is None:
            label_contours = not fill
        
        if part.lower()=='i':
            lens_plane = lens_plane.imag
        elif part.lower()=='r':
            lens_plane = lens_plane.real
        elif part.lower() in ('n','a'):
            lens_plane = abs(lens_plane)
        else:
            raise ValueError, "part=%s not understood.  Must be one of i,r,n,a" % part
        if gaussian_filter:
            R = gaussian_filter
            lens_plane = filters.gaussian_filter(lens_plane,R,mode='mirror')

        if extent is not None:
            assert len(extent)==4
            theta1 = numpy.linspace( extent[0],extent[1],lens_plane.shape[0] )
            theta2 = numpy.linspace( extent[2],extent[3],lens_plane.shape[1] )
        else:
            theta1 = numpy.arange(lens_plane.shape[0])
            theta2 = numpy.arange(lens_plane.shape[1])

        lmin = numpy.min(lens_plane)
        lmax = numpy.max(lens_plane)
        if loglevels:
            levels = 2**(numpy.linspace(numpy.log(lmin)/numpy.log(2),
                                        numpy.log(lmax)/numpy.log(2),
                                        nlevels) )
            levels.round(2)
        else:
            levels = nlevels

        theta = theta_comp_to_grid(theta1,theta2)

        if fill:
            CS = pylab.contourf(theta.real,theta.imag,
                                lens_plane,levels,**kwargs)
        else:
            CS = pylab.contour(theta.real,theta.imag,
                               lens_plane,levels,**kwargs)
        pylab.colorbar()

        if label_contours:
            pylab.clabel(CS,inline=1, fontsize=10, fmt = '%.2g')
        

    def plot_3_planes(self,
                      part = 'r',
                      loglevels=True,
                      gaussian_filter=0,
                      *args,
                      **kwargs):
        if 'cmap' not in kwargs:
            kwargs['cmap'] = pylab.cm.binary

        def my_imshow(x,y,z,*args,**kwargs):
            if 'interpolation' not in kwargs:
                kwargs['interpolation']='nearest'
            if 'origin' not in kwargs:
                kwargs['origin'] = 'lower'
            #print (min(x),max(x),min(y),max(y))
            return pylab.imshow(z,*args,
                                extent=(min(x),max(x),min(y),max(y)),
                                aspect=(x[-1]-x[0])*1./(y[-1]-y[0]),
                                **kwargs)
        
        if part.lower()=='i':
            lens_plane = self.data.imag
        elif part.lower()=='r':
            lens_plane = self.data.real
        elif part.lower() in ('n','a'):
            lens_plane = abs(self.data)
        else:
            raise ValueError, "part=%s not understood.  Must be one of i,r,n,a" % part

        x = numpy.arange(self.Nx)
        y = numpy.arange(self.Ny)
        z = numpy.arange(self.Nz)

        im = []

        #pylab.subplot(223)
        pylab.axes((0.07,0.1,0.35,0.4))
        Lsum = lens_plane.sum(0).T#/self.Nz
        if loglevels:
            Lsum = numpy.log10(Lsum)
        cmin = numpy.min(Lsum)
        cmax = numpy.max(Lsum)
        im.append( my_imshow(x,y,Lsum,
                             *args,**kwargs) )
        pylab.colorbar()
        pylab.xlabel('x')
        pylab.ylabel('y')
        pylab.xlim(x[0],x[-1])
        pylab.ylim(y[0],y[-1])
        
        
        #pylab.subplot(221)
        pylab.axes((0.07,0.55,0.35,0.4))
        Lsum = lens_plane.sum(2)#/self.Ny
        if loglevels:
            Lsum = numpy.log10(Lsum)
        cmin = min(cmin,numpy.min(Lsum))
        cmax = max(cmax,numpy.max(Lsum))
        im.append( my_imshow(x,z,Lsum,
                             *args, **kwargs) )
        
        pylab.colorbar()
        #pylab.xlabel('x')
        pylab.ylabel('z')
        pylab.xlim(x[0],x[-1])
        pylab.ylim(z[0],z[-1])
        
        #pylab.subplot(224)
        pylab.axes((0.5,0.1,0.35,0.4))
        Lsum = lens_plane.sum(1).T#/self.Nx
        if loglevels:
            Lsum = numpy.log10(Lsum)
        cmin = min(cmin,numpy.min(Lsum))
        cmax = max(cmax,numpy.max(Lsum))
        im.append( my_imshow(z,y,Lsum,
                             *args, **kwargs) )
        pylab.colorbar()
        pylab.xlabel('z')
        #pylab.ylabel('y')
        pylab.xlim(z[0],z[-1])
        pylab.ylim(y[0],y[-1])
        
        #ax = pylab.gcf().add_axes( (0.88,0.05,0.03,0.9) )
        #norm = matplotlib.colors.Normalize(vmin=cmin, vmax=cmax)
        #cmap = pylab.cm.binary
        # 
        #cb = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap,
        #                                      norm=norm,
        #                                      orientation='vertical')
        
        #for i in range(3):
        #    im[i].set_clim(cmin,cmax)
        

    def contourf_lens_plane(self,*args,**kwargs):
        kwargs['fill'] = True
        self.contour_lens_plane(*args,**kwargs)

    def imshow_lens_plane(self,i_z,part='r',
                          extent=None,
                          gaussian_filter=False,
                          label = None,
                          loglevels = True,
                          colorbar = True,
                          cbargs = {},
                          **kwargs):
        if i_z == 'all':
            lens_plane = self.data.sum(0)
        else:
            lens_plane = self.lens_plane(i_z)

        if extent is not None:
            assert len(extent)==4
            theta1 = numpy.linspace( extent[0],extent[1],lens_plane.shape[0] )
            theta2 = numpy.linspace( extent[2],extent[3],lens_plane.shape[1] )
        else:
            theta1 = numpy.arange(lens_plane.shape[0])
            theta2 = numpy.arange(lens_plane.shape[1])

        #correct extent so that pixels line up with fieldplot
        theta1_half_pixel = 0.5*(max(theta1)-min(theta1))/(len(theta1)-1)
        theta2_half_pixel = 0.5*(max(theta2)-min(theta2))/(len(theta2)-1)

        kwargs['extent'] = ( min(theta1)-theta1_half_pixel,
                             max(theta1)+theta1_half_pixel,
                             min(theta2)-theta2_half_pixel,
                             max(theta2)+theta2_half_pixel )
        
        if part.lower()=='i':
            lens_plane = lens_plane.imag
        elif part.lower()=='r':
            lens_plane = lens_plane.real
        elif part.lower() in ('n','a'):
            lens_plane = abs(lens_plane)
        else:
            raise ValueError, "part=%s not understood.  Must be one of i,r,n,a" % part
        
        if gaussian_filter:
            R = gaussian_filter
            lens_plane = filters.gaussian_filter(lens_plane,R,mode='mirror')
            
        if 'cmap' not in kwargs:
            kwargs['cmap'] = pylab.cm.gray

        kwargs['colorbar'] = False
        
        #check the spread in order that the colorbar has a useful label
        logspread = numpy.log10(lens_plane.max() / lens_plane.min())

        if (not loglevels) or (lens_plane.min() < 0):
            imshow_Lens3D(lens_plane,interpolation='nearest',**kwargs)
            if colorbar:
                try:
                    cb = pylab.colorbar(**cbargs)
                    if label:
                        cb.ax.set_ylabel(r'$\rm{%s}$' % label )
                except AttributeError:
                    print "cannot show colorbar"
        elif logspread > 2:
            # if there are zeros in lens_plane, it will screw things up
            i_zero = numpy.where(lens_plane==0)
            lens_plane[i_zero] = numpy.max(lens_plane)
            imshow_Lens3D(lens_plane,interpolation='nearest',
                          norm = matplotlib.colors.LogNorm(),**kwargs)
            # now put them back to zero: plot will be updated.
            #  bad hack: not sure why this works but it does
            lens_plane[i_zero] = 0.0
            LL = matplotlib.ticker.LogLocator(10,[0.01*x for x in range(10)] +\
                                              [0.1*x for x in range(10)])
            LF = matplotlib.ticker.LogFormatterMathtext(10)
            if colorbar:
                try:
                    cb = pylab.colorbar(ticks  = LL, format = LF, **cbargs )
                    if label:
                        cb.ax.set_ylabel(r'$\rm{%s}$' % label )
                except AttributeError:
                    print "cannot show colorbar"
        else:
            log_lens_plane = numpy.log10(lens_plane)
        
            imshow_Lens3D(log_lens_plane,interpolation='nearest',**kwargs)
            #imshow_Lens3D(lens_plane,interpolation='nearest',**kwargs)

            if colorbar:
                try:
                    cb = pylab.colorbar(**cbargs)
                    if label:
                        cb.ax.set_ylabel(r'$\log_{10}(\rm{%s})$' % label )
                    #cb.ax.set_ylabel(label)
                except AttributeError:
                    print "cannot show colorbar"

        try:
            return cb
        except:
            return 

    def fieldplot_lens_plane(self,i_z,n_bars=15,
                             normalize=True,
                             extent=None,
                             **kwargs):
        lens_plane = self.lens_plane(i_z)

        if extent is not None:
            assert len(extent)==4
            theta1 = numpy.linspace( extent[0],extent[1],lens_plane.shape[0] )
            theta2 = numpy.linspace( extent[2],extent[3],lens_plane.shape[1] )
        else:
            theta1 = numpy.arange(lens_plane.shape[0])
            theta2 = numpy.arange(lens_plane.shape[1])
        
        stride_0 = max( 1,int(lens_plane.shape[0]/n_bars) )
        stride_1 = max( 1,int(lens_plane.shape[1]/n_bars) )

        N_0 = stride_0*n_bars
        N_1 = stride_1*n_bars

        lp = numpy.zeros( (n_bars,n_bars),
                          dtype = complex )
        for i in range(stride_0):
            for j in range(stride_1):
                lp += lens_plane[i:N_0:stride_0,j:N_1:stride_1]
        lp /= (stride_0*stride_1)
        lens_plane = lp
        theta1 = theta1[stride_0/2:N_0+stride_0/2:stride_0]
        theta2 = theta2[stride_1/2:N_1+stride_1/2:stride_1]

        theta = theta_comp_to_grid(theta1,theta2)

        mag = numpy.abs(lens_plane)

        lp_plot = (lens_plane/mag)**0.5

        lp_plot[mag==0] = 0.0
        if normalize:
            scale = n_bars
        else:
            scale = None
            lp_plot *= mag

        lp_x = lp_plot.real
        lp_y = lp_plot.imag

        if 'color' not in kwargs:
            kwargs['color'] = 'g'
        
        pylab.quiver(theta.real,theta.imag,lp_x,lp_y,
                     pivot = 'middle',headwidth=0,
                     scale=scale,**kwargs )
    
    def plot_los(self,i_x,i_y,part='r',
                 z_array=None,**kwargs):
        los = self.line_of_sight(i_x,i_y)
        if part.lower()=='i':
            los = los.imag
        elif part.lower()=='r':
            los = los.real
        elif part.lower() in ('n','a'):
            los = abs(los)
        else:
            raise ValueError, "part=%s not understood.  Must be one of i,r,n,a" % part
        if z_array is not None:
            if len(z_array) == len(los):
                P = pylab.plot(z_array,los,**kwargs)
            else:
                print "Lens3D_vector.plot_los :"
                print "   warning: len(z_array) is incorrect.  Ignoring it."
                P = pylab.plot(los,**kwargs)
        else:
            P = pylab.plot(los,**kwargs)
        return P

    #------------------------------------------------------------
    # set border : useful for deweighting the pixels at the edge
    #   use noise.set_border(...)
    def set_border(self,border_size,val=0.0):
        if border_size>0:
            j = border_size
            for i in range(self.Nz):
                self.lens_plane(i)[:j,:] = val
                self.lens_plane(i)[-j:,:] = val
                self.lens_plane(i)[:,:j] = val
                self.lens_plane(i)[:,-j:] = val
        elif border_size<0:
            raise ValueError, "set_border: border size must be non-negative"
        
    #----------------------------------------------------------------------
    #  Arithemetic Operations

    def __mul__(self,other):
        return Lens3D_vector(self,self.Nz,self.Nx,self.Ny,
                             data = self.vec * other)
            
    def __rmul__(self,other):
        return Lens3D_vector(self,self.Nz,self.Nx,self.Ny,
                             data = self.vec * other)

    def __imul__(self,other):
        self.data_ *= other
        return self

    def write(self,filename):
        print "writing Lens3D_vector to", filename
        OF = open(filename,'w')
        OF.write('#Nz %i\n' % self.Nz)
        OF.write('#Nx %i\n' % self.Nx)
        OF.write('#Ny %i\n' % self.Ny)
        OF.write('#Re(v) Im(v)\n')
        for val in self.vec:
            OF.write('%.8g %.8g\n' % (numpy.real(val),numpy.imag(val)) )

###############################################################################
def Lens3D_vec_from_file(filename):
    print "reading Lens3D_vector from", filename
    Nz = None
    Nx = None
    Ny = None

    vec = None
    i=0
    
    for line in open(filename):
        line = line.strip()
        if len(line)==0:
            continue
        elif line.startswith('#'):
            if line.startswith('#Nx'):
                Nx = int(line.split()[1])
            elif line.startswith('#Ny'):
                Ny = int(line.split()[1])
            elif line.startswith('#Nz'):
                Nz = int(line.split()[1])
            else:
                continue

            if ( (vec is None) and Nx and Ny and Nz ):
                vec = numpy.zeros(Nx*Ny*Nz, dtype=complex)

        else:
            line = map(float,line.split())
            vec[i] = line[0] + 1j*line[1]
            i += 1

    return Lens3D_vector(Nz,Nx,Ny,vec)
###############################################################################

class Lens3D_matrix_base(object):
    """
    Abstract Base Class for Lens3D matrix objects.
    """
    def __init__(self, Nz_in, Nx_in, Ny_in,
                 Nz_out = None,
                 Nx_out = None,
                 Ny_out = None):
        if self.__class__ == Lens3D_matrix_base:
            raise NotImplementedError, "Lens3D_matrix_base is an abstract " +\
                  "class.  It can only be used as a subclass."
            
        self.Nz_in_ = Nz_in
        self.Nx_in_ = Nx_in
        self.Ny_in_ = Ny_in

        if Nz_out is None:  self.Nz_out_ = Nz_in
        else:               self.Nz_out_ = Nz_out

        if Nx_out is None:  self.Nx_out_ = Nx_in
        else:               self.Nx_out_ = Nx_out

        if Ny_out is None:  self.Ny_out_ = Ny_in
        else:               self.Ny_out_ = Ny_out

    #----

    def get_size(self):
        return self.Nz_out * self.Nx_out * self.Ny_out *\
               self.Nz_in  * self.Nx_in  * self.Ny_in
    #----

    def get_shape(self):
        return ( self.Nz_out * self.Nx_out * self.Ny_out,
                 self.Nz_in * self.Nx_in * self.Ny_in )
    #----

    def get_fullshape(self):
        return ( (self.Nz_out,self.Nx_out,self.Ny_out),
                 (self.Nz_in,self.Nx_in,self.Ny_in) )

    def check_data_(self,data,s=""):
        if data.shape != self.shape:
            raise ValueError, "%s : data shape mismatch" % s
    #----

    def check_vec_(self,vec,s=""):
        if vec.shape not in ( (self.shape[1],),(self.shape[1],1) ):
            raise ValueError, "%s : vector shape mismatch" % s
        if vec.__class__ == Lens3D_vector:
            if (vec.Nx != self.Nx_in) or \
                    (vec.Ny != self.Ny_in) or \
                    (vec.Nz != self.Nz_in):
                raise ValueError, "%s : vector shape mismatch" % s
    #----

    def check_mat_(self,mat,s=""):
        if mat.ndim !=2 or mat.shape[0] != self.shape[1]:
            raise ValueError, "%s : matrix shape mismatch" % s
    #----

    def view_as_Lens3D_vec(self,v):
        if v.__class__ == Lens3D_vector:
            return v
        elif len(v)==self.shape[0]:
            return Lens3D_vector(self.Nz_out,self.Nx_out,self.Ny_out,v)
        elif len(v)==self.shape[1]:
            return Lens3D_vector(self.Nz_in,self.Nx_in,self.Ny_in,v)
        else:
            raise ValueError, "view_as_Lens3D_vec: v does not match matrix dimensions"
    #----

    def view_as_normal_vec(self,v):
        if v.__class__ == Lens3D_vector:
            return v.vec
        else:
            return numpy.asarray(v,order='C').ravel()
    #----

    def view_as_same_type(self,v,v_type):
        if v_type.__class__ == Lens3D_vector:
            return self.view_as_Lens3D_vec(v)
        elif v_type.__class__ == numpy.ndarray:
            v_arr = numpy.asarray(self.view_as_normal_vec(v))
            if v_arr.size == v_type.size:
                return v_arr.reshape(v_type.shape)
            else:
                return v_arr
        elif v_type.__class__ == numpy.core.defmatrix.matrix:
            v_mat = numpy.asmatrix(self.view_as_normal_vec(v))
            if v_mat.size == v_type.size:
                return v_mat.reshape(v_type.shape)
            else:
                return v_mat
        elif v_type.__class__ == list:
            vL_arr = self.view_as_normal_vec(v)
            vtL_arr = numpy.asarray(v_type)
            if vL_arr.size == vtL_arr.size:
                return list(vL_arr.reshape(vtL_arr.shape))
            else:
                return list(vL_arr)
        else:
            raise ValueError, "view_as_same_type : unrecognized type",type(v_type)
    #----

    def __mul__(self,other):
        return self.matvec(other)
    #----

    #------------------------------------------------------------
    # undefined functions : these should be defined by subclasses
    
    def matvec(self,v):
        raise NotImplementedError, "%s: matvec must be defined by Lens3D_matrix_base subclasses" % self.__class__
    #----
    
    def matmat(self,v):
        raise NotImplementedError, "%s: matmat not implemented" % self.__class__
    #----
    
    def rmatvec(self,v):
        raise NotImplementedError, "%s: rmatvec not implemented" % self.__class__
    #----

    #------------------------------------------------------------
    # optional undefined functions : these may be defined by subclasses

    def get_data(self):
        raise NotImplementedError, "%s: subclass must define get_data" % self.__class__
    #----
    
    def set_data(self,data):
        raise NotImplementedError, "%s: subclass must define get_data" % self.__class__
    #----

    def transpose(self):
        raise NotImplementedError, "%s: transpose is not implemented" % self.__class__
    #----

    def conj(self):
        raise NotImplementedError, "%s: conj is not implemented" % self.__class__
    #----

    def conj_transpose(self):
        raise NotImplementedError, "%s: conj_transpose is not implemented" % self.__class__
    #----

    def inverse(self):
        raise NotImplementedError, "%s: inverse is not implemented" % self.__class__
    #----
        

    #------------------------------------------------------------
    #  data access
    def get_Nx(self):
        if self.Nx_in_ != self.Nx_out_:
            raise ValueError, "Lens3D_matrix_base : cannot get property Nx"
        else:
            return self.Nx_in_

    def get_Ny(self):
        if self.Ny_in_ != self.Ny_out_:
            raise ValueError, "Lens3D_matrix_base : cannot get property Ny"
        else:
            return self.Ny_in_

    def get_Nz(self):
        if self.Nz_in_ != self.Nz_out_:
            raise ValueError, "Lens3D_matrix_base : cannot get property Nz"
        else:
            return self.Nz_in_


    #------------------------------------------------------------
    #  properties

    @property
    def dtype(self):
        raise NotImplementedError, "dtype not implemented"

    @property
    def full_matrix(self):
        try:
            dtype = self.dtype
        except:
            dtype = self.matvec(numpy.ones(self.shape[1])).dtype

        MR = numpy.empty((self.shape[0],self.shape[1]),dtype=dtype)
        for i in range(self.shape[1]):
            v = numpy.zeros(self.shape[1])
            v[i] = 1
            MR[:,i] = self.matvec(v)

        return MR

    T = property(lambda self: self.transpose() )
    H = property(lambda self: self.conj_transpose() )
    I = property(lambda self: self.inverse() )
    data = property(lambda self: self.get_data(),
                    lambda self,d: self.set_data(d) )
    shape = property(lambda self: self.get_shape() )
    fullshape = property(lambda self: self.get_fullshape() )
    size = property(lambda self: self.get_size() )

    Nz_in = property(lambda self: self.Nz_in_)
    Nx_in = property(lambda self: self.Nx_in_)
    Ny_in = property(lambda self: self.Ny_in_)
    
    Nz_out = property(lambda self: self.Nz_out_)
    Nx_out = property(lambda self: self.Nx_out_)
    Ny_out = property(lambda self: self.Ny_out_)

    Nz = property(get_Nz)
    Nx = property(get_Nx)
    Ny = property(get_Ny)

###############################################################################

class Lens3D_ident(Lens3D_matrix_base):
    """
    efficient implementation of the identity matrix
    """
    def __init__(self,
                 Nz,Nx,Ny,
                 *args,**kwargs):
        if len(args) + len(kwargs) != 0:
            raise ValueError, "Lens3D_ident must be square"
        else:
            Lens3D_matrix_base.__init__(self,Nz,Nx,Ny,Nz,Nx,Ny)
    
    def matvec(self,v):
        self.check_vec_(v,"%s.matvec" % self.__class__.__name__)
        return v

    def matmat(self,m):
        self.check_mat_(v,"%s.matmat" % self.__class__.__name__)
        return m

    def transpose(self):
        return self

    def conj(self):
        return self

    def conj_transpose(self):
        return self

    def inverse(self):
        return self


###############################################################################
class Lens3D_const(Lens3D_matrix_base):
    """
    wrapper for multiplication by a constant
    """
    def __init__(self,Nz,Nx,Ny,val):
        Lens3D_matrix_base.__init__(self,Nz,Nx,Ny,Nz,Nx,Ny)
        self.val_ = val

    def matvec(self,v):
        self.check_vec_(v,"%s.matvec" % self.__class__.__name__)
        vv = self.view_as_Lens3D_vec(v)
        if vv.storage_type_ == 0:
            return self.view_as_same_type(self.val_*vv.vec,v)
        else:
            ret = Lens3D_vector( vv.Nz,vv.Nx,vv.Ny,numpy.zeros(vv.size) )
            ret.data_fft = self.val_ * vv.data_fft
            return self.view_as_same_type(ret,v)

    def transpose(self):
        return self

    def conj(self):
        return self.__class__(self.Nz,self.Nx,self.Ny,numpy.conj(self.val_))

    def conj_transpose(self):
        return self.conj()

    def inverse(self):
        return self.__class__(self.Nz,self.Nx,self.Ny,1.0/self.val_)

    def get_val(self):
        return self.val_

    def set_val(self,val):
        if hasattr(val,len):
            raise ValueError, "Lens3D_const.set_val: expect scalar value"
        self.val_ = val

    val = property(get_val,set_val)
    

###############################################################################

class Lens3D_diag(Lens3D_matrix_base):
    """
    Efficient implementation of a diagonal matrix
    """
    def __init__(self,Nz,Nx,Ny,diag):
        Lens3D_matrix_base.__init__(self,Nz,Nx,Ny,Nz,Nx,Ny)
        self.data_ = diag.reshape(diag.size)

    def copy(self):
        return self.__class__(self.Nz,self.Nx,self.Ny,
                              self.data_.copy())

    def matvec(self,v):
        self.check_vec_(v,"%s.matvec" % self.__class__.__name__)
        v_n = self.view_as_normal_vec(v)
        return self.view_as_same_type(self.data_*v_n,v)

    def transpose(self):
        return self

    def conj(self):
        return self.__class__(self.Nz,self.Nx,self.Ny,self.data_.conj())

    def conj_transpose(self):
        return self.conj()

    def inverse(self):
        return self.__class__(self.Nz,self.Nx,self.Ny,1.0/self.data_)

    def get_data(self):
        return self.data_

    def set_data(self,diag):
        if len(diag) != len(self.data_):
            raise ValueError, "%s.set_data : shape mismatch" \
                  % self.__class__.__name__

    def set_border(self,border_size,val):
        L3V = self.view_as_Lens3D_vec(self.data_)
        L3V.set_border(border_size,val)
        self.data_ = L3V.vec

    def line_of_sight(self,i_x,i_y):
        if(i_x>=self.Nx or i_x<0 or i_y>=self.Ny or i_y<0):
            raise IndexError, "line_of_sight : index out of range"

        D = self.data_.reshape((self.Nz,self.Nx,self.Ny))
        
        strides = numpy.divide( D.strides, D.itemsize )
        
        return D.ravel()[i_x*strides[1]+i_y*strides[2]::strides[0]]
        
        
###############################################################################
        

class Lens3D_full_mat(Lens3D_matrix_base):
    """
    a Lens3D matrix object which is implemented via a
    normal matrix
    """
    def __init__(self,
                 Nz_in, Nx_in, Ny_in,
                 Nz_out=None, Nx_out=None, Ny_out=None,
                 data = None):
        Lens3D_matrix_base.__init__(self,
                                    Nz_in,Nx_in,Ny_in,
                                    Nz_out,Nx_out,Ny_out)
        if data == None:
            self.data_ = numpy.zeros(self.shape_)
        else:
            self.check_data_(data,"Lens3D_mat.__init__")
            self.data_ = data
    #----
    
    def matvec(self,v):
        self.check_vec_(v,"%s.matvec" % self.__class__.__name__)
        try:
            retval = numpy.dot(self.data_,v.vec)
        except:
            retval = numpy.dot(self.data_,v)
        return self.view_as_same_type(retval,v)
    #----

    def get_data(self):
        return self.data_
    
    def set_data(self,data):
        self.data_ = data
    
    def matmat(self,m):
        return numpy.dot(self.data_,m)
    #----

    def transpose(self):
        return self.__class__(self.Nz_out,self.Nx_out,self.Ny_out,
                              self.Nz_in,self.Nx_in,self.Ny_in,
                              self.data_.T)
    #----

    def conj(self):
        return self.__class__(self.Nz_in,self.Nx_in,self.Ny_in,
                              self.Nz_out,self.Nx_out,self.Ny_out,
                              self.data_.conj() )
    #----

    def conj_transpose(self):
        return self.__class__(self.Nz_out,self.Nx_out,self.Ny_out,
                              self.Nz_in,self.Nx_in,self.Ny_in,
                              self.data_.conj().transpose() )
    #----

    def inverse(self):
        return self.__class__(self.Nz_out,self.Nx_out,self.Ny_out,
                              self.Nz_in,self.Nx_in,self.Ny_in,
                              numpy.linalg.inv(self.data_) )
    #----

######################################################################

class Lens3D_lp_diag(Lens3D_matrix_base):
    """
    implement multiplication of each lens plane by a diagonal matrix
    """
    def __init__(self,Nz,Nx,Ny,
                 diag = None):
        Lens3D_matrix_base.__init__(self,Nz,Nx,Ny)
        if diag is not None:
            assert len(diag) == Nx*Ny
        self.diag_ = numpy.asarray(diag)

    def matvec(self,v):
        va = self.view_as_normal_vec(v)
        va.resize((self.Nz,self.Nx*self.Ny))
        va *= self.diag_
        va.resize(self.Nx*self.Ny*self.Nz)
        return self.view_as_same_type(va,v)

    def transpose(self):
        return self

    def conj(self):
        return Lens3D_lp_diag(self.Nz,self.Nx,self.Ny,
                              numpy.conj(self.diag_) )
    
    def conj_transpose(self):
        return self.conj()

    def inverse(self):
        return Lens3D_lp_diag(self.Nz,self.Nx,self.Ny,
                              1./self.diag_ )

    @property
    def dtype(self):
        return self.diag_.dtype

class Lens3D_lp_mat(Lens3D_matrix_base):
    """
    A Lens3D matrix object which implements independent operations in
    each lens plane: in essence, a block-diagonal matrix with Nz blocks,
    each corresponding to an operation across a particular lens plane.

    This is useful for expressing the representation of a 
    Lens3D_lp_conv() operation (see class definition below)
    """
    def __init__(self, Nz, Nx_in, Ny_in,
                 Nx_out=None,Ny_out=None,
                 mat_list = None):
        """
        mat_list should be a list of matrices, with shape 
        [(Nx_out*Ny_out) x (Nx_in*Ny_in)]
        """
        Lens3D_matrix_base.__init__(self,Nz,Nx_in,Ny_in,
                                    Nz,Nx_out,Ny_out)
        
        mat_shape = (Nz,
                     self.Nx_out*self.Ny_out,
                     self.Nx_in*self.Ny_in)

        if mat_list is not None:
            mat_list_a = numpy.asarray(mat_list)
            if mat_list_a.shape == mat_shape:
                self.mat_list_ = mat_list
            elif mat_list_a.shape == mat_shape:
                self.mat_list_ = Nz*[mat_list[0]]
            elif mat_list_a.shape == mat_shape[1:]:
                self.mat_list_ = Nz*[mat_list_a]
            else:
                print mat_shape
                print mat_list_a.shape
                raise ValueError, "Lens3D_lp_mat : mat_list not understood."
        else:
            self.mat_list_ = [None]*Nz

    def matvec(self,v):
        self.check_vec_(v,"%s.matvec" % self.__class__.__name__)
        vv = self.view_as_normal_vec(v)
        ret = numpy.zeros(self.shape[0],dtype = complex)
        
        L_in = self.Nx_in*self.Ny_in
        L_out = self.Nx_out*self.Ny_out
        for i in range(self.Nz):
            vvi = vv[i*L_in:(i+1)*L_in]
            ret[i*L_out:(i+1)*L_out] = numpy.dot( self.mat_list_[i],vvi )
            
        return self.view_as_same_type(ret,v)

    def transpose(self):
        if self.mat_list_[-1] is self.mat_list_[0]:            
            return self.__class__(self.Nz,self.Nx_out,self.Ny_out,
                                  self.Nx_in,self.Ny_in,
                                  numpy.transpose(self.mat_list_[0]) )
        else:
            return self.__class__(self.Nz,self.Nx_out,self.Ny_out,
                                  self.Nx_in,self.Ny_in,
                                  map(numpy.transpose,self.mat_list_) )
                            
    def conj(self):
        if self.mat_list_[-1] is self.mat_list_[0]:            
            return self.__class__(self.Nz,self.Nx_in,self.Ny_in,
                                  self.Nx_out,self.Ny_out,
                                  numpy.conj(self.mat_list_[0]) )
        else:
            return self.__class__(self.Nz,self.Nx_in,self.Ny_in,
                                  self.Nx_out,self.Ny_out,
                                  map(numpy.conj,self.mat_list_) )

    def conj_transpose(self):
        ct = lambda x: numpy.conj(numpy.transpose(x))
        if self.mat_list_[-1] is self.mat_list_[0]:            
            return self.__class__(self.Nz,self.Nx_out,self.Ny_out,
                                  self.Nx_in,self.Ny_in,
                                  ct(self.mat_list_[0]) )
        else:
            return self.__class__(self.Nz,self.Nx_out,self.Ny_out,
                                  self.Nx_in,self.Ny_in,
                                  map(ct,self.mat_list_)  )

    def inverse(self):
        #do the full inverse of the matrix
        if self.mat_list_[-1] is self.mat_list_[0]:            
            return Lens3D_lp_mat(self.Nz,self.Nx,self.Ny,
                                 mat_list=numpy.linalg.inv(self.mat_list_[0]) )
        else:
            return Lens3D_lp_mat(self.Nz,self.Nx,self.Ny,
                                 mat_list=map(numpy.linalg.inv,self.mat_list_))
    
    @property
    def full_matrix(self):
        full_mat = numpy.zeros([self.Nx_out*self.Ny_out*self.Nz_out,
                                self.Nx_in*self.Ny_in*self.Nz_in],
                               dtype = self.dtype)
        NxNy = self.Nx*self.Ny
        for i in range(self.Nz):
            i1 = i*NxNy
            i2 = i1+NxNy
            full_mat[i1:i2,i1:i2] = self.mat_list_[i]
        
        return full_mat

    @property
    def dtype(self):
        return self.mat_list_[0].dtype

######################################################################

class Lens3D_lp_conv(Lens3D_matrix_base):
    """
    A Lens3D matrix object which implements convolutions of the
    source/lens planes via a 2-d fast fourier transform.  This is
    useful in particular for the operation which maps complex shear
    onto complex convergence.

    Takes an argument func which is a function mapping a complex vector
    to a complex vector.  For a pixel at location theta_x,theta_y,
    func takes a single complex argument (theta_x + i*theta_y) and
    returns a complex number.

    func and func_ft can either be a single callable function, or a list of
    functions of length Nz.

    If func is supplied and func_ft is not, then func will be binned and
    a discrete fourier transform will be performed.

    If func_ft is supplied, then the analytic fourier transform will be
    used.

    The fourier transform convention is assumed to be
      F(k) = integral[ f(t) exp(ikt)]
      f(t) = 1/2pi integral[F(k) exp(-ikt)]
    in mathematica, this corresponds to FourierParameters->{1,1}
     fourier transform is sqrt(2pi) * [mathematica default]

    Multiplication with a vector is equivalent to performing the following
    convolution:
      y_out[theta'] = integral{ dtheta y_in[theta] * func[theta'-theta] }
    where theta is a complex 2-d angle, y_in and y_out are [Nx x Ny]
    lens/source planes, and the operation is performed at each of the
    Nz lens/source planes.

    dx and dy are assumed to be given in units that match the
      expected input of func
    """
    def __init__(self,Nz,Nx,Ny,dx,dy,func=None,func_ft=None):
        Lens3D_matrix_base.__init__(self,Nz,Nx,Ny)
        self.dx_ = dx
        self.dy_ = dy

        self.func_ = func
        self.func_ft_ = func_ft

        if func_ft is not None:
            #analytic fourier transform is supplied: use this
            dkx = numpy.pi / Nx / dx
            dky = numpy.pi / Ny / dy

            kx = fftpack.ifftshift( dkx * (numpy.arange(2*Nx)-Nx) )
            ky = fftpack.ifftshift( dky * (numpy.arange(2*Ny)-Ny) )
            k = theta_comp_to_grid(kx,ky)

            if hasattr(func_ft,'__len__'):
                assert len(func_ft) == Nz
                self.fourier_map_ = numpy.asarray([f_ft(k) for f_ft in func_ft])
            else:
                if func is not None:
                    self.func_ = Nz*[func]
                self.func_ft_ = Nz*[func_ft]
                self.fourier_map_ = func_ft(k)
                
        elif func is not None:
            #no analytic fourier transform supplied: bin the data and do an fft
            
            theta_x = fftpack.ifftshift( dx * ( numpy.arange(2*Nx) - Nx ) )
            theta_y = fftpack.ifftshift( dy * ( numpy.arange(2*Ny) - Ny ) )
            theta = theta_comp_to_grid(theta_x,theta_y)
            
            if hasattr(func,'__len__'):
                assert len(func) == Nz
                self.func_ = func
                self.fourier_map_ = numpy.asarray( [dx*dy*fftpack.fft2(f(theta))
                                                    for f in self.func_] )
            else:
                self.func_ = Nz * [func]
                if func_ft is not None:
                    self.func_ft_ = Nz * [func_ft]
                self.fourier_map_ = dx*dy*fftpack.fft2(func(theta))

    def matvec(self,v):
        self.check_vec_(v,"%s.matvec" % self.__class__.__name__)

        v_L = self.view_as_Lens3D_vec(v)
                
        return_vec = self.view_as_Lens3D_vec( numpy.zeros(v_L.size,
                                                          dtype = complex) )
        return_vec.data_fft = self.fourier_map_ * v_L.data_fft
                
        return self.view_as_same_type(return_vec,v)

    def matvec_direct(self,v):
        """
        provided as a sanity check.  This implements the matvec method
        using a (very slow) direct convolution.  It should give the
        same result as matvec(), to within machine precision.
        """
        self.check_vec_(v,"%s.matvec" % self.__class__.__name__)
        v_L = self.view_as_Lens3D_vec(v)
        return_vec = self.view_as_Lens3D_vec( numpy.zeros(v_L.size,
                                                          dtype = v_L.dtype) )
        
        theta_plane = theta_comp_to_grid( self.dx_* numpy.arange(self.Nx),
                                          self.dy_* numpy.arange(self.Ny) )
        
        for iz in range(self.Nz):
            plane_i = v_L.lens_plane(iz)
            r_plane_i = return_vec.lens_plane(iz)
            func_i = self.func_[iz]
            
            for ix in range(self.Nx):
                for iy in range(self.Ny):
                    theta = theta_plane[ix,iy]
                    r_plane_i[ix,iy] = numpy.sum( plane_i * \
                                                  func_i(theta-theta_plane) )
            r_plane_i *= (self.dx_*self.dy_)
                    

        return self.view_as_same_type(return_vec,v)

    def transpose(self):
        # h(x) = (f*g)(x) = integral[ f(x')g(x-x')dx' ]
        # h(x[i]) = sum_j[ f(x[j])g(x[i]-x[j]) dx[j] ]
        #  define:
        #     F_j = f(x[j])
        #     H_i = h(x[i])
        #     G_ij = g(x[i]-x[j]) * dx
        #            (dx is the area of each pixel, in this case)
        #  and the matrix equation is
        #     H = G * F
        #  this means the transpose is GT where, by definition
        #     GT_ij = G_ji = g(x[j]-x[i]) = g[-(x[i]-x[j])]
        #      now define gt(x) such that
        #     GT_ij = gt(x[i]-x[j])
        #  then we see that
        #      gt(x) = g(-x)
        #
        # So we see:
        #  the argument of func for the transpose is
        #  opposite the argument for non-transpose.
        # In fourier space, this leads to G(k) -> G(-k)
        if self.func_ is None:
            new_func = self.func_
        elif (self.Nz==1) or (self.func_[0] is self.func_[1]):
            new_func = self.Nz * [lambda x: self.func_[0](-x)]
        else:
            new_func = [lambda x: f(-x) for f in self.func_]

        if self.func_ft_ is None:
            new_func_ft = self.func_ft_
        elif (self.Nz==1) or (self.func_ft_[0] is self.func_ft_[1]):
            new_func_ft = self.Nz * [lambda k: self.func_ft_[0](-k)]
        else:
            new_func_ft = [lambda k: F(-k) for F in self.func_ft_]

        return self.__class__(self.Nz, self.Nx, self.Ny,
                              self.dx_, self.dy_,
                              new_func, new_func_ft)
        

    def conj(self):
        if self.func_ is None:
            new_func = self.func_
        elif (self.Nz==1) or (self.func_[0] is self.func_[1]):
            new_func = self.Nz * [lambda x: numpy.conj(self.func_[0](x))]
        else:
            new_func = [lambda x: numpy.conj(f(x)) for f in self.func_]

        if self.func_ft_ is None:
            new_func_ft = self.func_ft_
        elif (self.Nz==1) or (self.func_ft_[0] is self.func_ft_[1]):
            new_func_ft = self.Nz * [lambda k: numpy.conj(self.func_ft_[0](k))]
        else:
            new_func_ft = [lambda k: numpy.conj(F(k)) for F in self.func_ft_]

        return self.__class__(self.Nz, self.Nx, self.Ny,
                              dx = self.dx_, dy = self.dy_,
                              func = new_func, func_ft = new_func_ft)

    def conj_transpose(self):
        if self.func_ is None:
            new_func = self.func_
        elif (self.Nz==1) or (self.func_[0] is self.func_[1]):
            new_func = self.Nz * [lambda x: numpy.conj(self.func_[0](-x))]
        else:
            new_func = [lambda x: numpy.conj(f(-x)) for f in self.func_]

        if self.func_ft_ is None:
            new_func_ft = self.func_ft_
        elif (self.Nz==1) or (self.func_ft_[0] is self.func_ft_[1]):
            new_func_ft = self.Nz * [lambda k: numpy.conj(self.func_ft_[0](-k))]
        else:
            new_func_ft = [lambda k: numpy.conj(F(-k)) for F in self.func_ft_]

        return self.__class__(self.Nz, self.Nx, self.Ny,
                              dx = self.dx_, dy = self.dy_,
                              func = new_func, func_ft = new_func_ft)
    
    def as_Lens3D_lp_mat(self):
        """
        multiply out each lensplane, and represent object as a 
        Lens3D_lp_mat object.
        """
        # so as to not duplicate code, we'll obtain this by
        # creating a Lens3D_lp_conv object for each
        #  lens plane
        
        if self.fourier_map_.ndim == 2:
            #single matrix for every lens plane:
            # compute it only once
            LP_op = Lens3D_lp_conv(1,self.Nx,self.Ny,
                                   self.dx_,self.dy_,None,None)
            LP_op.fourier_map_ = self.fourier_map_
            
            mat_list = LP_op.full_matrix

        else:
            #different matrix for each lens plane:
            # compute each one
            mat_list = []
            for i in range(self.Nz):
                #create an object for just one lens plane
                LP_op = Lens3D_lp_conv(1,self.Nx,self.Ny,
                                       self.dx_,self.dy_,None,None)
                
                LP_op.fourier_map_ = self.fourier_map_[i]

                mat_list.append(LP_op.full_matrix)
            
        return Lens3D_lp_mat(self.Nz,self.Nx,self.Ny,
                             mat_list=mat_list)

    def inverse(self, analytic=True):
        """
        compute the analytic inverse.  This is only the true inverse
        in the case of infinite limits, and infinitessimal pixel size
        
        can only be computed if transform is defined by the analytic
        fourier transform.
        """
        if analytic:
            if self.func_ft_ is None:
                raise ValueError, "Lens3D_lp_conv.inverse() : valid only when func_ft is supplied."
            elif (self.Nz==1) or (self.func_ft_[0] is self.func_ft_[1]):
                #print "just one inverse"
                try:
                    #if F is a ZeroProtectFunction
                    print "Lens3D: zeroprotect inverse"
                    new_func_ft = self.Nz * [self.func_ft_[0].I]
                except:
                    print "Lens3D: non-zeroprotect inverse"
                    new_func_ft = self.Nz * [lambda k: 1./self.func_ft_[0](k)]
            else:
                #print "multiple inverses"
                try:
                    #if F is a ZeroProtectFunction
                    new_func_ft = [F.I for F in self.func_ft_]
                except:
                    new_func_ft = [lambda k: 1./F(k) for F in self.func_ft_]
            
            return self.__class__(self.Nz,self.Nx,self.Ny,
                                  dx = self.dx_, dy = self.dy_,
                                  func = None, func_ft = new_func_ft)
        else: #compute the numerical inverse of each lens plane
            return self.as_Lens3D_lp_mat().inverse()


    @property
    def dtype(self):
        return complex


###############################################################################
class Lens3D_los_mat(Lens3D_matrix_base):
    """
    Line-of-sight matrix.  This is an operation which is the same
    along each line-of-sight.  For a data vector v_in, self.matvec(v_in)
    will return a vector v_out such that for each line of sight i,
    v_out[i] = dot(data,v_in[i])
    """
    def __init__(self,Nz_in,Nx,Ny,
                 Nz_out=None,data=None):
        Lens3D_matrix_base.__init__(self,Nz_in,Nx,Ny,
                                    Nz_out,Nx,Ny)
        assert data.shape == (self.Nz_out,self.Nz_in)
        self.data_ = data

    def matvec(self,v):
        self.check_vec_(v,"%s.matvec" % self.__class__.__name__)
        vL = self.view_as_Lens3D_vec(v)
        return_vec = self.view_as_Lens3D_vec(numpy.zeros(self.shape[0],
                                                         dtype=complex))
        #matrix multiply along each line of sight
        # To do this, we use numpy.dot and permute first 
        # two dimensions of v.data_ 
        #  (see documentation of numpy.dot)
        #return_vec.data = numpy.dot( self.data_,
        #                             vL.data.transpose(1,0,2) )

        #In theory, we should be able to perform the operation fully in
        # Fourier space.
        if vL.is_fourier():
            return_vec.data_fft = \
                                numpy.dot(self.data_,
                                          numpy.transpose(vL.data_fft,(1,0,2)))
        else:
            return_vec.data = numpy.dot( self.data_,
                                         numpy.transpose(vL.data,(1,0,2)))
                
        return self.view_as_same_type(return_vec,v)

    def transpose(self):
        return self.__class__(self.Nz_out,self.Nx,self.Ny,
                              self.Nz_in,self.data_.T)
    def conj(self):
        return self.__class__(self.Nz_in,self.Nx,self.Ny,
                              self.Nz_out,self.data_.conj())
    def conj_transpose(self):
        return self.__class__(self.Nz_out,self.Nx,self.Ny,
                              self.Nz_in,self.data_.conj().T)
    def inverse(self):
        if self.Nz_in == self.Nz_out:
            data_inv = numpy.linalg.inv(self.data_)
        else:
            data_inv = numpy.linalg.pinv(self.data_)
        
        return self.__class__(self.Nz_out,self.Nx,self.Ny,
                              self.Nz_in,data_inv)

def get_mat_rep(mat):
    try:
        mat_rep = mat.full_matrix
    except:
        mat_rep = numpy.empty( (mat.shape[0],mat.shape[1]),dtype=mat.dtype )
        for i in range(mat.shape[1]):
            v = numpy.zeros(mat.shape[1])
            v[i] = 1
            mat_rep[:,i] = mat.matvec(v)
            continue
            try:
                mat_rep[:,i] = mat.matvec(v)
            except:
                mat_rep[:,i] = numpy.dot( mat,v )
    return mat_rep

class Lens3D_multi(Lens3D_matrix_base):
    def __init__(self,*args):
        if len(args)==0:
            raise ValueError, "Lens3D_multi: must provide args"

        self.ops = args[::-1]
        self.debug = False
        
        N1,N2 = args[0].fullshape

        for arg in args[1:]:
            if N2!=arg.fullshape[0]: 
                raise ValueError, "Lens3D_multi: shape mismatch (problem with operator ordering?)"
            else:
                N2 = arg.fullshape[1]
                
        Lens3D_matrix_base.__init__(self,*(N2+N1))

    def matvec(self,v):
        vL = self.view_as_Lens3D_vec(v)
        for op in self.ops:
            vL = op.matvec(vL)
        return self.view_as_same_type(vL,v)

    def transpose(self):
        args = tuple([op.transpose() for op in self.ops])
        return Lens3D_multi(*args)

    def conj(self):
        args = tuple([op.conj() for op in self.ops[::-1]])
        return Lens3D_multi(*args)
    
    def conj_transpose(self):
        args = tuple([op.conj_transpose() for op in self.ops])
        return Lens3D_multi(*args)

    def inverse(self):
        args = tuple([op.inverse() for op in self.ops])
        return Lens3D_multi(*args)
            

class Lens3D_explicit_inverse(Lens3D_full_mat):
    """
    construct the explicit inverse of a Lens3D matrix.  This is accomplished
    by computing the matrix representation, then directly computing the
    inverse of the matrix using the scipy LAPACK interface.
    """
    def __init__(self,L3Dmat):
        print L3Dmat.shape
        M_rep = get_mat_rep(L3Dmat)
        
        if M_rep.shape[0] != M_rep.shape[1]:
            raise ValueError, "Lens3D_explicit_inverse : matrix must be square"
        
        Lens3D_full_mat.__init__(self,
                                 L3Dmat.Nz_in, L3Dmat.Nx_in, L3Dmat.Ny_in,
                                 L3Dmat.Nz_out, L3Dmat.Nx_out, L3Dmat.Ny_out,
                                 numpy.linalg.inv(M_rep))

class Lens3D_LinearOperator(Lens3D_matrix_base):
    """
    Wrap a scipy.sparse.linalg.LinearOperator object as a Lens3D object
    """
    def __init__(self,LinOp,Nz_in,Nx_in,Ny_in,
                 Nz_out=None,Nx_out=None,Ny_out=None,
                 rLinOp=None):
        Lens3D_matrix_base.__init__(self,Nz_in,Nx_in,Ny_in,
                                    Nz_out,Nx_out,Ny_out)
        self.LinOp = LinOp
        self.rLinOp = rLinOp
        assert LinOp.shape == self.shape
        if not (rLinOp is None):
            assert rLinOp.shape[::-1] == self.shape
        

    def matvec(self,v):
        factor = 1.0
        M = numpy.linalg.norm(self.view_as_normal_vec(v))
        if M>0.0 and M<1E-5:
            factor = 1./M
        ret = self.LinOp.matvec(factor*self.view_as_normal_vec(v))
        ret /= factor
        return self.view_as_same_type(ret,v)

    def rmatvec(self,v):
        if self.rLinOp is None:
            Lens3D_matrix_base.rmatvec(self,v)
        else:
            ret = self.rLinOp.matvec(self.view_as_normal_vec(v))
            return self.view_as_same_type(ret,v)

    def conj_transpose(self):
        if self.rLinOp is None:
            return Lens3D_matrix_base.conj_transpose(self)
        else:
            return self.__class__(self.rLinOp,
                                  self.Nz_in, self.Nx_in, self.Ny_in,
                                  self.Nz_out, self.Nx_out, self.Ny_out,
                                  self.LinOp)
        
                                 
def as_Lens3D_matrix(M):
    if issubclass( type(M),Lens3D_matrix_base ):
        return M
    else:
        raise NotImplementedError


if __name__ == "__main__":
    L = Lens3D_vector(2,3,4,numpy.arange(24))

    print "------------------------------"
    print "full data:"
    print L.data

    print "------------------------------"
    print "first redshift bin:"
    print L.lens_plane(0)

    print "------------------------------"
    print "second redshift bin:"
    print L.lens_plane(1)

    print "------------------------------"
    print "line-of-sight (2,1)"
    print L.line_of_sight(2,1)
    print "------------------------------"
    
    L.line_of_sight(2,1)[0] = 80*(1+1j)
    print L.lens_plane(0)[2,1]
