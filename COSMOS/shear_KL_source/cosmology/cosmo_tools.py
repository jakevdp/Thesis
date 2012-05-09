import numpy
from scipy import integrate, interpolate

def call_item_by_item(func):
    """
    Decorator for a function such that an array passed to
    it will be executed item by item.  Return value is the
    same type as the input value (list,ndarray,matrix,etc).

    also up-casts integers.
    """
    def new_func(self,val,*args,**kwargs):
        if type(val) in (int,long):
            val = float(val)
        v_array = numpy.asarray(val)
        v_raveled = v_array.ravel()
        retval = numpy.array([func(self,v,*args,**kwargs) for v in v_raveled],
                             dtype = v_array.dtype)
        retval.resize(v_array.shape)
        if type(val)==numpy.ndarray:
            return retval
        else:
            return type(val)(retval)
    return new_func

def call_as_array(func):
    """
    Decorator for a function such that an array passed to
    it will be executed in one step.  Return value is the
    same type as the input value (float,list,ndarray,matrix,etc).
    
    also up-casts integers.
    """
    def new_func(self,val,*args,**kwargs):
        if type(val) in (int,long):
            val = float(val)
        v_array = numpy.asarray(val)
        v_raveled = v_array.ravel()
        retval = func(self,v_raveled,*args,**kwargs)
        numpy.asarray(retval).resize(v_array.shape)
        if type(val)==numpy.ndarray:
            return retval
        else:
            return type(val)(retval)
    return new_func



class with_sampleable_methods:
    """
    Class which allows sampling and B-Spline fitting of class methods
    in derived classes.

    Example:

    #---------------------------------------------------------------
    class foo(with_sampleable_methods):
        def bar(self,x):
            return numpy.sin(x)
        #--
    #--

    F = foo()
    print F(4) #evaluates the function
    F.sample('bar',numpy.arange(0,10,0.1))
    print F(4) #evalueates a pre-computed B-spline of the function
    #---------------------------------------------------------------
    """
    class sampled_function:
        def __init__(self,func,x,*args,**kwargs):
            self.func = func
            self.x = x

            #assign function name
            if func.__name__ != None:
                self.__name__ = self.func.__name__ + \
                                " [Sampled to %i pts]" % len(x)
            else:
                self.__name__ = None

            #assign function doc string
            if func.__doc__ != None:
                self.__doc__ = "Sampled Function : \n\n" + self.func.__doc__
            else:
                self.__doc__ = None

            #set up the b-spline
            try:
                self.tck = interpolate.splrep(x,func(x,*args,**kwargs),s=0)
            except:
                self.tck = interpolate.splrep(x,func,s=0)
        def __call__(self,y):
            return interpolate.splev(y,self.tck,der=0)
    ###
        
    def sample(self,methodname,xrange,*args,**kwargs):
        if not hasattr(self,methodname):
            raise ValueError, methodname

        if self.is_sampled(methodname):
            self.unsample(methodname)

        tmp = getattr(self,methodname)





        setattr(self,methodname,
                with_sampleable_methods.sampled_function(tmp,xrange,
                                                         *args,**kwargs ) )

    def unsample(self,methodname):
        if self.is_sampled(methodname):
            tmp = getattr(self,methodname).func
            setattr( self,methodname,tmp )
        else:
            raise ValueError, "cannot unsample %s" % methodname
        


    def is_sampled(self,methodname):
        return getattr(self,methodname).__class__ == \
               with_sampleable_methods.sampled_function



class Cosmology(with_sampleable_methods):
    """
    Cosmology Class
    Used to automate cosmological computations.
    Note that all distance units come from Dh, so are in Mpc
    All times come from Th, and so are Gyr
    Defaults are WMAP-5 data
    """
    def __init__(self,
                 Om = 0.27,
                 Ol = 0.73,
                 Or = 8.4E-5,
                 w0 = -1.0,
                 w1 = 0.0,
                 h = 0.71,
                 sigma8 = 0.812,
                 T_CMB = 2.71):
        # COSMOLOGICAL PARAMETERS
        self.Om = Om
        self.Ol = Ol
        self.Or = Or
        self.Ok = 1. - Om - Ol - Or
        self.w0  = w0
        self.w1 = w1
        self.h  = h
        self.sigma8 = sigma8
        self.T_CMB = T_CMB

        self.__P0 = None
        
        # CONSTANTS
        self.c       = 2.9979E5    # km/s
        self.G       = 6.67259E-8  # cm^3 / g / s^2
        self.a_boltz   = 7.56576738E-15  # erg / cm^3 / K^4
        
        self.Msun    = 1.98892E30  # kg
        self.pc      = 3.085677E16 # m
        self.yr      = 31556926    # s 

        # OTHER PARAMETERS
        self.H0 = 100*h                   # km/s/Mpc - Hubble constant
        self.Th = self.pc / self.H0 / self.yr / 1e6 # Gyr  - hubble time
        self.Dh = self.c / self.H0        # Mpc - hubble distance

        if self.Or==0:
            self.z_rm = numpy.inf
        else:
            self.z_rm = (self.Om / self.Or) - 1
        
        if self.Om==0:
            self.a_rm = numpy.inf
        else:
            self.a_rm = self.Or / self.Om
    #end __init__

    #-----------------------------------------------------------------------
    def get_dict(self):
        """
        return a dictionary of arguments which can be used to
        re-create this object
        """
        keys = ['Om', 'Ol', 'Or', 'w0', 'w1', 'h', 'sigma8', 'T_CMB']
        return dict((key, getattr(self,key)) for key in keys)

    #-----------------------------------------------------------------------
    @call_item_by_item
    def __E(self,z):
        """
        dimensionless Hubble constant, used for integration routines
        Defined as in equation 14 from Hogg 1999, and modified
        for non-constant w parameterized linearly with z ( w = w0 + w1*z )
        """
        if z==numpy.inf:return numpy.inf
        return numpy.sqrt(self.Om * (1. + z)**3\
                          + self.Ok * (1.+ z)**2\
                          + self.Ol * numpy.exp(3*self.w1*z)\
                          * (1.+ z)**(3 * (1 + self.w0 - self.w1)))
    #end __E
    
    #-----------------------------------------------------------------------
    @call_as_array
    def H(self, z):
        """
        Hubble Constant at redshift z
        """
        return self.H0 * self.__E(z)
    #end H
    
    #-----------------------------------------------------------------------
    @call_as_array
    def w(self,z):
        """
        equation of state:
        w(z) = w0 + w1*z
        """
        return self.w0 + self.w1 * z
    #end w

    #-----------------------------------------------------------------------
    @call_as_array
    def a(self, z):
        """Scale factor at redshift z"""
        return 1. / (1. + z)
    #end a
    
    #-----------------------------------------------------------------------
    @call_item_by_item
    def Tl(self, z):
        """
        Lookback time
        Difference between the age of the Universe now and the age at z
        """
        f = lambda z: 1/(1.+z)/self.__E(z)
        I = integrate.quad(f, 0, z)
        return self.Th * I[0]
    #end Tl

    #-----------------------------------------------------------------------
    @call_item_by_item
    def Tu(self,z):
        """
        Age of the universe at redshift z
        """
        f = lambda z: 1/(1.+z)/self.__E(z)
        I = integrate.quad(f, z, integrate.Inf)
        return self.Th * I[0]
    #end Tu
    
    #-----------------------------------------------------------------------
    @call_item_by_item
    def Dc(self, z):
        """
        Line of sight comoving distance
        Remains constant with epoch if objects are in the Hubble flow
        """
        if z==0:
            return 0
        else:
            f = lambda z: 1.0/self.__E(z)
            I = integrate.quad(f, 0, z)
            return self.Dh * I[0]
    #end Dc

    #-----------------------------------------------------------------------
    @call_as_array
    def Dp(self,z):
        """
        Proper distance
        """
        return self.Dc(z)/(1.+z)
    #end Dp
    
    #-----------------------------------------------------------------------
    @call_as_array
    def Dm(self, z):
        """
        Transverse comoving distance
        At same redshift but separated by angle dtheta;
        Dm * dtheta is transverse comoving distance
        """
        sOk = numpy.sqrt(numpy.abs(self.Ok))
        if self.Ok < 0.0:
            return self.Dh * numpy.sin(sOk * self.Dc(z)/self.Dh ) / sOk
        elif self.Ok == 0.0:
            return self.Dc(z)
        else:
            return self.Dh * numpy.sinh(sOk * self.Dc(z)/self.Dh ) / sOk
    #end Dm
    
    #-----------------------------------------------------------------------
    @call_as_array
    def Da(self, z):
        """
        Angular diameter distance:
        Ratio of an objects physical transvserse size to its
        angular size in radians
        """
        return self.Dm(z) / (1.+ z)
    #end Da

    #-----------------------------------------------------------------------
    def Da12(self, z1, z2):
        """
        Angular diameter distance between objects at 2 redshifts
        Useful for gravitational lensing
        (eqn 19 in Hogg)
        """
        # does not work for negative curvature
        assert(self.Ok) >= -1E4

        # z1 < z2
        if (z2 < z1):
            z1,z2 = z2,z1

        Dm1 = self.Dm(z1)
        Dm2 = self.Dm(z2)
        Ok  = self.Ok
        Dh  = self.Dh

        return 1. / (1 + z2) * ( Dm2 * numpy.sqrt(1. + Ok * Dm1**2 / Dh**2)\
                                 - Dm1 * numpy.sqrt(1. + Ok * Dm2**2 / Dh**2) )
    #end Da12

    #-----------------------------------------------------------------------
    @call_as_array
    def Dl(self, z):
        """Luminosity distance"""
        return (1. + z) * self.Dm(z)
    #end Dl

    #-----------------------------------------------------------------------
    @call_item_by_item
    def D_hor(self,z):
        """
        Horizon Distance:
        returns the horizon distance at redshift z
        """
        f = lambda zp : 1 / self.__E(zp)
        I = integrate.quad(f,z,integrate.Inf)
        return self.Dh * I[0]
    #end D_hor

    #-----------------------------------------------------------------------
    @call_as_array
    def mu(self, z):
        """
        Distance Modulus
        """
        return 5. * numpy.log10(self.Dl(z) * 1E6) - 5.
    #end mu

    #-----------------------------------------------------------------------
    @call_as_array
    def T_BBKS(self,q):
        """
        Linear Transfer Function
        equation 15.82(1) from Peacock Cosmological Physics
        due to Bardeen et al 1986
        """
        return numpy.log(1+2.34*q)/(2.34*q) * \
               (1 + 3.89*q + (16.1*q)**2 + (5.46*q)**3 + (6.71*q)**4)**(-0.25)
    #end T_BBKS
    
    #-----------------------------------------------------------------------
    @call_as_array
    def P_BBKS(self,k,n=1.0):
        """
        Uses the transfer function defined in BBKS_Transfer to define an
        approximation to the linear power spectrum, appropriately
        normalized via sigma8.
        The primordial spectrum is assumed to be proportional to k^n
        
        Units of k are assumed to be Mpc^-1
        """
        #compute P0 if it is not yet computed
        
        if self.__P0 == None:
            self.__compute_P0()
        
        q = k/( self.Om * self.h**2)
        return self.__P0 * (k**n) * self.T_BBKS(q)**2
    #end P_BBKS

    #-----------------------------------------------------------------------
    def __compute_P0(self):
        """
        compute the power spectrum normalization based on the
        value of sigma8
        """
        self.__P0 = 1.0

        tophat = lambda kR: 3.0 * (numpy.sin(kR) - kR*numpy.cos(kR)) / (kR)**3
        
        I_dk = lambda k: k**2 * self.P_BBKS(k) * tophat(8*k)**2
        I = integrate.quad( I_dk, 0, integrate.Inf )
        self.__P0 = (self.sigma8**2) * (2 * numpy.pi**2) / I[0]
    #end __compute_P0

    #-----------------------------------------------------------------------
    def compute_sigma(self,windowfunc,z=0.0):
        """
        given a fourier-transformed window function (defined by the 1-parameter
        function windowfunc) this returns the average mass fluctuation within
        the given window, based on the power spectrum defined in self.P_BBKS
        """
        #divide power spectrum by a factor: for some reason, integration
        # does not converge for very large I_dk
        A = 1E4
        
        I_dk = lambda k: (1.0/A**2) * k**2 * self.P_BBKS(k) * windowfunc(k)**2
        I = integrate.quad( I_dk, 0, integrate.Inf)

        Gz = self.linear_growth_factor(z)
        
        return A * Gz * (1.+z) * numpy.sqrt( I[0] / (2*numpy.pi**2) )
        #return A * numpy.sqrt( I[0] / (2*numpy.pi**2) )
    #end compute_sigma
    
    #-----------------------------------------------------------------------
    @call_as_array
    def rho_crit(self,z=0):
        """
        calculate the critical (mass) density at z

        returns value in g/cm^3
        """
        return 3. / (8 * numpy.pi * self.G) *\
               (self.H(z) / 1E3 / self.pc)**2
    
    #-----------------------------------------------------------------------
    @call_as_array
    def e_crit(self,z=0):
        """
        calculate the critical (energy) density at z

        returns value in erg/cm^3
        """
        return 3*(1E5 * self.c)**2 / (8 * numpy.pi * self.G) *\
               (self.H(z) / 1E3 / self.pc)**2

    #-----------------------------------------------------------------------
    @call_as_array
    def lens_kernel(self,z,z_source):
        """
        returns the value of the lens kernel at a redshift z, given a
        source at redshift z_source

        units are strange: cm Mpc^2 / g
        """
        Ds = self.Dc(z_source)
        D = self.Dc(z)

        return 4*numpy.pi*self.G / (1E5 * self.c)**2 * self.Dh / self.__E(z) *\
               D*(Ds-D)/Ds / (1.+z)**2

    #-----------------------------------------------------------------------
    def kappa(self,epsilon,zs):
        """
        Returns the convergence kappa measured from a source at redshift zs,
        given an energy distribution epsilon, which takes redshift as an
        argument and returns the energy density in erg/cm^3.

        return value is in Mpc^2 s^-2
        """
        dK = lambda z: self.lens_kernel(z,zs) * epsilon(z)
        I = integrate.quad(dK,0,zs)

        return I[0]
        
    #-----------------------------------------------------------------------
    def kappa_background(self,zs):
        """
        Returns the convergence kappa due to the average background
        mass-energy density at intervening redshifts
        """
        dK = lambda z: self.lens_kernel(z,zs) * self.e_crit(z)
        I = integrate.quad(dK,0,zs)

        return I[0]
        
    
    #-----------------------------------------------------------------------
    @call_as_array
    def linear_growth_factor(self,z):
        """
        Solves the ODE for the linear growth of matter overdensity:
         d_k'' + 2Hd_k' - 3/2 Omega_m H^2 d_k = 0

        d_k(z) = G(z) d_k(z=0)

        uses boundary conditions G(z=0) = 1
                                 G(z=1100) = a(z=1100)
        """


        # y = [G(a),G'(a)]
        # y' = J * y
        # J is the Jacobian (gradient) matrix associated with the
        #   differential equation above

        if self.w1 != 0 or self.w0!=-1:
            raise ValueError, "linear_growth_factor defined only for w=-1"
            
        numerator = lambda a: -(3*self.Om*a**-3 \
                                + 4*self.Or*a**-4 \
                                + 2*self.Ok*a**-2)
        denominator = lambda a: 2*(self.Om*a**-3 \
                                   + self.Or*a**-4 \
                                   + self.Ok*a**-2 \
                                   + self.Ol)
        
        jacobian = lambda Y,a: [[0,1],
                                [3.*self.Om /2./a**2,
                                 -(3.+ numerator(a)/denominator(a))/a ,]]

        Yprime = lambda Y,a: numpy.dot(jacobian(Y,a),Y)
        
        a0 = 1./1100
        Y0 = [a0,1]
        a = 1./(1.+z)

        #need to sort in order of increasing a
        # will need to unsort at the end
        sort = numpy.argsort(a)
        unsort = numpy.argsort( numpy.arange(len(a))[sort] )

        a = a[sort]

        #need the first value to be a0
        start_index = 0
        if a[0] < a0:
            raise ValueError, "G(z) valid only for z < %.2g" % (1+1./a0)
        elif a[0] > a0:
            start_index = 1
            a = numpy.concatenate([[a0],a])

        #need to evaluate G(a=1) for normalization purposes
        end_index = len(a)
        if a[-1] > 1:
            raise ValueError, "G(z) not valid for z < 0"
        elif a[-1] == 1:
            pass
        else:
            a = numpy.concatenate([a,[1.0]])
        

        Y = integrate.odeint(Yprime,Y0,a,Dfun=jacobian)
        G =  Y[start_index:end_index,0][unsort]
        G0 = Y[-1,0]
        
        if numpy.shape(z) == ():
            G = G[0]

        return G/G0
    
    #-----------------------------------------------------------------------
    @call_item_by_item
    def dump(self,z=None):
        print '-----------------------------'
        print "Cosmology: "
        print "   Omega_m = %.4g" % self.Om
        print "   Omega_L = %.4g" % self.Ol
        print "   Omega_r = %.4g" % self.Or
        print "   H0      = %.4g km/s/Mpc" % self.H0
        print "   Dh      = %.4g Mpc" % self.Dh
        print "   e_crit  = %.4g erg/cm^3" % self.e_crit(0)
        print "   CMB dens= %.4g erg/cm^3" % (self.a_boltz * self.T_CMB**4)
        print "     (frac = %.4g)" %(self.a_boltz*self.T_CMB**4/self.e_crit(0))
        print ''
        print "   radiation-matter equality at z = %.4g" % self.z_rm
        print "   horizon distance at r-m equality:",
        print "%.4g Mpc" % self.D_hor(self.z_rm)
        print "   horizon distance today: %.4g Mpc" % self.D_hor(0)
        print ''
        if z!=None:
            print "For z = %.2f:" % z
            print "   Hubble Parameter H(z)          %.2f km/s/Mpc" % self.H(z)
            print '   Lookback time                  %.2f Gyr' % self.Tl(z)
            print '   Age of the universe            %.2f Gyr' % self.Tu(z)
            print '   Scale Factor a                 %.2f'     % self.a(z)
            print '   Comoving L.O.S. Distance (w)   %.2f Mpc' % self.Dc(z)
            print '   Angular diameter distance      %.2f Mpc' % self.Da(z)
            print '   Luminosity distance            %.2f Mpc' % self.Dl(z)
            print '   Distance modulus               %.2f mag' % self.mu(z)
        
        print '-----------------------------'
    #end dump
#end class Cosmology


  

if __name__ == "__main__":
    import pylab
    
    C = Cosmology()
    C.dump(1.0)

    a = numpy.linspace(1./1100,1.0,1000)
    z = (1./a-1)
    G = C.linear_growth_factor(z)

    pylab.plot(a,G)
    pylab.ylabel('G')
    pylab.xlabel('a')
    pylab.title("Linear Growth Factor")
    pylab.show()
