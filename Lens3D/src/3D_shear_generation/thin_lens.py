import numpy
import pylab
import matplotlib
import sys
sys.path.append('../')
from cosmo_tools import Cosmology
from Lens3D import Lens3D_vector, theta_comp_to_grid, theta_grid_to_comp, imshow_Lens3D

ARCMIN_TO_RAD = numpy.pi / 180. /60.
C_KM_S = 2.9979E5
G_cgs = 6.673E-8      #gravitational constant in cm^3/ g / s^2
c_cgs = 2.99792458E10 #speed of light in cm/s
Mpc_cm = 3.08568E24   #centimeters in a megaparsec
Msun_g = 1.98892E33   #Mass of the sun in grams

"""
thin_lens
Tools for creating thin lens matter distributions, and calculating
the corresponding shear/convergence maps.

In all cases, theta_1 is the x-component of the image, and theta_2
is the y-component of the image.

Arrays are structured such that A[i1,i2] is the value at location
theta_1[i1], theta_2[i2]

All angular locations are stored in complex form:
 theta = theta_1 + i*theta_2
Inputs are assumed to be in arcmin, and converted to radians
behind the scenes.

"""

def call_item_by_item(func):
    """
    Decorator for a function such that an array passed to
    it will be executed item by item.  Return value is the
    same type as the input value (list,ndarray,matrix,etc).

    also up-casts integers.
    """
    def new_func(self,val,*args,**kwargs):
        val_array = numpy.asarray(val)
        if val_array.size == 1:
            return func(self,val,*args,**kwargs)
        ret_array = numpy.zeros(val_array.shape)

        v_flat = val_array.flat
        r_flat = ret_array.flat

        for i in range(val_array.size):
            r_flat[i] = func(self,v_flat[i],*args,**kwargs)
        return ret_array
    return new_func
    

def theta_extent(theta):
    """
    this is needed to make pylab.imshow() line up with pylab.contour()
    """
    Ntheta1 = theta.shape[0]
    Ntheta2 = theta.shape[1]
    
    theta_1_min  = theta.real.min()
    theta_1_max  = theta.real.max()
    #theta_1_min -= 0.5 * (theta_1_max-theta_1_min)/(Ntheta1-1)
    #theta_1_max += 0.5 * (theta_1_max-theta_1_min)/(Ntheta1-1)
    
    theta_2_min  = theta.imag.min()
    theta_2_max  = theta.imag.max()
    #theta_2_min -= 0.5 * (theta_2_max-theta_2_min)/(Ntheta2-1)
    #theta_2_max += 0.5 * (theta_2_max-theta_2_min)/(Ntheta2-1)
    
    return (theta_1_min,theta_1_max,
            theta_2_min,theta_2_max)
    


class density_distribution(object):
    """
    base class for density profiles
    stores a redshift, a position, and a cosmology object
    """
    C = Cosmology(0.27,0.7299)

    @staticmethod
    def set_cosmo(cosmo):
        C = cosmo
    
    def __init__(self):
        raise NotImplementedError

    def gamma_M(self,theta,zs):
        raise NotImplementedError

    def gamma(self,theta,zs):
        raise NotImplementedError

    def kappa(self,theta,zs):
        raise NotImplementedError

    def Sigma(self,theta,z1,z2):
        raise NotImplementedError

    def plot_gammakappa(self,theta,zs,normalize=True,n_bars=15):
        """
        theta - 2D array of complex numbers
        """
        theta1 = numpy.unique(theta.real)
        theta2 = numpy.unique(theta.imag)
        extent = ( theta1[0], theta1[-1],
                   theta2[0], theta2[-1] )

        gamma = self.get_gamma_vector(theta1, theta2, [zs])
        kappa = self.get_kappa_vector(theta1, theta2, [zs])
        
        kappa.imshow_lens_plane(0,extent=extent,label='\kappa')
        gamma.fieldplot_lens_plane(0,extent=extent,
                                   n_bars=n_bars,
                                   normalize=normalize)
        
        pylab.title("shear and convergence")
        

    def plot_gamma(self,theta,zs,
                   n_bars=15,normalize = True):
        """
        theta - 2D array of complex numbers
        """
        theta1 = numpy.unique(theta.real)
        theta2 = numpy.unique(theta.imag)
        extent = ( theta1[0], theta1[-1],
                   theta2[0], theta2[-1] )

        gamma = self.get_gamma_vector(theta1, theta2, [zs])
        
        gamma.fieldplot_lens_plane(0,extent=extent,
                                   n_bars=n_bars,
                                   normalize=normalize)
        
        pylab.title("shear")
                                                  

    def plot_kappa(self,theta,zs,numlevels=10):
        """
        theta - 2D array of complex numbers
        """
        theta1 = numpy.unique(theta.real)
        theta2 = numpy.unique(theta.imag)
        extent = ( theta1[0], theta1[-1],
                   theta2[0], theta2[-1] )

        kappa = self.get_kappa_vector(theta1, theta2, [zs])
        
        kappa.imshow_lens_plane(0,extent=(0,1,0,1),label='\kappa')
        kappa.contour_lens_plane(0,extent=(0,1,0,1),
                                 nlevels=numlevels, colors='g' )

        pylab.title('convergence')

    def plot_Sigma(self,theta,z1,z2,numlevels=10):
        """
        theta - 2D array of complex numbers
        """
        theta1 = numpy.unique(theta.real)
        theta2 = numpy.unique(theta.imag)
        extent = ( theta1[0], theta1[-1],
                   theta2[0], theta2[-1] )

        Sigma = self.get_Sigma_vector(theta1, theta2, (z1,z2))
        
        Sigma.imshow_lens_plane(1,extent=(0,1,0,1),label='\Sigma')
        Sigma.contour_lens_plane(1,extent=(0,1,0,1),
                                 nlevels=numlevels, colors='g' )
        
        pylab.title(r"$\rm{projected\ density}\ (M_\odot/\rm{arcsec^2})$")

    def get_gamma_vector(self,theta1,theta2,zrange):
        try:
            Nz = len(zrange)
        except:
            zrange = [zrange]
            Nz = 1
        N1 = len(theta1)
        N2 = len(theta2)

        vec_data = numpy.zeros( (Nz,N1,N2), dtype = complex )

        theta = theta_comp_to_grid(theta1,theta2)
        for i in range(Nz):
            vec_data[i] = self.gamma(theta,zrange[i])

        i = numpy.where(numpy.isnan(vec_data))
        vec_data[i] = 0

        return Lens3D_vector(Nz,N1,N2,vec_data)

    def get_kappa_vector(self,theta1,theta2,zrange,
                         n=1):
        try:
            Nz = len(zrange)
        except:
            zrange = [zrange]
            Nz = 1
        N1 = len(theta1)
        N2 = len(theta2)

        vec_data = numpy.zeros( (Nz,N1,N2), dtype = complex )

        theta = theta_comp_to_grid(theta1,theta2)
        if n==1:
            for i in range(Nz):
                vec_data[i] = self.kappa(theta,zrange[i])
        else:
            dx = theta1[1]-theta1[0]
            dy = theta2[1]-theta2[0]
            for j in range(N1):
                for k in range(N2):
                    t = theta_comp_to_grid( numpy.linspace(theta1[j]-0.5*dx,
                                                           theta1[j]+0.5*dx,
                                                           n),
                                            numpy.linspace(theta2[k]-0.5*dy,
                                                           theta2[k]+0.5*dy,
                                                           n) )
                    for i in range(Nz):
                        vec_data[i,j,k] = numpy.mean(self.kappa(t,zrange[i]))

        return Lens3D_vector(Nz,N1,N2,vec_data)

    def get_Sigma_vector(self,theta1,theta2,zrange,
                         n=1):
        """
        n tells how many points to evaluate within the pixel
        there will be n^2 evaluations per pixel
        """
        try:
            Nz = len(zrange)
        except:
            zrange = [zrange]
            Nz = 1
        N1 = len(theta1)
        N2 = len(theta2)

        assert n>0

        #make zrange[-1] = 0.0
        zrange = numpy.concatenate( (zrange,[0.0]) )

        vec_data = numpy.zeros( (Nz,N1,N2), dtype = complex )

        theta = theta_comp_to_grid(theta1,theta2)
        if n==1:
            for i in range(Nz):
                vec_data[i] = self.Sigma(theta,zrange[i-1],zrange[i])
        else:
            dx = theta1[1]-theta1[0]
            dy = theta2[1]-theta2[0]
            for j in range(N1):
                for k in range(N2):
                    t = theta_comp_to_grid( numpy.linspace(theta1[j]-0.5*dx,
                                                           theta1[j]+0.5*dx,
                                                           n),
                                            numpy.linspace(theta2[k]-0.5*dy,
                                                           theta2[k]+0.5*dy,
                                                           n) )
                    for i in range(Nz):
                        vec_data[i,j,k] = numpy.mean(self.Sigma(t,zrange[i-1],
                                                                zrange[i]))

        if numpy.any(numpy.isinf(vec_data)):
            raise ValueError, "get_Sigma_vector : infinity encountered"

        return Lens3D_vector(Nz,N1,N2,vec_data)
    
    def write_to_file(self,filename,
                      z_range,theta1_range,theta2_range = None):
        if(theta2_range == None):
            theta2_range = theta1_range

        of = open(filename,'w')
        of.write('#positions measured in arcmin:\n')
        of.write('#redshift  theta1  theta2  Sigma  kappa  gamma1  gamma2\n')

        theta = theta_comp_to_grid(theta1_range,theta2_range)
        
        for i in range(len(z_range)):
            z = z_range[i]
            if i==0:
                z0 = 0
            else:
                z0 = z_range[i-1]
            kappa = self.kappa(theta,z)
            gamma = self.gamma(theta,z)
            Sigma = self.Sigma(theta,z0,z)
            
            for j in range(theta.shape[0]):
                for k in range(theta.shape[1]):
                    of.write('%.3f %.3f %.3f %.6g %.6g %.6g %.6g\n' % \
                             (z, theta.real[j,k], theta.imag[j,k],
                              Sigma[j,k],
                              kappa[j,k],
                              gamma.real[j,k], gamma.imag[j,k]) )
                #end for
            #end for
        #end for
        of.close()
        
                             
        
class symmetric_distribution(density_distribution):
    """
    base class for symmetric distributions
    """
    def __init__( self,z,pos,C=None,**kwargs ):
        self.z = z
        self.pos = complex( pos )
        
        if C:
            self.set_cosmo(C)
        else:
            self.set_cosmo(Cosmology(**kwargs))

class SIS(symmetric_distribution):
    """singular isothermal sphere"""
    def __init__(self, z, sig,
                 theta,C=None, **kwargs):
        """
        z is redshift of the cluster, sig is km/s, C is a cosmology object
        theta is a complex angle (theta_1 + i*theta_2) in arcmin
        """
        symmetric_distribution.__init__(self,z,theta,C,**kwargs)
        self.sig = sig / C_KM_S #store sig in units of c

    def gamma_M(self,theta,zs):
        if zs <= self.z:
            return 0*theta
        Ds = self.C.Da(zs)
        Dds = self.C.Da12(self.z,zs)
        theta_c = ( theta-self.pos ) * ARCMIN_TO_RAD
        
        factor = 2*numpy.pi*self.sig**2 * Dds / Ds
        
        return factor / abs(theta_c)

    def gamma(self,theta,zs):
        """
        compute the complex shear at theta,z
        
        theta is a complex angle (theta_1 + i*theta_2) in arcmin
        
        return the complex shear (gamma_1 + i*gamma_2)
        (form is from Bartelmann 2001, eq 3.18)
        """
        if zs <= self.z:
            return 0*theta
        Ds = self.C.Da(zs)
        Dds = self.C.Da12(self.z,zs)
        theta_c = ( theta-self.pos ) * ARCMIN_TO_RAD

        return -2*numpy.pi*self.sig**2 * Dds / Ds * theta_c**2/abs(theta_c)**3

    def kappa(self,theta,zs):
        """
        compute the real convergence at theta,z
        
        theta is a complex angle (theta_1 + i*theta_2) in arcmin
        
        return the convergence kappa
        (form is from Bartelmann 2001, eq 3.17)
        """
        if zs <= self.z:
            return 0*theta.real
        Ds = self.C.Da(zs)
        Dds = self.C.Da12(self.z,zs)
        theta_c = ( theta-self.pos ) * ARCMIN_TO_RAD

        return 2*numpy.pi*self.sig**2 * Dds / Ds / abs(theta_c)

    def Sigma(self,theta,z1,z2,units='Msun/Mpc^2'):
        """
        Return the projected density at position theta
          between redshifts z1...z2
        theta is a complex angle (theta_1 + i*theta_2) in arcmin

        units can be one of
            'Msun/Mpc^2'         (solar masses per square megaparsec)
            'g/cm^2' <--> 'cgs'  (cgs units: grams per square centimeter)
        """
        if not (self.z > z1 and self.z <= z2):
            return 0.0 * theta.real
        
        # rho(r) = sig_v^2 / (2pi G r^2 )  [ r is the 3D radial coordinate ]
        #
        #Integrating over z gives
        #
        #  Sigma(R) = sig_v^2 / (2 G R)     [ R is the 2D projected radius ]
        #
        #We must be careful to treat the units correctly.
        # note that sig_v is in units of c

        theta_c = abs(theta-self.pos) * ARCMIN_TO_RAD
        R =  theta_c * self.C.Da(self.z) #projected distance (Mpc)

        G_c2 = G_cgs / c_cgs / c_cgs # G/c^2 in cm/g
        G_c2 *= Msun_g/Mpc_cm  #G/c^2 in Mpc/Msun
        
        Sigma = 0.5 * self.sig**2 / G_c2 / R #Sigma in Msun/Mpc^2
        if units=='Msun/Mpc^2':
            return Sigma
        elif units=='g/cm^2' or units=='cgs':
            return Sigma * Msun_g/Mpc_cm/Mpc_cm
        else:
            raise ValueError, "Sigma: units = '%s' not recognized" % units
        
        

class NFW(symmetric_distribution):
    """Navarro Frenk & White"""
    def __init__(self, z, theta,
                 Mvir, rvir, alpha,
                 c = None,
                 C=None, **kwargs):
        """
        z is redshift of the cluster,
        theta is a complex angle (theta_1 + i*theta_2) in arcmin
        C is a cosmology object
        rvir is the virial radius in Mpc
        Mvir is the virial mass in Msun
        alpha is the NFW parameter
        c is the concentration parameter
        """
        #compute c = concentration parameter by eq 12 in Takada & Jain 2003
        if c is None:
            c0,beta = 9.0,0.3
            Mstar = 1E10
            c = c0/(1.+z) * (Mvir / Mstar)**(-beta)
        symmetric_distribution.__init__(self,z,theta,C,**kwargs)
        self.c = c
        self.Mvir = Mvir
        self.rvir = rvir
        self.alpha = alpha
        self.f = 1./(numpy.log(1.+c) - c/(1.+c))
        #compute theta_vir in radians
        #  (note that Da(z) is given in Mpc, as is rvir)
        self.theta_vir = self.rvir / self.C.Da(z)

    @call_item_by_item
    def G_(self,x):
        """
        Equation 17 from Takada & Jain 2003
        """
        sqrt = numpy.sqrt
        log = numpy.log
        arccosh = numpy.arccosh
        arccos = numpy.arccos
        
        c = 1.*self.c
        x = 1.*x
        x2 = 1.0*x*x
        c2 = 1.0*c*c
        if x<1:
            return 1./(x2*(1.+c)) * ( (2.-x2)*sqrt(c2-x2)/(1.-x2) - 2*c) \
                   + (2./x2)*log(x*(1.+c)/(c+sqrt(c2-x2))) \
                   + (2.-3*x2)/(x2*(1.-x2)**1.5)*arccosh( (x2+c)/(x*(1.+c)) )
        elif x==1:
            return 1./(3*(1.+c)) * ( (11*c+10)*sqrt(c2-1)/(1.+c) - 6*c ) \
                   + 2*log( (1.+c)/(c+sqrt(c2-1)) )
        elif x <= c:
            return 1./(x2*(1.+c)) * ( (2.-x2)*sqrt(c2-x2)/(1.-x2) - 2*c ) \
                   + (2./x2)*log(x*(1.+c)/(c+sqrt(c2-x2))) \
                   - (2.-3*x2)/(x2*(x2-1.)**1.5)*arccos( (x2+c)/(x*(1.+c)) )
        else:
            return 2./x2/self.f

    @call_item_by_item
    def F_(self,x):
        """
        equation 27 from Takada & Jain 2003 (MNRAS 340:580)
        """
        c = self.c
        c2 = 1.*c*c
        x2 = 1.*x*x
        sqrt = numpy.sqrt
        arccosh = numpy.arccosh
        arccos = numpy.arccos

        if x<1:
            return - sqrt(c2-x2)/((1.-x2)*(1.+c)) \
                   + (1.-x2)**(-1.5)*arccosh( (x2+c)/(x*(1.+c)) )
        
        elif x==1:
            return sqrt(c2-1)/(3.*(1.+c)) * (1. + 1./(1.+c))
        
        elif x<=c:
            return - sqrt(c2-x2)/((1.-x2)*(1.+c)) \
                   - (x2-1.)**(-1.5)*arccos( (x2+c)/(x*(1.+c)) )

        else:
            return 0.0

    def gamma(self,theta,zs):
        theta_c = ( theta-self.pos ) * ARCMIN_TO_RAD
        return -self.gamma_M(theta,zs) * (theta_c / numpy.abs(theta_c) )**2

    def gamma_M(self,theta,zs):
        """
        compute the shear amplitude for a source at zs
        from equation 16-17 in Takada&Jain
        """
        if zs <= self.z:
            return 0*theta
        D = self.C.Da(self.z)
        Ds = self.C.Da(zs)
        Dds = self.C.Da12(self.z,zs)
        
        theta_c = abs( theta-self.pos ) * ARCMIN_TO_RAD

        factor = 2.*G_cgs/(c_cgs*c_cgs)*(1.+self.z) * D * Dds / Ds
        #factor now has units cm*Mpc/g
        # convert to Mpc^2 / Msun
        #    (because these are the units of Mvir and rvir)
        factor *= Msun_g / Mpc_cm

        factor *= self.Mvir * self.f * self.c**2 / self.rvir**2

        return factor * self.G_(self.c*theta_c/self.theta_vir)

    def kappa(self,theta,zs):
        """
        compute the real convergence at theta,z
        
        theta is a complex angle (theta_1 + i*theta_2) in arcmin
        
        return the convergence kappa
        """
        if zs <= self.z:
            return 0*theta.real
        D = self.C.Da(self.z)
        Ds = self.C.Da(zs)
        Dds = self.C.Da12(self.z,zs)
        theta_c = ( theta-self.pos ) * ARCMIN_TO_RAD

        factor = 4*numpy.pi*G_cgs/(c_cgs*c_cgs)*(1.+self.z) * D * Dds / Ds
        #factor is cm*Mpc/g
        # convert to Mpc^2 / Msun
        factor *= Msun_g/Mpc_cm
        return factor * self.Sigma(theta,0,zs,'Msun/Mpc^2')

    def Sigma(self,theta,z1,z2,units='Msun/Mpc^2'):
        """
        Return the projected density at position theta
          between redshifts z1...z2
        theta is a complex angle (theta_1 + i*theta_2) in arcmin

        units can be one of
            'Msun/Mpc^2'         (solar masses per square megaparsec)
            'g/cm^2' <--> 'cgs'  (cgs units: grams per square centimeter)
        """
        if not (self.z > z1 and self.z <= z2):
            return 0.0 * theta.real
        
        theta_c = abs(theta-self.pos) * ARCMIN_TO_RAD

        x = (self.c*theta_c/self.theta_vir)

        #compute Sigma in Msun/Mpc^2
        Sigma = self.Mvir * (self.f * self.c * self.c) \
                / (2 * numpy.pi * self.rvir * self.rvir) \
                * self.F_(x)
        
        if units == 'Msun/Mpc^2':
            return Sigma
        elif units in ['g/cm^2','cgs']:
            return Sigma * Msun_g / Mpc_cm**2
        else:
            raise ValueError, "units not recognized"

class ProfileSet(density_distribution):
    """
    Holds multiple density_profile objects
    """
    def __init__(self,*args):
        self.L = list( args )

    def add(self,dist):
        self.L.append(dist)

    def gamma_M(self,theta,zs):
        return abs( self.gamma(theta,zs) )

    def gamma(self,theta,zs):
        return sum( [profile.gamma(theta,zs) for profile in self.L] )

    def kappa(self,theta,zs):
        return sum( [profile.kappa(theta,zs) for profile in self.L] )

    def Sigma(self,theta,z1,z2):
        return sum( [profile.Sigma(theta,z1,z2) for profile in self.L] )

def read_density_file(filename):
    """
    returns z, theta1, theta2,.Sigma, kappa, gamma
     z, theta1, and theta2 are lists of values
     Sigma, kappa, and gamma are Lens3D_vector objects.
    """
    L = []
    for line in open(filename):
        line = line.strip()
        if len(line)==0 or line[0]=='#':
            continue
        L.append(map(float,line.split()))
    L = numpy.asarray(L)
    z = numpy.unique(L[:,0])
    theta1 = numpy.unique(L[:,1])
    theta2 = numpy.unique(L[:,2])

    Nz = len(z)
    N1 = len(theta1)
    N2 = len(theta2)

    if not numpy.all(L[:N1+N2,0] == z[0]):
        print "warning: read_density_file: contents may be out of order"

    if not numpy.all(L[:N2,1] == theta1[0]):
        print "warning: read_density_file: contents may be out of order"

    if not numpy.all(L[:N2,2] == theta2):
        print "warning: read_density_file: contents may be out of order"

    Sigma = Lens3D_vector(Nz,N1,N2,L[:,3])
    kappa = Lens3D_vector(Nz,N1,N2,L[:,4])
    gamma = Lens3D_vector(Nz,N1,N2,L[:,5] + 1j*L[:,6])
    
    return z,theta1,theta2,Sigma,kappa,gamma
    #kappa = Lens3D_vector(Nz,N1,N2,L[:,3])
    #gamma = Lens3D_vector(Nz,N1,N2,L[:,4] + 1j*L[:,5])
    #return z,theta1,theta2,kappa,gamma
        

if __name__ == '__main__':
    P = NFW(0.3, 10+10j, 
            Mvir = 1E14,
            rvir = 1.0, alpha = 1, c = 10.0)
    
    print P.kappa(10+10j,0.3)
    print P.gamma_M(10+10j,0.3)
    print P.Sigma(10+10j,0.29,0.31)
