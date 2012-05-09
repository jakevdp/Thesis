"""
compute a few versions of the shear correlation function:
  xi_plus
  xi_minus  both as defined in Schneider 2001
  xi_circ   correlation between circular pixels
"""
import numpy as np
import pylab
from scipy import integrate,interpolate
from scipy.ndimage import filters
from scipy.special import j0, j1, jn

from .shear_power import Pspec
from .cosmology import Cosmology

class xi_plus:
    """
    xi_+ shear correlation function
    """
    def __init__(self, n, Nz=20,
                 Nell=5E4,cosmo=None,**kwargs):
        """
        n is a `zdist` object which encodes the redshift distribution
        """
        if cosmo is None:
            cosmo = Cosmology(**kwargs)
        self.ell = np.logspace(-1,8,Nell)
        P_ell = Pspec(self.ell, [n], Nz, cosmo)[0][0]
        self.F_ell = P_ell * self.ell

    def __integrand(self,theta):
        """
        Internal method to construct integrand.  
        Theta should be in radians
        """
        if theta==0:
            return self.F_ell
        else:
            return self.F_ell * j0(self.ell*theta)

    def __call__(self,theta):
        """
        Compute correlation function at an angle theta
        theta is assumed to be in arcmin
        """
        theta = np.pi * theta / 180. / 60.
        integrand = self.__integrand(theta)

        #integrand is in log space: x = log(l) is evenly spaced
        #                           dx = dl/l
        # integrate using simpson's rule in log space: 
        #     Bessel func oscillations are better behaved
        dx = np.log(self.ell[1]) - np.log(self.ell[0])
        return (0.5/np.pi) * integrate.simps(integrand*self.ell, dx=dx )

    def plot_integrand(self,theta):
        theta = np.pi * theta / 180. / 60.
        integrand = self.__integrand(theta)
        pylab.semilogx(self.ell,integrand*self.ell)

class xi_minus:
    """
    xi_- shear correlation function
    """
    def __init__(self, n, Nz=20,
                 Nell=5E4, cosmo=None, **kwargs):
        """
        n is a `zdist` object which encodes the redshift distribution
        """
        if cosmo is None:
            cosmo = Cosmology(**kwargs)
        self.ell = np.logspace(-1,8,Nell)
        P_ell = Pspec(self.ell, [n], Nz, cosmo)[0][0]
        self.F_ell = P_ell*self.ell

    def __integrand(self,theta):
        """
        Internal method to construct integrand.  
        Theta should be in radians
        """
        if theta==0:
            return 0.
        else:
            return self.F_ell * jn(4,self.ell*theta)

    def __call__(self,theta):
        """
        Compute correlation function at an angle theta
        theta is assumed to be in arcmin
        """
        theta = np.pi * theta / 180. / 60.
        integrand = self.__integrand(theta)

        #integrand is in log space: x = log(l) is evenly spaced
        #                           dx = dl/l
        # integrate using simpson's rule in log space: 
        #     Bessel func oscillations are better behaved
        dx = np.log(self.ell[1]) - np.log(self.ell[0])
        return (0.5/np.pi) * integrate.simps(integrand*self.ell, dx=dx )

    def plot_integrand(self,theta):
        theta = np.pi * theta / 180. / 60.
        integrand = self.__integrand(theta)
        pylab.semilogx(self.ell,integrand*self.ell)


class xi_circ:
    """
    correlation function between circular pixels
    """
    def __init__(self, n, theta_R=3.0,
                 Nz=20,Nell=5E4,cosmo=None,**kwargs):
        """
        n is a `zdist` object which encodes the redshift distribution
        theta_R: radius of pixels
         - should be in arcmin
        """
        if cosmo is None:
            cosmo = Cosmology(**kwargs)
        self.ell = np.logspace(-1,8,Nell)
        self.theta_R = theta_R * np.pi/180./60.
        
        P_ell = Pspec(self.ell,[n], Nz,cosmo)[0][0]
        self.F_ell = P_ell / self.ell * ( j1(self.ell*self.theta_R) )**2

    def __integrand(self,theta):
        """
        theta and theta_R should be in radians
        """
        if theta==0:
            return self.F_ell
        else:
            return self.F_ell *\
                j0(theta*self.ell)
            

    def __call__(self,theta):
        """
        Compute the correlation between two circular pixels of radius
        theta_R, separated by a distance theta
        theta and theta_R should be in arcmin
        """
        theta = theta * np.pi/180./60.

        integrand = self.__integrand(theta)

        #integrand is in log space: x = log(l) is evenly spaced
        #                           dx = dl/l
        # integrate using simpson's rule in log space: 
        #     J0 oscillations are better behaved
        dx = np.log(self.ell[1]) - np.log(self.ell[0])
        return (2./np.pi/self.theta_R**2) * \
            integrate.simps(integrand*self.ell, dx=dx )

    def plot_integrand(self,theta):
        theta = theta*np.pi/180./60.
        
        integrand = self.__integrand(theta)
        
        pylab.semilogx(self.ell,integrand*self.ell)


def Sample_xi(xi,Dmin,Dmax,N):
    """
    evaluates xi at N unequally-spaced locations between Dmin and
    Dmax, returning a function which wraps the scipy b-spline
    The range of values is roughly logarithmic, with denser sampling
    near theta=0 where the second derivative is larger.
    To make up for deficiencies in the integration, the result is 
    filtered at large angular scales.
    """
    breaks = 10**np.arange(-3.,4.)
    breaks = breaks[np.where( (breaks>Dmin) & (breaks<Dmax) )]
    breaks = np.concatenate( ([Dmin],breaks,[Dmax]) )
    N_segments = len(breaks)-1
    N_per_segment = int( N/N_segments )
    assert N_per_segment>1

    theta = np.concatenate( [np.linspace(breaks[i],
                                               breaks[i+1],
                                               N_per_segment+1)[:-1] \
                                    for i in range(N_segments)] )

    corr = np.asarray( [xi(t) for t in theta] )
    i = theta.searchsorted(1.0)

    #smooth larger scales
    corr_g = filters.gaussian_filter(corr[i:],1)
    corr[i:] = corr_g

    #create interpolation function
    spl = interpolate.splrep(theta,corr)
    def ret(x):
        return interpolate.splev(x,ret.spl)
    ret.spl = spl

    return ret

if __name__ == '__main__':
    raise ValueError, "change n to a zdist object"
    n = lambda z,z0=0.5: z**2 * np.exp(-(z/z0)**1.5)
    zlim = (0,3)

    cosmo = Cosmology(Om=0.3,
                      Ol=0.7,
                      Or=0.0,
                      sigma8=0.82)

    theta = np.logspace(0,2.5,100)
    xi_p = xi_plus(n,Nz=20,cosmo = cosmo)
    xi_m = xi_minus(n,Nz=20,cosmo = cosmo)
    xi_R = xi_circ(n,theta_R=3.0,Nz=20,cosmo = cosmo)
    
    xi_p_t = np.asarray( [xi_p(t) for t in theta] )
    xi_m_t = np.asarray( [xi_m(t) for t in theta] )
    xi_R_t = np.asarray( [xi_R(t) for t in theta] )

    pylab.figure()
    pylab.loglog(theta,xi_p_t,label=r'$\xi_+$')
    pylab.loglog(theta,xi_m_t,label=r'$\xi_-$')
    pylab.loglog(theta,xi_R_t,label=r'$\xi_R$')

    pylab.legend()

    pylab.xlabel(r'$\theta\ \rm{(arcmin)}$')
    pylab.ylabel(r'$\xi(\theta)$')
    
    pylab.show()
    
