import numpy
import pylab
from scipy import integrate,special

from Lens3D import ZeroProtectFunction

import sys, os
dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append( os.path.join(dirname,'PowerSpectrum') )
from halofit import PowerSpectrum
from scipy.special import j0,j1

ARCMIN_TO_RAD = numpy.pi / 180. /60.

def integrate_onecall(f,xmin,xmax,N=1000):
    """
    integrate with simpson's rule on fixed grid
    of N points.  This is useful when it is very
    inefficient to call f(x) value-by-value
    """
    x = numpy.linspace(xmin,xmax,N)
    return integrate.simps( f(x),dx=x[1]-x[0] )

def integrate_log(f,xmin,xmax,imethod = integrate_onecall,**kwargs):
    """
    integrate f in log space:
      int{ f(x) dx }  =>  int{ g(y) dy }
        with y = log(x)

    imethod is the integration method.  kwargs are passed to this
    """
    f_star = lambda y: numpy.exp(y)*f(numpy.exp(y))
    return imethod(f_star,numpy.log(xmin),numpy.log(xmax),**kwargs)

class P_2D(ZeroProtectFunction):
    """
    implement equation 41 from Simon 2009, using a
    trapezoidal integration scheme.
    
    P_2D(ell) = 1/A^2 * int_{w0}^{w1} dw fk(w)^2 * P_3D(ell/fk(w),z)
    with A = int_{w0}^{w1} dw fk(w)^2
    
    The Simon 2009 form incorrectly uses A = (w1-w0)
    For comparison, this is implemented by setting use_ST_form=True
    
    Because the power spectrum for ell=0 is undefined, so we make
    this a subclass of ZeroProtectFunction.
    """
    def __init__(self,zmin,zmax,Nz,cosmo,use_ST_form=True):
        """
        if use_ST_form is true, use the form of P_2D from Simon Taylor 09
        otherwise, use the form I derived (normalized correctly).
        """
        self.Nz = Nz
        z = numpy.linspace(zmin,zmax,Nz)
        self.PSpecs = [PowerSpectrum(zi) for zi in z]
        self.fkw = [cosmo.Da(zi) for zi in z]
        w = [cosmo.Dc(zi) for zi in z]
        dw = numpy.diff(w)
        
        #weights are set up for trapezoidal integration
        # each weight is the weight associated with a 3D
        # power spectrum at a given redshift
        self.weights = numpy.zeros(Nz)
        self.weights[:-1] += 0.5*dw
        self.weights[1:] += 0.5*dw

        #put fkw factors in weights to speed repeated computations
        if use_ST_form:
            self.weights /= numpy.asarray(self.fkw)**2
            self.weights /= (w[-1]-w[0])**2
        else:
            self.weights *= numpy.asarray(self.fkw)**2
            A = numpy.sum(self.weights) #normalization of qi window functions
            self.weights /= A**2
            
        #doing a function call on this class will call
        # self.evaluate, and where the argument is zero,
        # replace the result with 0.0
        ZeroProtectFunction.__init__(self,self.evaluate,0.0)

    def evaluate(self,ell):
        ellshape = ell.shape

        if len(ellshape)==0:
            ell.resize([1])
        
        integrand = numpy.asarray([self.PSpecs[i].D2_NL(ell/self.fkw[i]) \
                                   * 2*numpy.pi**2/(ell/self.fkw[i])**3
                                   for i in range(self.Nz)] )
        if len(ellshape) == 0:
            integrand.resize(self.Nz)

        #return numpy.dot(self.weights,integrand)
        #use broadcasting so that  this works for any shape of input.
        wgt_shape = [len(self.weights)]+(len(integrand.shape)-1)*[1]
        ret = ( self.weights.reshape(wgt_shape) * integrand ).sum(0)
        
        #print ret.shape
        #print ret[:3,:3]
        #print "ret limits:",numpy.min(ret),numpy.max(ret)
        #print "done with evaluate"

        return ret
        
    def evaluate_old(self,ell):
        ell = numpy.asarray(ell)
        ellshape = ell.shape

        integrand = numpy.asarray([self.PSpecs[i].D2_NL(ell/self.fkw[i]) \
                                   * 2*numpy.pi**2/(ell/self.fkw[i])**3
                                   for i in range(self.Nz)] )

        if len(ellshape)==0:
            ell.resize([1])

        #we will have infinity where ell is zero: take care of this.
        i_zero = numpy.where(ell==0)
        if len(i_zero[0]) == 0:
            i_zero = tuple([numpy.arange(0)] + list(i_zero))
        else:
            i_zero = tuple([numpy.arange(self.Nz)] + list(i_zero))

        integrand = numpy.asarray([self.PSpecs[i].D2_NL(ell/self.fkw[i]) \
                                   * 2*numpy.pi**2/(ell/self.fkw[i])**3
                                   for i in range(self.Nz)] )

        integrand[i_zero] = 0.0
        if len(ellshape) == 0:
            integrand.resize(self.Nz)

        #return numpy.dot(self.weights,integrand)
        #use broadcasting so that  this works for any shape of input.
        wgt_shape = [len(self.weights)]+(len(integrand.shape)-1)*[1]
        return ( self.weights.reshape(wgt_shape) * integrand ).sum(0)

class S_DD:
    """
    implement equation 40 from Simon 2009
    Integrate P_2D above over the window function
    """
    def __init__(self,zmin,zmax,Nz,cosmo,theta_s,window = 'circ',
                 integ_lmin = 1E-2,
                 integ_lmax = 2E4):
        """
        theta_s is the pixel radius in arcmin
        """
        if zmin==0:
            zmin = 1E-3
        self.window = window
        self.theta_s = theta_s
        self.zmin = zmin
        self.zmax = zmax
        self.theta_s_rad = theta_s * ARCMIN_TO_RAD
        self.P2D = P_2D(zmin,zmax,Nz,cosmo,False)

        self.integ_lmin = integ_lmin
        self.integ_lmax = integ_lmax

    def evaluate(self,theta):
        """
        theta should be a scalar value in arcmin
        """
        #print theta

        if self.window=='gaus':
            N = 100
        else:
            N = 100*(1+theta)
        factor,integrand = self.integrand(theta)

        I = integrate_log(integrand,self.integ_lmin,self.integ_lmax,
                          imethod = integrate_onecall,N=N)

        return factor * I
        
        

    def __call__(self,theta):
        """
        integrate the projected power spectrum within a circular
        window function
        """
        result = theta.copy()

        for i in range(theta.size):
            result.flat[i] = self.evaluate(theta.flat[i])

        return result

    def integrand(self,theta):
        if self.window == 'circ':
            return self.integrand_circ(theta)
        elif self.window == 'gaus':
            return self.integrand_gaus(theta)
        else:
            raise ValueError, "window = %s not recognized" % self.window

    def integrand_circ(self,theta):
        #circular window: F = 2*J_1(theta*ell)/(theta*ell)
        theta *= ARCMIN_TO_RAD

        A = 1E10
        factor = 2./ numpy.pi / self.theta_s_rad**2 / A

        integrand = lambda ell : \
                    A * self.P2D(ell) \
                    * special.j1(self.theta_s_rad*ell)**2 \
                    * special.j0(theta * ell) / ell

        return factor,integrand

    def integrand_gaus(self,theta):
        #gaussian window: F = exp( -0.5*(theta*ell)**2 )
        theta *= ARCMIN_TO_RAD

        A = 1E10
        factor = 0.5 / numpy.pi / A

        integrand = lambda ell: \
                        A * self.P2D(ell) \
                        * numpy.exp( -(theta*ell)**2 ) \
                        * special.j0(theta * ell) * ell

        return factor,integrand

    def plot_integrand(self):
        for theta in (1,10,100):
            A,I = self.integrand(theta)
            ell = 10**numpy.linspace(numpy.log10(self.integ_lmin),
                                     numpy.log10(self.integ_lmax),5000)
            pylab.semilogx(ell,I(ell),label=r'$\theta=%i^\prime$' % theta)

        pylab.xlabel(r'$\ell$')
        pylab.ylabel(r'$[d\omega/d\ell](\ell)$')
        pylab.xlim(self.integ_lmin,self.integ_lmax)
        pylab.legend(loc=0)
        pylab.title(r'$\omega(\theta)\ \rm{integrand\ for}\ \theta_s = %i^\prime\ (%.1f<z<%.1f)$' % (self.theta_s,self.zmin,self.zmax))

    def plot(self,N=100):
        theta = 10**numpy.linspace(-0.5,2.15,N)
        P = self(theta)
        pylab.loglog(theta,abs(P),label='%.2f<z<%.2f' % (self.zmin,self.zmax))
        pylab.xlim(theta[0],theta[-1])
        pylab.ylim(1E-4,2)
        pylab.xlabel(r'$\theta\ \rm{arcmin}$')
        pylab.ylabel(r'$\omega(\theta)$')
