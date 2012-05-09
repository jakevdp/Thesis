import warnings

import numpy
from scipy import special

"""
Nonlinear Power spectrum from R.E. Smith et al, MNRAS 341:1311 (2003)
Adapted from the fortran code made available by the author

python adaptation by Jake VanderPlas, 2010
  email: vanderplas@astro.washington.edu
  web:   http://www.astro.washington.edu/users/vanderplas

please let me know if you find this code useful.
"""

class HalofitNonConvergence(Exception):
    pass


class PowerSpectrum(object):
    """
    This class implements the power spectrum from Smith 2003.
    This is a direct adaptation from fortran of the halofit code
    that was made available with the paper.
    """
    def __init__(self, z,
                 om_m0 = 0.3,
                 om_v0 = 0.7,
                 sig8  = 0.85,
                 gams  = 0.21):
        self.om_m0 = om_m0
        self.om_v0 = om_v0
        self.sig8 = sig8
        self.gams  = gams
        self.z_ = z
        self.calc_spectral_params_()

    def get_z(self):
        return self.z_

    def set_z(self,z):
        self.z_ = z
        self.calc_spectral_params_()

    z = property(get_z,set_z)

    def calc_spectral_params_(self):
        """
        Set the redshift of the power spectrum, and compute the
        associated spectral parameters
        """
        self.om_m = self.omega_m(self.z_)
        self.om_v = self.omega_v(self.z_)

        gg  = self.growth_factor(self.z_)
        gg0 = self.growth_factor(0)
        self.amp = gg/gg0/(1.+self.z_)

        xlogr1_init = -6.0
        xlogr2_init = 5.0
        
        xlogr1 = xlogr1_init
        xlogr2 = xlogr2_init

        #iterate to determine
        #  rknl  : wavenumber where nonlinear effects become important
        #  rneff : effective spectral index
        #  rncur : second derivative of the power spectrum at rknl
        while True:
            rmid = 10**( 0.5*(xlogr2+xlogr1) )
            
            sig,d1,d2 = self.wint_(rmid)
            
            diff = sig-1.0
            
            if xlogr1==xlogr2:
                warnings.warn(
                    "spectral params did not converge. Setting rknl to large")
                self.rknl = 1./rmid
                self.rneff = -3.
                self.rncur = 0.
                break
            elif diff > 0.001:
                xlogr1 = numpy.log10(rmid)
                continue
            elif diff < -0.001:
                xlogr2 = numpy.log10(rmid)
                continue
            else:
                self.rknl = 1./rmid
                self.rneff = -3. - d1
                self.rncur = -d2
                #print 'final:',xlogr1, xlogr2, d1, d2
                #print '  rknl [h/Mpc] = %.3g' % self.rknl,
                #print '  rneff = %.3g' % self.rneff,
                #print '  rncur = %.3g' % self.rncur
                break
        

    def wint_(self,r):
        """
        The subroutine wint, finds the effective spectral quantities
        rknl, rneff & rncur. This it does by calculating the radius of 
        the Gaussian filter at which the variance is unity = rknl.
        rneff is defined as the first derivative of the variance, calculated 
        at the nonlinear wavenumber and similarly the rncur is the second
        derivative at the nonlinear wavenumber.
        
        returns (sig,d1,d2)
        """
        nint=3000.
        t = ( numpy.arange(nint)+0.5 )/nint
        y = 1./t - 1.
        rk = y
        d2 = self.D2_L(rk)
        x2 = y * y * r * r
        w1 = numpy.exp(-x2)
        w2 = 2 * x2 * w1
        w3 = 4 * x2 * (1 - x2) * w1

        mult = d2 / y / t / t
        
        sum1 = numpy.sum(w1*mult)/nint
        sum2 = numpy.sum(w2*mult)/nint
        sum3 = numpy.sum(w3*mult)/nint
        
        sig = numpy.sqrt(sum1)
        d1  = - sum2 / sum1
        d2  = - sum2 * sum2 / sum1 / sum1 - sum3 / sum1
        
        return sig,d1,d2
    
    def omega_m(self,z):
        """
        evolution of omega matter with redshift
        """
        a = 1./(1.+z)
        Ok = 1.-self.om_m0-self.om_v0
        omega_t = 1.0 - Ok / (Ok + self.om_v0*a*a + self.om_m0/a)
        return omega_t * self.om_m0 / (self.om_m0 + self.om_v0*a*a*a)

    def omega_v(self,z):
        """
        evolution of omega vacuum with redshift
        """
        a = 1./(1.+z)
        Ok = 1.-self.om_m0-self.om_v0
        omega_t = 1.0 - Ok / (Ok + self.om_v0*a*a + self.om_m0/a)
        return omega_t*self.om_v0 / (self.om_v0 + self.om_m0/a/a/a)

    def growth_factor(self,z):
        """
        g(Omega) from Carroll, Press & Turner (1992)
        """
        om_m = self.omega_m(z)
        om_v = self.omega_v(z)
        return 2.5*om_m/(om_m**(4./7.)-om_v+(1.+om_m/2.)*(1.+om_v/70.))

    def p_cdm(self,rk):
        """
        the un-normalized linear CDM power spectrum
        """
        rk = numpy.asarray(rk)
        p_index = 1.
        rkeff=0.172+0.011*numpy.log(self.gams/0.36)*numpy.log(self.gams/0.36)
        q=1.e-20 + rk/self.gams
        q8=1.e-20 + rkeff/self.gams
        tk=1./(1+(6.4*q+(3.0*q)**1.5+(1.7*q)**2)**1.13)**(1/1.13)
        tk8=1./(1+(6.4*q8+(3.0*q8)**1.5+(1.7*q8)**2)**1.13)**(1/1.13)
        return self.sig8*self.sig8*((q/q8)**(3.+p_index))*tk*tk/tk8/tk8

    def D2_L(self,rk):
        """
        return the linear power spectrum
        """
        rk = numpy.asarray(rk)
        return self.amp * self.amp * self.p_cdm(rk)

    def D2_NL(self,rk,return_components = False):
        """
        halo model nonlinear fitting formula as described in 
        Appendix C of Smith et al. (2002)
        """        
        rk = numpy.asarray(rk)
        rn    = self.rneff
        rncur = self.rncur
        rknl  = self.rknl
        plin  = self.D2_L(rk)
        om_m  = self.om_m
        om_v  = self.om_v
        
        gam=0.86485+0.2989*rn+0.1631*rncur
        a=10**(1.4861+1.83693*rn+1.67618*rn*rn+0.7940*rn*rn*rn+\
               0.1670756*rn*rn*rn*rn-0.620695*rncur)
        b=10**(0.9463+0.9466*rn+0.3084*rn*rn-0.940*rncur)
        c=10**(-0.2807+0.6669*rn+0.3214*rn*rn-0.0793*rncur)
        xmu=10**(-3.54419+0.19086*rn)
        xnu=10**(0.95897+1.2857*rn)
        alpha=1.38848+0.3701*rn-0.1452*rn*rn
        beta=0.8291+0.9854*rn+0.3400*rn**2
        
        if abs(1-om_m) > 0.01: #omega evolution
            f1a=om_m**(-0.0732)
            f2a=om_m**(-0.1423)
            f3a=om_m**(0.0725)
            f1b=om_m**(-0.0307)
            f2b=om_m**(-0.0585)
            f3b=om_m**(0.0743)       
            frac=om_v/(1.-om_m) 
            f1=frac*f1b + (1-frac)*f1a
            f2=frac*f2b + (1-frac)*f2a
            f3=frac*f3b + (1-frac)*f3a
        else:         
            f1=1.0
            f2=1.0
            f3=1.0

        y=(rk/rknl)
        
        ph = a*y**(f1*3)/(1+b*y**(f2)+(f3*c*y)**(3-gam))
        ph /= (1+xmu*y**(-1)+xnu*y**(-2))
        pq = plin*(1+plin)**beta/(1+plin*alpha)*numpy.exp(-y/4.0-y**2/8.0)
        
        pnl=pq+ph

        if return_components:
            return pnl,pq,ph,plin
        else:
            return pnl

    def D2_NL_PD96(self,rklin):
        """
        implement the Peacock & Dodds 1996 power spectrum.  Because
        of the way this is calculated, the user must supply the linear
        wave number, and a tuple (rk_pd,pnl_pd) is returned.
        rk_pd is the nonlinear wave number associated with the input
        linear wave number, and pnl_pd is the nonlinear power spectrum
        associated with rk_pd.
        """
        plin  = self.D2_L(rklin)
        
        rn_pd = self.rn_cdm(rklin)

        pnl_pd = self.f_pd(plin,rn_pd)
        
        rk_pd = rklin * (1+pnl_pd)**(1./3.)

        return rk_pd,pnl_pd

    def rn_cdm(self,rk):
        """
        effective spectral index used in Peacock & Dodds (1996)
        """
        y     = self.p_cdm(rk/2.)
        yplus = self.p_cdm(rk*1.01/2.)
        return -3.+numpy.log(yplus/y)*100.5

    def f_pd(self,y,rn):
        """
        Peacock & Dodds (1996) fitting formula
        """
        g = 2.5*self.om_m / \
            (self.om_m**(4./7.) - \
             self.om_v + \
             (1.+self.om_m/2.)*(1.+self.om_v/70.))
        a=0.482*(1.+rn/3.)**(-0.947)
        b=0.226*(1.+rn/3.)**(-1.778)
        alp=3.310*(1.+rn/3.)**(-0.244)
        bet=0.862*(1.+rn/3.)**(-0.287)
        vir=11.55*(1.+rn/3.)**(-0.423)
        return y * ( (1.+ b*y*bet + (a*y)**(alp*bet)) / \
                     (1.+ ((a*y)**alp*g*g*g/vir/y**0.5)**bet ) )**(1./bet)
        

if __name__ == '__main__':
    import pylab

    z = 0.2
    
    PSpec = PowerSpectrum(z=z)

    #calculate S03 power law
    N = 1000
    rk = 10**( -5.0 + 12.0*numpy.linspace(0,1,N) )

    pnl,pq,ph,plin = PSpec.D2_NL(rk,return_components = True)
    
    pylab.loglog(rk,pnl,'-k',label='nonlinear')
    pylab.loglog(rk,pq,':k',label='quasi-linear')
    pylab.loglog(rk,ph,':r',label='halo')
    pylab.loglog(rk,plin,'--r',label='linear')

    #calculate PD96 power law
    rk_lin = 10**( -2.0+4.0*numpy.linspace(0,1,N) )
    rk_pd,pnl_pd = PSpec.D2_NL_PD96(rk_lin)
    
    pylab.loglog(rk_pd,pnl_pd,'--k',label='PD96')
    
    pylab.ylim(10**-1.5,3E3)
    pylab.xlim(10**-1.5,1E2)
    
    pylab.xlabel(r'$k$')
    pylab.ylabel(r'$\Delta^2(k)$')
    pylab.title('z=%.2f Power Spectrum' % z)
    
    pylab.legend(loc=0)

    pylab.show()

    
