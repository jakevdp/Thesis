import numpy
import pylab
from matplotlib import font_manager

from Lens3D import *
from cosmo_tools import Cosmology

from scipy.sparse.linalg import LinearOperator, cg, cgs, gmres
from scipy.sparse.linalg.eigen import arpack
from scipy.special import j0,j1
from scipy import integrate, interpolate, special

import sys, os
dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append( os.path.join(dirname,'PowerSpectrum') )
from halofit import PowerSpectrum

from S_DD import P_2D, S_DD, integrate_log
from integrate_log_2D import integrate_log_2D

from time import time

Mpc_cm = 3.08568E24   #centimeters in a megaparsec
Msun_g = 1.98892E33   #Mass of the sun in grams

RAD_TO_ARCMIN = 180.*60./numpy.pi

ARCMIN_TO_RAD = numpy.pi / 180. /60. 

RAD_TO_DEG = 180./numpy.pi

DEG_TO_RAD = numpy.pi / 180.


def printtime(sec):
    min = int(sec * 1./60)
    hr =  int(min * 1./60)

    sec -= 60*min
    min -= 60*hr

    if hr>0:
        return "%i hr %i min %i sec" % (hr,min,sec)
    elif min>0:
        return "%i min %i sec" % (min,sec)
    else:
        return "%.2g sec" % sec
    


class SampledFunction:
    def __init__(self,func,xmin,xmax,Nx,check_range=False):
        self.__func = func
        self.__xmin = xmin
        self.__xmax = xmax
        self.check_range = check_range
        
        self.x = numpy.linspace(xmin,xmax,Nx)
        self.y = func(self.x)
        
        self.__tck = interpolate.splrep(self.x,self.y)

    def __call__(self,x):
        x = numpy.asarray(x)
        if self.check_range and \
               numpy.any( x<self.__xmin or x>self.__xmax ):
            raise ValueError,"x out of range"

        #interpolate.splev expects a 1D array:
        # we must flatten x and compute y from this
        # we must also do this carefully to avoid copying data
        # (if possible) and to avoid reshaping the final answer
        # in the wrong way.
        if x.flags.f_contiguous:
            x_1D = x.ravel('F')
            order = 'F'
        elif x.flags.c_contiguous:
            x_1D = x.ravel('C')
            order = 'C'
        else:
            x_1D = numpy.asarray(x,order='C').ravel('C')
            order = 'C'

        y_1D = interpolate.splev(x_1D,self.__tck)
        return y_1D.reshape( x.shape, order=order)

    def plot(self,*args,**kwargs):
        pylab.plot(self.x,self.y)
    def loglog(self,*args,**kwargs):
        pylab.loglog(self.x,self.y)

############################################################
# Kaiser-Squires kernel

@ZeroProtectFunction
def KS_kernel_real(theta):
    """
    KS_kernel_real:
    theta is a complex 2-D angle theta_x + i*theta_y
       (theta measured in radians)
    """
    theta = numpy.asarray(theta)
    return - 1.0 / ( numpy.pi * theta.conj()**2 )

@ZeroProtectFunction
def KS_kernel_fourier(ell):
    """
    ell is a complex 2-D wave number ell_x + i*ell_y
       (ell measured in inverse radians)
    """
    return ell / ell.conj()

#----

def construct_P_gk(theta1_min,theta1_max,N1,
                   theta2_min,theta2_max,N2,
                   zrange):
    """
    construct the P_gk (gamma-kappa) matrix such that
    gamma = P_gk * kappa
    theta is in units of arcminutes
    """
    print "constructing P_gk"
    t0 = time()
    Nz = len(zrange)
    dt1 = (theta1_max-theta1_min) * 1./(N1-1) * ARCMIN_TO_RAD
    dt2 = (theta2_max-theta2_min) * 1./(N2-1) * ARCMIN_TO_RAD

    ret = Lens3D_lp_conv(Nz,N1,N2,
                         dt1,dt2,
                         func = KS_kernel_real,
                         func_ft = KS_kernel_fourier)
    print ' - finished in',printtime(time()-t0)
    return ret
#----

def construct_P_kd(N1,N2,z_kappa,z_Delta,
                   cosmo=None,**kwargs):
    """
    construct the P_kd (kappa-Delta) matrix such that
    kappa = P_kd * Delta
    
    equivalent to equation 31 & 32 in Simon 2009,
    using Delta = delta/a as in Hu and Keeton 2003
    """
    print "constructing P_kd"
    t0 = time()
    
    if cosmo==None:
        cosmo = Cosmology(**kwargs)
    Nj = len(z_kappa)
    Nk = len(z_Delta)

    if max(z_Delta) > max(z_kappa):
        print "-------"
        print "WARNING: construct_P_kd: singular matrix [ min(z_kappa) < min(z_Delta) ]"
        print "-------"
    
    P = numpy.zeros([Nj,Nk])

    #array to hold the comoving distance to each z in z_Delta
    Dk = numpy.zeros(Nk+1)
    for k in range(Nk):
        Dk[k] = cosmo.Dc(z_Delta[k])

    Dj = numpy.zeros(Nj+1)
    for j in range(Nj):
        Dj[j] = cosmo.Dc(z_kappa[j])

    #for ease of calculation below,
    # make z_Delta[-1] = 0
    z_Delta = numpy.concatenate([z_Delta,[0]])

    for j in range(Nj):
        for k in range(Nk):
            if Dj[j] < Dk[k]:
                P[j,k] = 0
            else:
                #P[j,k] = (Dj-Dk[k])*Dk[k]/Dj \
                #         * (z_Delta[k]-z_Delta[k-1]) / cosmo.H(z_kappa[j])
                P[j,k] = (Dk[k]-Dk[k-1]) \
                    * (Dj[j]-Dk[k])*Dk[k]/Dj[j]*(1.+z_Delta[k])

    #P *= ( 1.5 * cosmo.c*cosmo.Om*(cosmo.H0)**2 )
    P *= ( 1.5 * cosmo.Om*(cosmo.H0/cosmo.c)**2 )

    #pylab.imshow(P,cmap=pylab.cm.gray,interpolation='nearest')
    #pylab.colorbar()
    #pylab.show()
    #exit()

    ret = Lens3D_los_mat(Nk,N1,N2,Nj,data=P)
    print ' - finished in',printtime(time()-t0)
    return ret

def sin_x_over_x(x,EPS=1E-8):
    x = numpy.asarray(x)
    if x.shape==():
        if abs(x)<EPS:
            return 1.
        else:
            return numpy.sin(x)/x
    else:
        i = numpy.where(abs(x)<EPS)
        x[i] = 1.
        ret = numpy.sin(x)/x
        ret[i] = 1
        return ret
    
def j1_x_over_x(x,EPS=1E-8):
    x = numpy.asarray(x)
    if x.shape==():
        if abs(x)<EPS:
            return 1./3.
        else:
            return j1(x)/x
    else:
        x = numpy.asarray(x)
        i = numpy.where(abs(x)<EPS)
        x[i] = 1.
        ret = j1(x)/x
        ret[i] = 1./3.
        return ret

class RWF_integrand:
    def __init__(self,Di,Dj,dDi,dDj,Pi,Pj,thetaS):
        #cosmological parameters
        self.DTi = Di*thetaS
        self.DTj = Dj*thetaS
        self.dDi_2 = 0.5*dDi
        self.dDj_2 = 0.5*dDj
        self.Dij = Di-Dj
        self.Pi = Pi
        self.Pj = Pj

    def __call__(self,kpar,kperp):
        kk = numpy.sqrt(kpar**2 + kperp**2)
        return numpy.cos(kpar*self.Dij) \
               * sin_x_over_x(kpar*self.dDi_2) \
               * sin_x_over_x(kpar*self.dDj_2)  \
               * kperp * j1_x_over_x(kperp*self.DTi) \
               * j1_x_over_x(kperp*self.DTj)\
               * numpy.sqrt( self.Pi.D2_NL(kk)*self.Pj.D2_NL(kk) )

    def integrate(self,
                  kpar_min = 1E-5, kpar_max = 1E1, Nkpar = 200,
                  kperp_min = 1E-2, kperp_max = 1E4, Nkperp = 200):
        """
        Integral over kpar is from -inf to inf
        Because the imaginary part is an odd function of kpar, it vanishes.
        [this is why we changed e^(kpar*Dij) to cos(kpar*Dij) above]
        The real part is an even function, so we'll integrate 0 to inf and
        double the result.
        
        Integral over kperp is radial (d^2 kperp -> 2 pi kperp dkperp)
          so it is from 0 to inf

        We integrate in log space to better handle the oscillations of
        j1(x)/x and sin(x)/x
        """
        return 2.* integrate_log_2D(self,
                                    kpar_min,kpar_max,Nkpar,
                                    kperp_min,kperp_max,Nkperp)

    def plot(self,kpar0=1.0,kperp0=1.0):
        kpar = 10**numpy.linspace(-5,0,1000)
        kperp = 10**numpy.linspace(-2,4,1000)

        fpar = self(kpar,kperp0)
        fperp = self(kpar0,kperp)

        plotfunc = pylab.plot

        pylab.subplot(211)
        plotfunc(kpar,fpar)
          
        pylab.subplot(212)
        plotfunc(kperp,fperp)
    
def construct_radial_S_dd(Nx,Ny,zrange,
                          pixel_width,
                          cosmo=None,**kwargs):
    """
    Return radial covariance based on eqns 37-39 in HK02 and
    eqns 43-46 in STH09
    """
    print "constructing radial S_dd"
    t0 = time()
    if cosmo == None:
        cosmo = Cosmology(**kwargs)

    thetaS = pixel_width/numpy.sqrt(numpy.pi) * ARCMIN_TO_RAD

    Nz = len(zrange)
    S = numpy.zeros((Nz,Nz),dtype=complex)

    #create a power spectrum object for each redshift bin
    PSpecs = [PowerSpectrum(z) for z in zrange]

    #compute comoving distance & bin width
    w = numpy.asarray([cosmo.Dc(z) for z in zrange])
    dw = w.copy()
    dw[1:] -= w[:-1]
    w -= 0.5*dw

    for i in range(Nz):
        for j in range(i,Nz):
            integrand = RWF_integrand(w[i],w[j],
                                      dw[i],dw[j],
                                      PSpecs[i],PSpecs[j],
                                      thetaS)
            #integrand.plot()
            #pylab.show()
            #exit()
            S[i,j] = integrand.integrate() / numpy.pi / numpy.pi
            S[j,i] = S[i,j]

    #pylab.figure()
    #pylab.imshow(S.real,
    #             interpolation = 'nearest')
    #cb = pylab.colorbar()
    #cb.set_label('S')
    #pylab.show()

    ret = Lens3D_los_mat(Nz,Nx,Ny,data=S)
    print ' - finished in',printtime(time()-t0)
    return ret

class square_pixel_ft:
    def __init__(self,dx,dy):
        """
        fourier transform of the window function for a rectangular pixel
        of dimensions dx by dy
        """
        self.dx = dx
        self.dy = dy
  
    def __call__(self,k):
        """
        k is a complex wave number, k_x + 1j*k_y
        """
        k = numpy.asarray(k)
        k_x = k.real
        k_y = k.imag

        i_x = numpy.where(k_x == 0)
        ret_x = numpy.sin(k_x*self.dx) / (k_x*self.dx)
        ret_x[i_x] = 1.0

        i_y = numpy.where(k_y == 0)
        ret_y = numpy.sin(k_y*self.dy) / (k_y*self.dy)
        ret_y[i_y] = 1.0

        return ret_x * ret_y
    
def construct_angular_S_dd(theta1_min,theta1_max,N1,
                           theta2_min,theta2_max,N2,
                           zrange,pixel_radius = None,
                           cosmo=None, 
                           use_discrete = False,
                           **kwargs):
    """
    construct a matrix of the signal covariance
    this is an implementation of eqns 39-41 of Simon et al 2009
    
    theta is assumed to be given in arcmin
    """
    print "constructing transverse S_dd"
    t0 = time()
    
    zrange_ext = [min(0.001,zrange[0]/2)]+list(zrange)
    
    if cosmo is None:
        cosmo = Cosmology(**kwargs)
        
    theta1 = numpy.linspace(theta1_min,theta1_max,N1) * ARCMIN_TO_RAD
    theta2 = numpy.linspace(theta2_min,theta2_max,N2) * ARCMIN_TO_RAD
    
    dt1 = theta1[1]-theta1[0]
    dt2 = theta2[1]-theta2[0]

    if use_discrete:
        if pixel_radius is None:
            pixel_radius = 0.25 * ( dt1 + dt2 ) * RAD_TO_ARCMIN
            
        print "pixel radius: %.2f arcmin" % pixel_radius    

        theta_min = 0
        theta_max = numpy.sqrt(   4*(theta1_max-theta1_min)**2 + \
                                      4*(theta2_max-theta2_min)**2     )
        Ntheta = int( numpy.sqrt(N1**2 + N2**2) )
        
        S_DD_funcs = [S_DD(zrange_ext[i],zrange_ext[i+1],2,cosmo,pixel_radius)
                      for i in range(len(zrange))]
        
        S_DD_Sampled = [SampledFunction(S_DD_func,theta_min,theta_max,Ntheta)
                        for S_DD_func in S_DD_funcs]
        
        S_DD_abs = [lambda x: F(abs(x))/dt1/dt2 for F in S_DD_Sampled]

        
        ret = Lens3D_lp_conv(len(zrange),N1,N2,dt1,dt2,
                              func = S_DD_abs)
    else:
        P_2D_funcs = [P_2D(zrange_ext[i],zrange_ext[i+1],2,cosmo)
                      for i in range(len(zrange))]

        wind = square_pixel_ft(dt1,dt2)

        P_2D_window = [ZeroProtectFunction( lambda ell: wind(ell)**2 \
                                            * P(abs(ell)) )
                       for P in P_2D_funcs]
        
        ret = Lens3D_lp_conv(len(zrange),N1,N2,dt1,dt2,
                              func_ft = P_2D_window)
    print " - completed in", printtime(time()-t0)
    return ret

def estimate_condition_number(P_kd,
                              P_gk,
                              S_dd,
                              N,
                              alpha,
                              compute_exact = False):
    P_kd = as_Lens3D_matrix(P_kd)
    P_gk = as_Lens3D_matrix(P_gk)
    S_dd = as_Lens3D_matrix(S_dd)
    N = as_Lens3D_matrix(N)
    
    P_gk_cross = P_gk.conj_transpose()
    P_kd_T = P_kd.transpose()
    
    def matvec(v):
        v0 = P_gk_cross.view_as_Lens3D_vec(v)

        v2 = P_gk_cross.matvec( v0 )
        v2 = P_kd_T.matvec( v2 )
        v2 = S_dd.matvec( v2 )
        v2 = P_kd.matvec( v2 )
        v2 = P_gk.matvec( v2 )
        
        v1 = N.matvec(v0)
        v1 *= alpha

        ret = numpy.zeros(v.shape,dtype=complex)
        ret += v1.vec
        ret += v2.vec

        return P_gk_cross.view_as_same_type( ret , v )

    M = LinearOperator(P_gk.shape,
                       matvec=matvec,
                       dtype=complex)

    #compute the exact condition number
    if compute_exact:
        v = numpy.random.random(M.shape[1])
        t0 = time()
        v2 = M.matvec(v)
        t = time()-t0
        print " - constructing matrix representation (est. %s)" \
            % printtime(t*M.shape[0])
        t0 = time()
        M_rep = get_mat_rep(M)
        print "    time to get mat rep: %.2g sec" % (time()-t0)
        print " - computing SVD"
        t0 = time()
        sig = numpy.linalg.svd(M_rep,compute_uv=False)
        print "    time for SVD: %.2g sec" % (time()-t0)
        print 'true condition number:      %.2e / %.2e = %.2e' \
            % (sig[0],sig[-1],
               sig[0]/sig[-1])

    #estimate condition number, assuming the noiseless matrix
    # is rank-deficient.  This will be true if there are more
    # source lens-planes than mass lens-planes
    eval_max,evec_max = arpack.eigen(M,1)
    print 'estimated condition number: %.2e / %.2e = %.2e' \
        % ( abs(eval_max[0]) , numpy.min(N.data),
            abs(eval_max[0]) / numpy.min(N.data) )
 
def calculate_delta_simple(gamma,     #3-dim array (Nz,Nx,Ny) of complex shear
                           P_kd,      #line-of-sight kappa-delta transform
                           P_gk,      #gamma-kappa transform
                           S_dd,      #expected signal covariance of delta
                           # (can be arbitrary form)
                           N,         #shear noise
                           alpha,     #weiner filter strength
                           M_factor = 1 #optional scaling of M before cg-method
                           ):    
    """
    Implement equation A3 from Simon & Taylor 2009
    Note: their notation Q -> our notation P_kd

    Also, here we factor N_d^-1 out of M: M should be Hermitian for
    the cg method. The Simon 09 expression is Hermitian only if N_d is
    proportional to the identity, which will not be the case for
    deweighted border pixels.
    """
    P_kd = as_Lens3D_matrix(P_kd)
    P_gk = as_Lens3D_matrix(P_gk)
    S_dd = as_Lens3D_matrix(S_dd)
    N = as_Lens3D_matrix(N)
    
    P_gk_cross = P_gk.conj_transpose()
    P_kd_T = P_kd.transpose()

    #define an operator which performs matrix-vector
    # multiplication representing M
    def matvec(v):
        v0 = P_gk_cross.view_as_Lens3D_vec(v)
        
        v1 = N.matvec(v0)
        v1 *= alpha

        v2 = P_gk_cross.matvec( v0 )
        v2 = P_kd_T.matvec( v2 )
        v2 = S_dd.matvec( v2 )
        v2 = P_kd.matvec( v2 )
        v2 = P_gk.matvec( v2 )

        ret = numpy.zeros(v.shape,dtype=complex)
        ret += v1.vec
        ret += v2.vec

        return P_gk_cross.view_as_same_type( ret * M_factor , v )

    M = LinearOperator(P_gk.shape,
                       matvec=matvec,
                       dtype=complex)

    #---define callback function---
    def callback(self):
        callback.N += 1
        if callback.N%100 == 0: print callback.N,'iterations'
    callback.N = 0
    #------------------------------

    step1_vec = gamma.vec
    
    t0 = time()
    print "calculating cg:"
    ret,errcode = cg( M, step1_vec,
                      x0 = numpy.zeros(M.shape[1],dtype=step1_vec.dtype),
                      callback=callback )
    if errcode != 0:
        raise ValueError, "calculate_delta_simple: cg iterations did not converge: err = %s" % (str(errcode))
    tf = time()

    print "   cg:   total time           = %.2g sec" % (tf-t0)
    print "         number of iterations = %i" % callback.N
    print "         time per iteration   = %.2g sec" % ( (tf-t0)/callback.N )

    ret *= M_factor
    
    ret = P_gk_cross * ret
    ret = P_kd_T * ret
    delta = S_dd * ret

    return P_kd.view_as_Lens3D_vec(delta)
    
    

def calculate_delta(gamma,     #3-dim array (Nz,Nx,Ny) of complex shear
                    P_kd,      #line-of-sight kappa-delta transform
                    P_gk,      #gamma-kappa transform
                    S_dd,      #expected signal covariance of delta
                               # (can be arbitrary form)
                    N,         #shear noise
                    alpha,     #weiner filter strength
                    M_factor = 1 #optional scaling of M before cg-method
                    ):    
    """
    Implement equation A3 from Simon & Taylor 2009
    Note: their notation Q -> our notation P_kd

    Also, here we factor N_d^-1 out of M: M should be Hermitian for
    the cg method. The Simon 09 expression is Hermitian only if N_d is
    proportional to the identity, which will not be the case for
    deweighted border pixels.
    """
    P_kd = as_Lens3D_matrix(P_kd)
    P_gk = as_Lens3D_matrix(P_gk)
    S_dd = as_Lens3D_matrix(S_dd)
    N = as_Lens3D_matrix(N)
    
    P_gk_cross = P_gk.conj_transpose()
    P_kd_T = P_kd.transpose()

    print "calculating delta:"

    print "  shape of P_gk:",P_gk.shape
    print "  shape of P_kd:",P_kd.shape

    print "constructing linear operator M"
    #define an operator which performs matrix-vector
    # multiplication representing M
    def matvec(v):
        v0 = P_gk_cross.view_as_Lens3D_vec(v)
        
        v1 = N.matvec(v0)
        v1 *= alpha

        v2 = P_gk_cross.matvec( v0 )
        v2 = P_kd_T.matvec( v2 )
        v2 = S_dd.matvec( v2 )
        v2 = P_kd.matvec( v2 )
        v2 = P_gk.matvec( v2 )

        ret = numpy.zeros(v.shape,dtype=complex)
        ret += v1.vec
        ret += v2.vec

        return P_gk_cross.view_as_same_type( ret * M_factor , v )

    M = LinearOperator(P_gk.shape,
                       matvec=matvec,
                       dtype=complex)

    v = numpy.random.random(M.shape[1])
    t0 = time()
    v2 = M.matvec(v)
    t = time()-t0
    print "  M multiplication: %.3g sec" % t

    #print M.matvec(numpy.ones(M.shape[1]))[:10]
    #exit()

    print "constructing preconditioner for M"

    #define an operator which can quickly approximate the inverse of
    # M using fourier-space inversions.  This inverse will be exact for
    # a noiseless reconstruction on an infinite field
    P_gk_I = P_gk.inverse(False)
    #P_kd_I = P_kd.inverse()
    S_dd_I = S_dd.inverse(False)
    #P_kd_I_T = P_kd_I.transpose()
    P_gk_I_cross = P_gk_I.conj_transpose()

    def matvec_pc(v):
        v0 = P_gk_I.view_as_Lens3D_vec(v)
        v0 = P_gk_I.matvec(v0)
        #v0 = P_kd_I.matvec(v0)
        v0 = S_dd_I.matvec(v0)
        #v0 = P_kd_I_T.matvec(v0)
        v0 = P_gk_I_cross.matvec(v0)
        return P_gk_I.view_as_same_type(v0,v)
    
    M_pc = LinearOperator( (M.shape[1],M.shape[0]), 
                           matvec = matvec_pc,
                           dtype = M.dtype )
    
    v = numpy.random.random(M_pc.shape[1])
    t0 = time()
    v3 = M_pc.matvec(v)
    t_pc = time()-t0
    print "  preconditioner multiplication: %.3g sec" % t_pc

    step1_vec = gamma.vec

    use_cg = True

    #---define callback function---
    def callback(self):
        callback.N += 1
        if callback.N%100 == 0: print callback.N,'iterations'
    callback.N = 0
    #------------------------------
        
    t0 = time()
    print "calculating cg:"
    ret,errcode = cg( M, step1_vec,
                      x0 = numpy.zeros(M.shape[1],dtype=step1_vec.dtype),
                      callback=callback,
                      M = M_pc)
    if errcode != 0:
        raise ValueError, "calculate_delta: cg iterations did not converge: err = %s" % (str(errcode))
    tf = time()

    print "   cg:   total time           = %.2g sec" % (tf-t0)
    print "         number of iterations = %i" % callback.N
    print "         time per iteration   = %.2g sec" % ( (tf-t0)/callback.N )
    
    ret *= M_factor
    
    ret = P_gk_cross * ret
    ret = P_kd_T * ret
    delta = S_dd * ret

    return P_kd.view_as_Lens3D_vec(delta)
#---
    

def test_lin_op(mat):
    """
    make sure the linear operator is linear
    """
    try:
        mat_rep = mat.full_matrix
    except:
        mat_rep = get_mat_rep(mat)
    
    print "|========="
    print "| testing linearity of operator:"
    print "|   mat rep type:       ",mat_rep.dtype
    if numpy.all(mat_rep==0):
        print "|     (mat rep all zero)"
    v = numpy.random.random(mat.shape[1]) + 1j*numpy.random.random(mat.shape[1])
    x1 = numpy.dot(mat_rep,v)
    x2 = mat.matvec(v)

    diff = abs( x1-x2 )

    print "|   total error:         %.3g \pm %.3g" % \
          (numpy.mean(diff),numpy.std(diff))
    #print "|         min = %.3g ; max = %.3g" % ( min(diff),max(diff) )

    abs_diff = diff/abs(x1)
    #take care of 0/0 nans
    abs_diff[numpy.where(diff==0)] = 0

    print "|   fractional error:    %.3g \pm %.3g" % \
          (numpy.mean(abs_diff),numpy.std(abs_diff))
    #print "|         min = %.3g ; max = %.3g" % ( min(abs_diff),max(abs_diff) )
    
    print "|========="

def gaussian_smooth(L3Dvec, pixel_width, smoothing_radius):
    """
    return a smoothed version of L3Dvec.
    """
    window = lambda ell: \
             numpy.dot( - 0.5 * ell * smoothing_radius * 1./ pixel_width )

    
def delta_to_Sigma(delta,zrange,cosmo=None):
    Sigma = delta.copy()
    assert len(zrange) == delta.Nz
    if cosmo is None: cosmo = Cosmology()
    zrange = list(zrange) + [0.0]
    for i in range(Sigma.Nz):
        lp = Sigma.lens_plane(i)
        z = zrange[i]
        #lp += 1
        lp  *= cosmo.H0**2 * cosmo.Om * cosmo.rho_crit(z)*(1.+z)**3 \
               /cosmo.H(z)**2

        #now we're in g/cm^3
        lp *= Mpc_cm**3 / Msun_g

        #multiply by size of bin to get surface density
        lp *= ( cosmo.Dc(zrange[i]) - cosmo.Dc(zrange[i-1]) )
    return Sigma

def Sigma_to_delta(Sigma,zrange,cosmo=None):
    delta = Sigma.copy()
    assert len(zrange) == Sigma.Nz
    if cosmo is None: cosmo = Cosmology(Or=0.0)
    zrange = list(zrange) + [0.0]
    for i in range(delta.Nz):
        lp = delta.lens_plane(i)
        z = zrange[i]
        
        #divide by bin size to turn surface density into 3D density
        lp /= ( cosmo.Dc(zrange[i]) - cosmo.Dc(zrange[i-1]) )

        #convert to g/cm^3
        lp *= Msun_g / Mpc_cm**3

        lp /= cosmo.H0**2 * cosmo.Om * cosmo.rho_crit(z)*(1.+z)**3 \
               /cosmo.H(z)**2

        #i = numpy.where(lp.real>0)
        #lp[i] -= 1
        
    return delta
