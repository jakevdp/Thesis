#-append the correct paths for importing Lens3D modules--------------
import sys, os
Lens3D_path = os.environ.get('LENS3D_PATH')
if Lens3D_path is None:
    raise ValueError, 'must set environment variable LENS3D_PATH'
sys.path.append(Lens3D_path)
sys.path.append(os.path.join(Lens3D_path,'3D_shear_generation'))
#--------------------------------------------------------------------

from Simon_Taylor_method import *
from Lens3D import *
from SVD_method import *
from thin_lens import *

from scipy.sparse.linalg import LinearOperator
from time import time

def compute_N_los(z_gamma,
                  z0=0.57,
                  sig=0.3,
                  Ngal=70):
    """
    compute the line-of-sight shear noise based on the
    galaxy distribution defined by z0 and Ngal (number of galaxies per pixel)
    """
    z_gamma = numpy.asarray(z_gamma)
    Nzg = len(z_gamma)
    if z0 == 0:
        N_per_bin = numpy.ones(Nzg)
        N_per_bin /= Nzg
        N_per_bin *= Ngal
    else:
        N_per_bin = z_gamma**2 * numpy.exp( -(z_gamma/z0)**1.5 )
        N_per_bin /= N_per_bin.sum()
        N_per_bin *= Ngal
    noise = sig / numpy.sqrt(N_per_bin)

    return noise**2

def compute_N_angular(N,border_size,border_noise):
    """
    return the angular component of the noise: this is the simplest
    approximation, where each pixel has the same number of galaxies,
    and the border is deweighted
    """
    noise = numpy.ones( (N,N) ) * border_noise**2
    
    noise[border_size:-border_size,
          border_size:-border_size] = 1.0

    return noise

def create_N_gg_tensor(N_los, N_angular):
    """
    compute the noise covariance of gamma, given the line-of-sight 
    and angular noise.  Result is the tensor product of the two
    """
    Nz = len(N_los)
    Nx,Ny = N_angular.shape
    return Lens3D_diag( Nz,Nx,Ny,
                        N_los[:,None,None] * N_angular[None,:,:] )

def create_N_gg_constborder(N_los, N, border_size, border_noise):
    """
    compute the noise covariance of gamma, given the line-of-sight 
    and angular noise.  Result is the tensor product of the two.

    Here we're doing it a bit differently: turns out that the tensor
    product leads to a much smaller condition number of the WF matrix.
    This noise gives the same final result, but is computed nearly 10
    times faster.
    """
    N_a = numpy.ones((N,N))
    N_gg = N_los[:,None,None] * N_a[None,:,:]

    N_gg[:,:border_size,:] = border_noise
    N_gg[:,-border_size:,:] = border_noise
    N_gg[:,:,:border_size] = border_noise
    N_gg[:,:,-border_size:] = border_noise

    return Lens3D_diag( len(N_los),N,N,N_gg )
    

def compute_SVD(P_gk,
                P_kd,
                N_angular,
                N_los):
    """
    compute the SVD of the matrix [N^(1/2) P_gk P_kd]
    using the tensor-product formalism and the fourier space approximation
    """
    Nx = P_gk.Nx
    Ny = P_gk.Ny
    Nzd = P_kd.Nz_in
    Nzg = P_kd.Nz_out

    N1I_1_2 = N_angular**-0.5

    N2I_1_2 = N_los**-0.5

    #compute SVDs
    print "computing svds"
    t0 = time()
    S1 = N1I_1_2
    U2,S2,V2 = numpy.linalg.svd(N2I_1_2[:,None]*P_kd.data_,
                                full_matrices=0)

    Nz1 = P_kd.Nz_in
    Nz2 = P_kd.Nz_out
    
    U_gk = Lens3D_ident(Nz2, Nx, Ny)
    S_gk = Lens3D_diag(Nz2, Nx, Ny,
                       numpy.outer(numpy.ones(Nz2),S1).ravel() )
    V_gk = Lens3D_lp_conv( Nz1,Nx,Ny,P_gk.dx_,P_gk.dy_,
                           P_gk.func_[0],P_gk.func_ft_[0])

    U_kd = Lens3D_los_mat(Nz1, Nx, Ny, Nz2, U2)
    S_kd = Lens3D_diag(Nz1, Nx, Ny,
                       numpy.outer(S2,numpy.ones(Nx*Ny)).ravel() )
    V_kd = Lens3D_los_mat(Nz1, Nx, Ny, Nz1, V2)

    U = Lens3D_multi(U_gk,U_kd)
    S = Lens3D_diag(Nz1, Nx, Ny,
                    numpy.outer(S2,S1).ravel()  )
    V = Lens3D_multi(V_gk,V_kd)
    tf = time()
    print " - SVDs computed in %.2g sec" % (tf-t0)

    return U,S,V

class M_WF(LinearOperator):
    """
    A class to create the matrix M from the Wiener Filter method.
    This is the matrix to be inverted in solving for the Wiener filter
    estimator.
    """
    def __init__(self,
                 P_kd,
                 P_gk,
                 S_dd,
                 N_gg,
                 alpha):
        self.P_kd = P_kd
        self.P_gk = P_gk
        self.S_dd = S_dd
        self.N_gg = N_gg
        self.alpha = alpha

        self.P_gk_cross = P_gk.conj_transpose()
        self.P_kd_T = P_kd.transpose()

        self.shape = P_gk.shape
        self.dtype = numpy.dtype(complex)

    def _matvec(self,v):
        v0 = self.P_gk_cross.view_as_Lens3D_vec(v)
        
        v1 = self.N_gg.matvec(v0)
        v1 *= self.alpha

        v2 = self.P_gk_cross.matvec( v0 )
        v2 = self.P_kd_T.matvec( v2 )
        v2 = self.S_dd.matvec( v2 )
        v2 = self.P_kd.matvec( v2 )
        v2 = self.P_gk.matvec( v2 )

        ret = numpy.zeros(v.shape,dtype=self.dtype)
        ret += v1.vec
        ret += v2.vec

        return self.P_gk_cross.view_as_same_type( ret , v )

class Minv_WF(LinearOperator):
    """
    A class to compute the inverse of M_WF above, using conjugate
    gradient iterations.  It's a subclass of LinearOperator, so
    solving the equation M_WF*x = b for x is accomplished by
    calling Minv_WF.matvec(b)
    """
    def __init__(self,
                 P_kd,
                 P_gk,
                 S_dd,
                 N_gg,
                 alpha,
                 verbose = 1):
        self.M_WF = M_WF(P_kd,P_gk,S_dd,N_gg,alpha)
        self.verbose = verbose

        self.shape = (0,0)
        self.shape = (self.M_WF.shape[1],self.M_WF.shape[0])

    def _callback(self,*args):
        self.Niter += 1
        if self.verbose and self.Niter%100 == 0:
            print self.Niter,'iterations'

    def _matvec(self,v):
        self.Niter = 0
    
        t0 = time()
        if self.verbose:
            print "calculating cg:"
        ret,errcode = cg( self.M_WF, v,
                          x0 = numpy.zeros(self.shape[1],
                                           dtype=v.dtype),
                          callback=self._callback )
        tf = time()
        
        if errcode != 0:
            raise ValueError, \
                  "perform_cg: cg iterations did not converge: err = %s" % (str(errcode))

        if self.verbose:
            print "   cg:   total time           = %.2g sec" % (tf-t0)
            print "         number of iterations = %i" % self.Niter
            print "         time per iteration   = %.2g sec" % ( (tf-t0)/self.Niter )
        
        return ret

def create_WF_Rmatrix(P_gk,
                      P_kd,
                      S_dd,
                      N_gg,
                      alpha,
                      verbose = 1):
    """
    Create the Lens3D_matrix R such that the WF estimator is
    \hat{\delta} = R\gamma
    """
    #create the conjugate-gradient object and LinearOperator
    Minv_LinOp = Minv_WF(P_kd, P_gk, S_dd, N_gg, alpha, verbose)
    Minv = Lens3D_LinearOperator(Minv_LinOp, 
                                 P_kd.Nz_out,
                                 P_gk.Nx, P_gk.Ny,
                                 rLinOp = Minv_LinOp)

    return Lens3D_multi(S_dd,
                        Minv_LinOp.M_WF.P_kd_T,
                        Minv_LinOp.M_WF.P_gk_cross,
                        Minv)

def create_SVD_Rmatrix(P_gk,
                       P_kd,
                       N_angular,
                       N_los,
                       v_cut):
    """
    Create the Lens3D_matrix R such that the SVD estimator is
    \hat{\delta} = R\gamma
    """
    U,S,V = compute_SVD(P_gk,
                        P_kd,
                        N_angular,
                        N_los)

    Nx = P_gk.Nx
    Ny = P_gk.Ny

    i_sort = numpy.argsort(S.data_)

    sig = S.data_[i_sort]
    sig *= sig
    sig_cumsum = sig.cumsum()
    sig_sum = sig.sum()

    N_tot = len(sig)
    N_cut = sig_cumsum.searchsorted(sig_sum*v_cut)
    sig_cut = numpy.sqrt( sig[N_cut] )
    
    SI = Lens3D_diag(P_kd.Nz_in,Nx,Ny,
                     1./S.data_)
    SI.data_[i_sort[:N_cut]] = 0
    
    NI_1_2 = Lens3D_diag(P_kd.Nz_out,Nx,Ny,
                         numpy.outer(N_los,N_angular)**-0.5)
    
    R = Lens3D_multi(V.H,
                     SI,
                     U.H,
                     NI_1_2)

    R.sig_cut = sig_cut
    R.N_cut = N_cut

    return R

def calculate_delta_trans(gamma,
                          z_delta,
                          z_gamma,
                          alpha = 1.0,
                          theta_min = 0,
                          theta_max = None,
                          N = 64,
                          border_size = None,
                          border_noise = 1E3,
                          Ngal = 70,
                          z0 = 0.57,
                          sig = 0.3,
                          cosmo = None,
                          **kwargs):
    """
    Calculate delta using the transverse Wiener filter
    """
    #Take care of argument defaults
    if cosmo is None:
        cosmo = Cosmology(**kwargs)
    if theta_max is None:
        theta_max = N-1
    if border_size is None:
        border_size = N/16

    #calculate all the transformation matrices
    P_gk = construct_P_gk( theta_min, theta_max, N,
                           theta_min, theta_max, N,
                           z_gamma )
    
    P_kd = construct_P_kd(N,N,z_gamma,z_delta)
    
    S_dd = construct_angular_S_dd( theta_min, theta_max, N,
                                   theta_min, theta_max, N,
                                   z_delta, cosmo=cosmo )

    N_los = compute_N_los(z_gamma,z0,sig,Ngal)
    N_angular = compute_N_angular(N,border_size,border_noise)
    #N_gg = create_N_gg_tensor(N_los, N_angular)
    N_gg = create_N_gg_constborder(N_los, N, border_size, border_noise)

    R = create_WF_Rmatrix(P_gk,
                          P_kd,
                          S_dd,
                          N_gg,
                          alpha,
                          verbose = 1)

    return R.matvec(gamma)


def calculate_delta_rad(gamma,
                        z_delta,
                        z_gamma,
                        alpha = 1.0,
                        theta_min = 0,
                        theta_max = None,
                        N = 64,
                        border_size = None,
                        border_noise = 1E3,
                        Ngal = 70,
                        z0 = 0.57,
                        sig = 0.3,
                        cosmo = None,
                        **kwargs):
    """
    Calculate delta using the radial Wiener filter
    """
    #Take care of argument defaults
    if cosmo is None:
        cosmo = Cosmology(**kwargs)
    if theta_max is None:
        theta_max = N-1
    if border_size is None:
        border_size = N/16

    #calculate all the transformation matrices
    P_gk = construct_P_gk( theta_min, theta_max, N,
                           theta_min, theta_max, N,
                           z_gamma )
    
    P_kd = construct_P_kd(N,N,z_gamma,z_delta)
    
    S_dd = construct_radial_S_dd(N, N, z_delta,
                                 (theta_max-theta_min)*1./(N-1),
                                 cosmo=cosmo )

    N_los = compute_N_los(z_gamma,z0,sig,Ngal)
    N_angular = compute_N_angular(N,border_size,border_noise)
    #N_gg = create_N_gg_tensor(N_los, N_angular)
    N_gg = create_N_gg_constborder(N_los, N, border_size, border_noise)

    R = create_WF_Rmatrix(P_gk,
                          P_kd,
                          S_dd,
                          N_gg,
                          alpha,
                          verbose = 1)

    return R.matvec(gamma)



def calculate_delta_SVD(gamma,
                        z_delta,
                        z_gamma,
                        v_cut = 0.1,
                        theta_min = 0,
                        theta_max = None,
                        N = 64,
                        border_size = None,
                        border_noise = 1E3,
                        Ngal = 70,
                        z0 = 0.57,
                        sig = 0.3,
                        cosmo = None,
                        **kwargs):
    """
    Calculate delta using the SVD method
    """
    if cosmo is None:
        cosmo = Cosmology(**kwargs)
    if theta_max is None:
        theta_max = N-1
    if border_size is None:
        border_size = N/16
    
    P_gk = construct_P_gk( theta_min, theta_max, N,
                           theta_min, theta_max, N,
                           z_gamma )
    
    P_kd = construct_P_kd(N,N,z_gamma,z_delta)

    N_los = compute_N_los(z_gamma,z0,sig,Ngal)
    N_angular = compute_N_angular(N,border_size,border_noise)

    R = create_SVD_Rmatrix(P_gk,
                           P_kd,
                           N_angular,
                           N_los,
                           v_cut)
    return R.matvec(gamma), R.N_cut, R.sig_cut

def add_noise_to_gamma(gamma,
                       z_gamma,
                       z0 = 0.57,
                       sig = 0.3,
                       Ngal = 70):
    Nx = gamma.Nx
    Ny = gamma.Ny
    Nz = gamma.Nz

    N_los = compute_N_los(z_gamma,z0,sig,Ngal)
    noise = numpy.sqrt(N_los)
    
    for i in range(Nz):
        sp = gamma.source_plane(i)
        sp += ( noise[i] * numpy.random.normal(size = (Nx,Ny)) \
                * numpy.exp(2j*numpy.pi*numpy.random.random((Nx,Ny))) )

    return gamma
