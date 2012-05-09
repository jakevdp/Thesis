import numpy
from Lens3D import *
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg.eigen import arpack
from time import time

def calculate_SVD_true(P_gk,
                       P_kd,
                       N_angular,
                       N_los,
                       border_size,
                       border_noise):
    """
    compute the true SVD of the transformation
    returns U, sig, VT as Lens3D_matrix objects
    """
    Nx = P_gk.Nx
    Ny = P_gk.Ny
    Nzd = P_kd.Nz_in
    Nzg = P_kd.Nz_out

    N_a = Lens3D_vector(1,Nx,Ny,N_angular)
    N_a.set_border(border_size,border_noise)

    N1I = 1./N_a.vec

    N2I = 1./N_los
    
    NI = Lens3D_vector(Nzg,Nx,Ny,numpy.outer(N2I,N1I))

    #need the representation of P_gk
    P_gk_r = P_gk.as_Lens3D_lp_mat()

    #compute SVDs
    print "computing svds"
    U1,S1,V1 = numpy.linalg.svd(N1I[:,None]*P_gk_r.mat_list_[0],
                                full_matrices=0)
    U2,S2,V2 = numpy.linalg.svd(N2I[:,None]*P_kd.data_,
                                full_matrices=0)

    Nz1 = P_kd.Nz_in
    Nz2 = P_kd.Nz_out
    
    U_gk = Lens3D_lp_mat(Nz2, Nx, Ny, mat_list=U1)
    S_gk = Lens3D_diag(Nz2, Nx, Ny,
                       numpy.outer(numpy.ones(Nz2),S1).ravel() )
    V_gk = Lens3D_lp_mat(Nz1, Nx, Ny, mat_list=V1)

    U_kd = Lens3D_los_mat(Nz1, Nx, Ny, Nz2, U2)
    S_kd = Lens3D_diag(Nz1, Nx, Ny,
                       numpy.outer(S2,numpy.ones(Nx*Ny)).ravel() )
    V_kd = Lens3D_los_mat(Nz1, Nx, Ny, Nz1, V2)

    U = Lens3D_multi(U_gk,U_kd)
    S = Lens3D_diag(Nz1, Nx, Ny,
                    numpy.outer(S2,S1).ravel()  )
    V = Lens3D_multi(V_gk,V_kd)

    return U,S,V

def calculate_SVD_approx(P_gk,
                         P_kd,
                         N_angular,
                         N_los,
                         border_size,
                         border_noise):
    """
    compute the approximate SVD of the transformation
    returns U, sig, VT as Lens3D_matrix objects
    """
    Nx = P_gk.Nx
    Ny = P_gk.Ny
    Nzd = P_kd.Nz_in
    Nzg = P_kd.Nz_out

    N_a = Lens3D_vector(1,Nx,Ny,N_angular)
    N_a.set_border(border_size,border_noise)

    N1I = 1./N_a.vec

    N2I = 1./N_los
    
    NI = Lens3D_vector(Nzg,Nx,Ny,numpy.outer(N2I,N1I))

    #compute SVDs
    print "computing svds"
    S1 = N1I
    U2,S2,V2 = numpy.linalg.svd(N2I[:,None]*P_kd.data_,
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

def calculate_delta_svd_method(gamma,
                               P_gk,
                               P_kd,
                               N_angular,
                               N_los,
                               border_size,
                               border_noise,
                               sig_cut):
    """
    gamma should be a vector of length (Nx*Ny*Nz)
    N_angular is a vector of length (Nx*Ny)
    N_los is a vector of length Nz
    sigma gives the percentage of variance to cut out of the inversion
    """

    #compute SVDs
    #This is the big svd.  Note that if you remain in fourier space,
    # P_gk*P_gk.H is simply the identity. Thus if
    #   D P_gk = U S V.H
    # is the singular value decomposition, then
    #   D P_gk P_gk.H D = U S^2 U.H
    #   D^2 = U S^2 U.H
    # setting U = I, S = D, V.H = P_gk is a valid svd.  So we don't need
    # to calculate the svd of P_gk at all!  It's already (approximately)
    # done for us.

    Nx = gamma.Nx
    Ny = gamma.Ny
    Nzd = P_kd.Nz_in
    Nzg = P_kd.Nz_out

    N_a = Lens3D_vector(1,Nx,Ny,N_angular)
    N_a.set_border(border_size,border_noise)

    N1I = 1./N_a.vec

    N2I = 1./N_los
    
    NI = Lens3D_vector(Nzg,Nx,Ny,numpy.outer(N2I,N1I))

    #compute SVDs
    print "computing svds"
    S1 = N1I
    U2,S2,V2 = numpy.linalg.svd(N2I[:,None]*P_kd.data_,
                                full_matrices=0)

    #print len(numpy.where(S1>1)[0])
    #print (Nx-2*border_size)*(Ny-2*border_size)

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

    i_sort = numpy.argsort(S.data_)

    sig = S.data_[i_sort]
    sig *= sig
    sig_cumsum = sig.cumsum()
    sig_sum = sig.sum()

    N_tot = len(sig)

    N_cut = sig_cumsum.searchsorted(sig_sum*sig_cut)
    
    SI = Lens3D_diag(Nzd,Nx,Ny,
                     1./S.data_)
    SI.data_[i_sort[:N_cut]] = 0
    
    #compute delta
    print "computing delta"
    v1 = Lens3D_vector(Nzg,Nx,Ny,gamma.vec*NI.vec)
    v1 = U.H.matvec(v1)
    v1 = SI.matvec(v1)
    delta = V.H.matvec(v1)
    
    return delta,N_cut,v_cut,S1,S2
    

def calculate_delta_svd(gamma,
                        P_gk,
                        P_kd,
                        N_angular,
                        N_los,
                        border_size,
                        border_noise,
                        sig_cuts):
    """
    gamma should be a vector of length (Nx*Ny*Nz)
    N_angular is a vector of length (Nx*Ny)
    N_los is a vector of length Nz
    sigma gives the percentage of variance to cut out of the inversion
    """

    Nx = gamma.Nx
    Ny = gamma.Ny
    Nzd = P_kd.Nz_in
    Nzg = P_kd.Nz_out

    P_gk_r = P_gk.as_Lens3D_lp_mat()

    N_a = Lens3D_vector(1,Nx,Ny,N_angular)
    N_a.set_border(border_size,border_noise)

    N1I = 1./N_a.vec

    N2I = 1./N_los
    
    NI = Lens3D_vector(Nzg,Nx,Ny,numpy.outer(N2I,N1I))

    #compute SVDs
    print "computing svds"
    U1,S1,V1 = numpy.linalg.svd(N1I[:,None]*P_gk_r.mat_list_[0],
                                full_matrices=0)
    U2,S2,V2 = numpy.linalg.svd(N2I[:,None]*P_kd.data_,
                                full_matrices=0)

    #print len(numpy.where(S1>1)[0])
    #print (Nx-2*border_size)*(Ny-2*border_size)

    Nz1 = P_kd.Nz_in
    Nz2 = P_kd.Nz_out
    
    U_gk = Lens3D_lp_mat(Nz2, Nx, Ny, mat_list=U1)
    S_gk = Lens3D_diag(Nz2, Nx, Ny,
                       numpy.outer(numpy.ones(Nz2),S1).ravel() )
    V_gk = Lens3D_lp_mat(Nz1, Nx, Ny, mat_list=V1)

    U_kd = Lens3D_los_mat(Nz1, Nx, Ny, Nz2, U2)
    S_kd = Lens3D_diag(Nz1, Nx, Ny,
                       numpy.outer(S2,numpy.ones(Nx*Ny)).ravel() )
    V_kd = Lens3D_los_mat(Nz1, Nx, Ny, Nz1, V2)

    U = Lens3D_multi(U_gk,U_kd)
    S = Lens3D_diag(Nz1, Nx, Ny,
                    numpy.outer(S2,S1).ravel()  )
    V = Lens3D_multi(V_gk,V_kd)

    i_sort = numpy.argsort(S.data_)

    sig = S.data_[i_sort]
    sig *= sig
    sig_cumsum = sig.cumsum()
    sig_sum = sig.sum()

    N_tot = len(sig)

    N_cuts = []
    v_cuts = []
    deltas = []
    
    for sig_cut in sig_cuts:
        N_cut = sig_cumsum.searchsorted(sig_sum*sig_cut)
        v_cut = numpy.sqrt( sig[N_cut] )
    
        SI = Lens3D_diag(Nzd,Nx,Ny,
                         1./S.data_)
        SI.data_[i_sort[:N_cut]] = 0
        
        #compute delta
        print "computing delta"
        v1 = Lens3D_vector(Nzg,Nx,Ny,gamma.vec*NI.vec)
        v1 = U.H.matvec(v1)
        v1 = SI.matvec(v1)
        delta = V.H.matvec(v1)

        N_cuts.append(N_cut)
        v_cuts.append(v_cut)
        deltas.append(delta)
    
    return deltas,N_cuts,v_cuts,S1,S2



def calculate_delta_svd_partial(gamma,
                                P_gk,
                                P_kd,
                                N_angular,
                                N_los,
                                border_size,
                                border_noise,
                                sig_cuts):
    """
    gamma should be a vector of length (Nx*Ny*Nz)
    N_angular is a vector of length (Nx*Ny)
    N_los is a vector of length Nz
    sigma gives the percentage of variance to cut out of the inversion
    """

    Nx = gamma.Nx
    Ny = gamma.Ny
    Nzd = P_kd.Nz_in
    Nzg = P_kd.Nz_out

    P_gk_r = P_gk.as_Lens3D_lp_mat()

    N_a = Lens3D_vector(1,Nx,Ny,N_angular)
    N_a.set_border(border_size,border_noise)

    N1I = 1./N_a.vec

    N2I = 1./N_los
    
    NI = Lens3D_vector(Nzg,Nx,Ny,numpy.outer(N2I,N1I))

    num_sing_vals = (Nx-2*border_size)*(Ny-2*border_size)+1

    #compute SVDs
    print "computing svds"
    U1,S1,V1 = numpy.linalg.svd(N1I[:,None]*P_gk_r.mat_list_[0],
                                full_matrices=0)
    
    U1 = U1[:,:num_sing_vals]
    S1 = S1[:num_sing_vals]
    V1 = V1[:num_sing_vals,:]

    U2,S2,V2 = numpy.linalg.svd(N2I[:,None]*P_kd.data_,
                                full_matrices=0)

    Nz1 = P_kd.Nz_in
    Nz2 = P_kd.Nz_out
    
    U_gk = Lens3D_lp_mat(Nz2, num_sing_vals,1,Nx, Ny, 
                         mat_list=U1)
    S_gk = Lens3D_lp_diag(Nz2, num_sing_vals, 1,
                          S1)
    V_gk = Lens3D_lp_mat(Nz1, Nx, Ny, 
                         num_sing_vals,1,mat_list=V1)

    U_kd = Lens3D_los_mat(Nz1, Nx, Ny, Nz2, U2)
    S_kd = Lens3D_diag(Nz1, Nx, Ny,
                       numpy.outer(S2,numpy.ones(Nx*Ny)).ravel() )
    V_kd = Lens3D_los_mat(Nz1, Nx, Ny, Nz1, V2)

    U = Lens3D_multi(U_kd,U_gk)

    S = Lens3D_diag(Nz1, num_sing_vals,1,
                    numpy.outer(S2,S1).ravel()  )
    
    V = Lens3D_multi(V_gk,V_kd)

    i_sort = numpy.argsort(S.data_)

    sig = S.data_[i_sort]
    sig *= sig
    sig_cumsum = sig.cumsum()
    sig_sum = sig.sum()

    N_tot = len(sig)

    N_cuts = []
    v_cuts = []
    deltas = []
    
    for sig_cut in sig_cuts:
        N_cut = sig_cumsum.searchsorted(sig_sum*sig_cut)
        v_cut = numpy.sqrt( sig[N_cut] )
    
        SI = Lens3D_diag(Nzd,Nx,Ny,
                         1./S.data_)
        SI.data_[i_sort[:N_cut]] = 0
        
        #compute delta
        print "computing delta"
        v1 = Lens3D_vector(Nzg,Nx,Ny,gamma.vec*NI.vec)
        v1 = U.H.matvec(v1)
        v1 = SI.matvec(v1)
        delta = V.H.matvec(v1)

        N_cuts.append(N_cut)
        v_cuts.append(v_cut)
        deltas.append(delta)
    
    return deltas,N_cuts,v_cuts,S1,S2

def calculate_noise_nosvd(gamma,P_gk,P_kd,N_angular,N_los,
                          border_size,border_noise,
                          sig_cuts, i_x, i_y):
    """
    Noise is V sig^-2 VT
    """

    #compute SVDs
    #This is the big svd.  Note that if you remain in fourier space,
    # P_gk*P_gk.H is simply the identity. Thus if
    #   D P_gk = U S V.H
    # is the singular value decomposition, then
    #   D P_gk P_gk.H D = U S^2 U.H
    #   D^2 = U S^2 U.H
    # setting U = I, S = D, V.H = P_gk is a valid svd.  So we don't need
    # to calculate the svd of P_gk at all!  It's already (approximately)
    # done for us.

    Nx = gamma.Nx
    Ny = gamma.Ny
    Nzd = P_kd.Nz_in
    Nzg = P_kd.Nz_out

    N_a = Lens3D_vector(1,Nx,Ny,N_angular)
    N_a.set_border(border_size,border_noise)

    N1I = 1./N_a.vec

    N2I = 1./N_los
    
    NI = Lens3D_vector(Nzg,Nx,Ny,numpy.outer(N2I,N1I))

    #compute SVDs
    print "computing svds"
    S1 = N1I
    U2,S2,V2 = numpy.linalg.svd(N2I[:,None]*P_kd.data_,
                                full_matrices=0)

    #print len(numpy.where(S1>1)[0])
    #print (Nx-2*border_size)*(Ny-2*border_size)

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

    i_sort = numpy.argsort(S.data_)

    sig = S.data_[i_sort]
    sig *= sig
    sig_cumsum = sig.cumsum()
    sig_sum = sig.sum()

    print "sum of sigma^2 =", sig_sum

    N_tot = len(sig)

    noises = []

    #v = U.H.matvec( (1+1j)*numpy.ones( Nzg*Nx*Ny, dtype=complex) )
    
    for sig_cut in sig_cuts:
        N_cut = sig_cumsum.searchsorted(sig_sum*sig_cut)
    
        SI = Lens3D_diag(Nzd,Nx,Ny,
                         1./S.data_)
        SI.data_[i_sort[:N_cut]] = 0

        #v1 = SI.matvec(v)
        #v1 = V.H.matvec(v1)
        #noises.append( abs( v1.reshape((Nzd,Nx,Ny) )[:,i_x,i_y].real ) )
        #
        #continue
        
        noises.append([])
        
        for i in range(Nzd):
            v = numpy.zeros((Nzd,Nx,Ny),dtype=complex)
            v[i,i_x,i_y] = 1
            v = v.ravel()
            v1 = V.matvec(v)
            v1 *= SI.data_
            v1 *= SI.data_
            v1 *= SI.data_
            v1 *= SI.data_
            v1 = V.H.matvec(v1)

            noises[-1].append( numpy.sqrt(v1.reshape((Nzd,Nx,Ny))[i,i_x,i_y]) )
    noises = numpy.asarray(noises)
    
    return noises
    

def calculate_delta_nosvd(gamma,
                          P_gk,
                          P_kd,
                          N_angular,
                          N_los,
                          border_size,
                          border_noise,
                          sig_cuts):
    """
    gamma should be a vector of length (Nx*Ny*Nz)
    N_angular is a vector of length (Nx*Ny)
    N_los is a vector of length Nz
    sigma gives the percentage of variance to cut out of the inversion
    """

    #compute SVDs
    #This is the big svd.  Note that if you remain in fourier space,
    # P_gk*P_gk.H is simply the identity. Thus if
    #   D P_gk = U S V.H
    # is the singular value decomposition, then
    #   D P_gk P_gk.H D = U S^2 U.H
    #   D^2 = U S^2 U.H
    # setting U = I, S = D, V.H = P_gk is a valid svd.  So we don't need
    # to calculate the svd of P_gk at all!  It's already (approximately)
    # done for us.

    Nx = gamma.Nx
    Ny = gamma.Ny
    Nzd = P_kd.Nz_in
    Nzg = P_kd.Nz_out

    N_a = Lens3D_vector(1,Nx,Ny,N_angular)
    N_a.set_border(border_size,border_noise)

    N1I = 1./N_a.vec

    N2I = 1./N_los
    
    NI = Lens3D_vector(Nzg,Nx,Ny,numpy.outer(N2I,N1I))

    #compute SVDs
    print "computing svds"
    S1 = N1I
    U2,S2,V2 = numpy.linalg.svd(N2I[:,None]*P_kd.data_,
                                full_matrices=0)

    #print len(numpy.where(S1>1)[0])
    #print (Nx-2*border_size)*(Ny-2*border_size)

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

    i_sort = numpy.argsort(S.data_)

    sig = S.data_[i_sort]
    sig *= sig
    sig_cumsum = sig.cumsum()
    sig_sum = sig.sum()

    N_tot = len(sig)

    N_cuts = []
    v_cuts = []
    deltas = []
    
    for sig_cut in sig_cuts:
        t0 = time()
        N_cut = sig_cumsum.searchsorted(sig_sum*sig_cut)
        v_cut = numpy.sqrt( sig[N_cut] )
    
        SI = Lens3D_diag(Nzd,Nx,Ny,
                         1./S.data_)
        SI.data_[i_sort[:N_cut]] = 0
        
        #compute delta
        print "computing delta"
        v1 = Lens3D_vector(Nzg,Nx,Ny,gamma.vec*NI.vec)
        v1 = U.H.matvec(v1)
        v1 = SI.matvec(v1)
        delta = V.H.matvec(v1)
        t = time()-t0
        
        N_cuts.append(N_cut)
        v_cuts.append(v_cut)
        deltas.append(delta)
        print " - delta computed in %.2g sec" % t

    return deltas,N_cuts,v_cuts,S1,S2


def calculate_delta_arpack(gamma,
                           P_gk,
                           P_kd,
                           N_angular,
                           N_los,
                           border_size,
                           border_noise,
                           sig_cuts):
    """
    gamma should be a vector of length (Nx*Ny*Nz)
    N_angular is a vector of length (Nx*Ny)
    N_los is a vector of length Nz
    sigma gives the percentage of variance to cut out of the inversion
    """

    Nx = gamma.Nx
    Ny = gamma.Ny
    Nz1 = P_kd.Nz_in
    Nz2 = P_kd.Nz_out

    #for all the lens-plane operations, we'll need two versions of
    # Lens3D matrices, with Nz1 and Nz2 source planes.
    P_gk1 = Lens3D_lp_conv(Nz1,Nx,Ny,
                           P_gk.dx_,P_gk.dy_,
                           func=P_gk.func_[0],
                           func_ft=P_gk.func_ft_[0])
    P_gk2 = P_gk


    #get a single convolution plane
    P_gk_one = Lens3D_lp_conv(1,Nx,Ny,
                              P_gk.dx_,P_gk.dy_,
                              func=P_gk.func_[0],
                              func_ft=P_gk.func_ft_[0])
    P_gk_one_H = P_gk_one.H
    

    N_a = Lens3D_vector(1,Nx,Ny,N_angular)
    N_a.set_border(border_size,border_noise)

    N1I = 1./N_a.vec

    N2I = 1./N_los
    
    NI = Lens3D_vector(Nz2,Nx,Ny,numpy.outer(N2I,N1I))

    #use arpack to find singular vectors
    num_sing_vals = (Nx-2*border_size)*(Ny-2*border_size)+1
    
    use_arpack = True
    if use_arpack:
        #define the function to send to ARPACK
        def matvec(v):
            matvec.N += 1
            if matvec.N % 100 == 0: print matvec.N
            vv = v*N1I
            #vv = P_gk_one.view_as_Lens3D_vec(vv)
            vv = P_gk_one_H.matvec(vv)
            vv = P_gk_one.matvec(vv)
            #vv = P_gk_one.view_as_same_type(vv,v)
            vv *= N1I
            return vv
        matvec.N = 0
    
    
        DMMD = LinearOperator(shape = (Nx*Ny,Nx*Ny),
                              matvec = matvec,
                              dtype = complex)

        print "ARPACK: finding %i singular values of [%ix%i] matrix" \
            % (num_sing_vals,DMMD.shape[0],DMMD.shape[1])
        t = time()
        S1_2,U1 = arpack.eigen(DMMD,num_sing_vals,which='LR')
        t = time()-t
        print "   finished in %i iterations" % matvec.N
        print "      (time: %.3g sec: %.2g sec per iteration)" \
            % (t,t*1./matvec.N)
        exit()
        S1 = numpy.sqrt(S1_2.real)
    else:
        P_gk_r = P_gk.as_Lens3D_lp_mat()
        U1,S1,V1 = numpy.linalg.svd(N1I[:,None]*P_gk_r.mat_list_[0],
                                    full_matrices=0)
        S1 = S1[:num_sing_vals]
        U1 = U1[:,:num_sing_vals]

    compare_to_direct = False
    if compare_to_direct:
        print "compare to direct svd"
        print "computing svds"
        S1.sort()
        pylab.plot( S1[::-1] )
        
        P_gk_r = P_gk.as_Lens3D_lp_mat()
        U1,S1,V1 = numpy.linalg.svd(N1I[:,None]*P_gk_r.mat_list_[0],
                                    full_matrices=0)

        pylab.plot( S1[:num_sing_vals] )
        pylab.show()
        exit()

    U2,S2,V2 = numpy.linalg.svd(N2I[:,None]*P_kd.data_,
                                full_matrices=0)

    Nz1 = P_kd.Nz_in
    Nz2 = P_kd.Nz_out
    
    U_gk1 = Lens3D_lp_mat(Nz1, num_sing_vals,1,Nx, Ny, 
                          mat_list=U1)
    U_gk2 = Lens3D_lp_mat(Nz2, num_sing_vals,1,Nx, Ny, 
                          mat_list=U1)
    S_gk1 = Lens3D_lp_diag(Nz1, num_sing_vals, 1,
                           S1)
    S_gk2 = Lens3D_lp_diag(Nz2, num_sing_vals, 1,
                           S1)

    #V_gk will be S_gk^-1 * U_gk^H * N1I * N1I * P_gk
    V_gk = Lens3D_multi(Lens3D_lp_diag(Nz1, num_sing_vals, 1, 1./S1),
                         U_gk1.H,
                         Lens3D_lp_diag(Nz1,Nx,Ny,N1I),
                         P_gk1)
    
    U_kd = Lens3D_los_mat(Nz1, Nx, Ny, Nz2, U2)
    S_kd = Lens3D_diag(Nz1, Nx, Ny,
                       numpy.outer(S2,numpy.ones(Nx*Ny)).ravel() )
    V_kd = Lens3D_los_mat(Nz1, Nx, Ny, Nz1, V2)

    U = Lens3D_multi(U_kd,U_gk1)

    S = Lens3D_diag(Nz1, num_sing_vals,1,
                    numpy.outer(S2,S1).ravel()  )

    V = Lens3D_multi(V_gk,V_kd)

    i_sort = numpy.argsort(S.data_)

    sig = S.data_[i_sort]
    sig *= sig
    sig_cumsum = sig.cumsum()
    sig_sum = sig.sum()

    N_tot = len(sig)

    N_cuts = []
    v_cuts = []
    deltas = []
    
    for sig_cut in sig_cuts:
        N_cut = sig_cumsum.searchsorted(sig_sum*sig_cut)
        v_cut = numpy.sqrt( sig[N_cut] )
    
        SI = Lens3D_diag(Nzd,Nx,Ny,
                         1./S.data_)
        SI.data_[i_sort[:N_cut]] = 0
        
        #compute delta
        print "computing delta"
        v1 = Lens3D_vector(Nzg,Nx,Ny,gamma.vec*NI.vec)
        v1 = U.H.matvec(v1)
        v1 = SI.matvec(v1)
        delta = V.H.matvec(v1)

        N_cuts.append(N_cut)
        v_cuts.append(v_cut)
        deltas.append(delta)
    
    return deltas,N_cuts,v_cuts,S1,S2
