"""
Use masked KL to reconstruct DES mock shear fields

"""
import numpy
import pylab
from scipy import linalg

from tools import *
from ..shear_correlation import xi_plus
from ..shear_KL import compute_correlation_matrix, compute_KL
from ..Map_peaks import pyQ_NFW, Map_map

def reconstruct_shear(shear_in,
                      mask,
                      evals,
                      evecs,
                      n,
                      alpha=0,
                      full_output = False):
    """
    given whitented input shear, a mask, and an eigenbasis,
    eigenbasis is assumed to be from the whitened covariance matrix
    compute the reconstruction coefficients from the first n modes
    n is the number of eigenmodes to use
    alpha is the level of Wiener filtering (between 0&1, inclusive)
    """
    assert alpha>=0 and alpha<=1

    evecs_n = evecs[:,:n]
    evals_n = evals[:n]

    #create mixing matrix, using Connolly&Szalay 1999 technique
    M = numpy.dot(evecs_n.T*mask,evecs_n)

    #apply wiener filter if alpha>0
    if alpha!=0:
        M += alpha*numpy.diag(evals_n**-1)
        
    #compute a_n = M_n^-1 F
    F = numpy.dot(evecs_n.T,mask*shear_in)
    an = numpy.linalg.solve(M,F)

    shear_r = numpy.dot(evecs[:,:n],an)

    if full_output:
        return shear_r, M, an
    else:
        return shear_r
    

def reconstruct_shear_N(shear_in,
                        Ngal,
                        sigma,
                        ngal_pix,
                        evals,
                        evecs,
                        nmodes,
                        alpha=0,
                        full_output = False,
                        weight_by_noise = False):
    """
    given whitented input shear, an eigenbasis,
    eigenbasis is assumed to be from the whitened covariance matrix
    compute the reconstruction coefficients from the first n modes
    n is the number of eigenmodes to use
    N is the diagonal of the noise covariance of the shear
    alpha is the level of Wiener filtering (between 0&1, inclusive)

    Ngal is the observed count of galaxies: zero in masked regions
    sigma is the intrinsic ellipticity
    ngal is the number of galaxies per pixel assumed in the eigenbasis

    weight_by_noise tells whether to use a weighted mask or a binary mask
    """
    assert alpha>=0 and alpha<=1

    #the mask and noise covariance we use depends on the problem:
    # use inverse noise to weight the input, or weight it equally?
    i = numpy.where(Ngal==0)
    
    if weight_by_noise:
        mask = Ngal

        #first option: equal weighting. This leads to amplified shear in
        # masked regions for alpha=1
        N = sigma**2/ngal_pix * numpy.ones(Ngal.shape)

        #second option: whiten based on noise.  This leads to suppressed shear
        # in masked regions.  alpha doesn't seem to matter here
        #Ngal[i]=ngal_pix
        #N = sigma**2/Ngal
    else:
        mask = numpy.ones(Ngal.shape)
        mask[i] = 0
        #first option: don't whiten the input.  For alpha=1, this seems to
        # give the most reasonable reconstruction
        #N=1

        #second option: whiten the input.  This leads to suppressed
        Ngal[i]=ngal_pix
        N = sigma**2/Ngal

    N1_2 = numpy.sqrt(N)

    if numpy.all(N1_2==0):
        N1_2=1

    evecs_n = evecs[:,:nmodes]
    evals_n = evals[:nmodes]

    #create mixing matrix, using Connolly&Szalay 1999 technique
    M = numpy.dot(evecs_n.T*mask,evecs_n)

    #apply wiener filter if alpha>0
    if alpha!=0:
        M += alpha*numpy.diag(evals_n**-1)
        
    #compute a_n = M_n^-1 F
    F = numpy.dot(evecs_n.T,mask * shear_in/N1_2)
    an = numpy.linalg.solve(M,F)

    shear_r = N1_2*numpy.dot(evecs_n,an)

    if full_output:
        return shear_r, M, an
    else:
        return shear_r
    
def reconstruct_shear_with_noise(shear_in,
                                 Ngal,
                                 sigma,
                                 ngal_pix,
                                 evals,
                                 evecs,
                                 nmodes,
                                 alpha=0,
                                 full_output = False,
                                 weight_by_noise = False,
                                 compute_shear_noise = False):
    """
    given whitented input shear, an eigenbasis,
    eigenbasis is assumed to be from the whitened covariance matrix
    compute the reconstruction coefficients from the first n modes
    n is the number of eigenmodes to use
    N is the diagonal of the noise covariance of the shear
    alpha is the level of Wiener filtering (between 0&1, inclusive)

    Ngal is the observed count of galaxies: zero in masked regions
    sigma is the intrinsic ellipticity
    ngal is the number of galaxies per pixel assumed in the eigenbasis

    weight_by_noise tells whether to use a weighted mask or a binary mask
    """
    assert alpha>=0 and alpha<=1

    #the mask and noise covariance we use depends on the problem:
    # use inverse noise to weight the input, or weight it equally?
    i = numpy.where(Ngal==0)
    
    if weight_by_noise:
        mask = Ngal

        #first option: equal weighting. This leads to amplified shear in
        # masked regions for alpha=1
        N = sigma**2/ngal_pix * numpy.ones(Ngal.shape)

        #second option: whiten based on noise.  This leads to suppressed shear
        # in masked regions.  alpha doesn't seem to matter here
        #Ngal[i]=ngal_pix
        #N = sigma**2/Ngal
    else:
        mask = numpy.ones(Ngal.shape)
        mask[i] = 0
        #first option: don't whiten the input.  For alpha=1, this seems to
        # give the most reasonable reconstruction
        #N=1

        #second option: whiten the input.  This leads to suppressed
        Ngal[i]=ngal_pix
        N = sigma**2/Ngal

    N1_2 = numpy.sqrt(N)

    if numpy.all(N1_2==0):
        N1_2=1

    evecs_n = evecs[:,:nmodes]
    evals_n = evals[:nmodes]

    #create mixing matrix, using Connolly&Szalay 1999 technique
    M = numpy.dot(evecs_n.T*mask,evecs_n)

    #apply wiener filter if alpha>0
    if alpha!=0:
        M += alpha*numpy.diag(evals_n**-1)

    #compute LU decomposition of M
    LU = linalg.lu_factor(M)

    #reconstruct the shear
    # compute a_n = M^-1 F
    F = numpy.dot(evecs_n.T,mask * shear_in/N1_2)
    an = linalg.lu_solve(LU,F)
    # use a_n to reconstruct shear
    shear_r = N1_2*numpy.dot(evecs_n,an)

    if compute_shear_noise:
        NN = len(shear_r)
        Ni = numpy.zeros(NN)
        Nshear = numpy.zeros(NN)
        for i in range(NN):
            Ni[i] = N[i]
            x = numpy.dot(evecs_n.T,Ni)
            x = linalg.lu_solve(LU,x)
            x = numpy.dot(evecs_n,x)
            x *= mask**2
            x = numpy.dot(evecs_n.T,x)
            x = linalg.lu_solve(LU,x)
            x = numpy.dot(evecs_n,x)
            Nshear[i] = numpy.dot(Ni,x)
            Ni[i] = 0

        if full_output:
            return shear_r, Nshear, M, an
        else:
            return shear_r, Nshear

    else:
        if full_output:
            return shear_r, M, an
        else:
            return shear_r
    
def reconstruct_shear_and_Map(shear_in,
                              Ngal,
                              sigma,
                              ngal_pix,
                              evals,
                              evecs,
                              nmodes,
                              alpha=0,
                              dtheta = None, #arcmin
                              rmax = None, #arcmin
                              xc = None,
                              full_output = False,
                              weight_by_noise = False,
                              compute_shear_noise = False,
                              compute_Map = False,
                              compute_Map_noise = False):
    """
    given whitented input shear, an eigenbasis,
    eigenbasis is assumed to be from the whitened covariance matrix
    compute the reconstruction coefficients from the first n modes
    n is the number of eigenmodes to use
    N is the diagonal of the noise covariance of the shear
    alpha is the level of Wiener filtering (between 0&1, inclusive)

    Ngal is the observed count of galaxies: zero in masked regions
    sigma is the intrinsic ellipticity
    ngal is the number of galaxies per pixel assumed in the eigenbasis

    weight_by_noise tells whether to use a weighted mask or a binary mask

    This also computes the aperture mass (and optionally noise)
    at the central pixels
    """
    assert alpha>=0 and alpha<=1

    #the mask and noise covariance we use depends on the problem:
    # use inverse noise to weight the input, or weight it equally?
    i = numpy.where(Ngal==0)
    
    if weight_by_noise:
        mask = Ngal

        #first option: equal weighting. This leads to amplified shear in
        # masked regions for alpha=1
        N = sigma**2/ngal_pix * numpy.ones(Ngal.shape)

        #second option: whiten based on noise.  This leads to suppressed shear
        # in masked regions.  alpha doesn't seem to matter here
        #Ngal[i]=ngal_pix
        #N = sigma**2/Ngal
    else:
        mask = numpy.ones(Ngal.shape)
        mask[i] = 0
        #first option: don't whiten the input.  For alpha=1, this seems to
        # give the most reasonable reconstruction
        #N=1

        #second option: whiten the input.  This leads to suppressed
        Ngal[i]=ngal_pix
        N = sigma**2/Ngal

    N1_2 = numpy.sqrt(N)

    #Nx,Ny is the shape of the field
    Nx = Ny = int(numpy.sqrt(len(N)))
    assert Nx==Ny

    if numpy.all(N1_2==0):
        N1_2=1

    evecs_n = evecs[:,:nmodes]
    evals_n = evals[:nmodes]

    #create mixing matrix, using Connolly&Szalay 1999 technique
    M = numpy.dot(evecs_n.T*mask,evecs_n)

    #apply wiener filter if alpha>0
    if alpha!=0:
        M += alpha*numpy.diag(evals_n**-1)

    #compute LU decomposition of M
    LU = linalg.lu_factor(M)

    #reconstruct the shear
    # compute a_n = M^-1 F
    F = numpy.dot(evecs_n.T,mask * shear_in/N1_2)
    an = linalg.lu_solve(LU,F)
    # use a_n to reconstruct shear
    shear_r = N1_2*numpy.dot(evecs_n,an)

    #compute positions
    pos_x = dtheta*numpy.arange(Nx)
    pos_y = dtheta*numpy.arange(Ny)
    pos = pos_x[:,None] + 1j*pos_y[None,:]

    ret = (shear_r,)

    if compute_shear_noise:
        NN = len(shear_r)
        Ni = numpy.zeros(NN)
        Nshear = numpy.zeros(NN)
        for i in range(NN):
            Ni[i] = N1_2[i]
            x = numpy.dot(evecs_n.T,Ni)
            x = linalg.lu_solve(LU,x)
            x = numpy.dot(evecs_n,x)
            x *= mask**2
            x = numpy.dot(evecs_n.T,x)
            x = linalg.lu_solve(LU,x)
            x = numpy.dot(evecs_n,x)
            Nshear[i] = numpy.dot(Ni,x)
            Ni[i] = 0

        ret += (Nshear,)

    pos_M = pos[Nx/4 : 3*Nx/4, Ny/4 : 3*Ny/4]    

    if compute_Map:
        MapE,MapB = Map_map(shear_r.reshape((Nx,Ny)),
                            pos.reshape((Nx,Ny)),
                            pos_M.reshape((Nx/2,Ny/2)),
                            rmax,xc)
        Map = MapE + 1j*MapB

        ret += (Map.reshape(Nx*Ny/4),)

    if compute_Map_noise:
        N_M = numpy.zeros(pos_M.shape)
        for i in range(pos_M.shape[0]):
            for j in range(pos_M.shape[1]):
                Qi = pyQ_NFW(abs(pos-pos_M[i,j])/rmax,
                             xc).reshape(Nx*Ny)
                Qi[numpy.where(numpy.isnan(Qi))] = 0

                x = Qi*N1_2
                x = numpy.dot(evecs_n.T,x)
                x = linalg.lu_solve(LU,x)
                x = numpy.dot(evecs_n,x)
                x *= mask**2
                x = numpy.dot(evecs_n.T,x)
                x = linalg.lu_solve(LU,x)
                x = numpy.dot(evecs_n,x)
                N_M[i,j] = numpy.dot(Qi,x*N1_2)/(numpy.sum(Qi)**2)

        ret += (N_M,)

    if full_output:
        ret += (M,an)

    return ret
        
    


def plot_masked_shear_reconstructions(filename='shear_out.dat',
                                      alpha = 0,
                                      mask_frac = 0.1,
                                      sigma = 0,
                                      neigs=[100,400,800,1000],
                                      whiten=True):
    #----------------------------------------------------------------------
    #load shear
    shear,dtheta,Ngal = read_shear_out(filename,return_N=True)

    ngal_mean = numpy.mean(Ngal)
    shot_noise = sigma**2/ngal_mean

    print "Noise: sigma = %.2f" % sigma
    print "       |ngal| = %.1f/arcmin^2" % (ngal_mean/dtheta**2)

    N = shear.shape[0]
    assert N==shear.shape[1]

    print N,N,"dtheta =",dtheta
                
    #----------------------------------------------------------------------   
    #create mask
    assert mask_frac>=0 and mask_frac<=1
    
    numpy.random.seed(1)
    mask = create_mask(N,N,mask_frac)
    
    #----------------------------------------------------------------------
    #add noise to shear, with intrinsic ellipticity given by sigma
    if sigma:
        noise = numpy.zeros((N,N),dtype=complex)
        noise += sigma/numpy.sqrt(Ngal)*numpy.random.normal(size=(N,N))
        noise *= numpy.exp(2j*numpy.pi*numpy.random.random((N,N)))
        shear_true = shear.copy()
        shear += noise
    else:
        shear_true = shear
    
    #----------------------------------------------------------------------
    # construct correlation matrix
    print "constructing correlation matrix"
    n = lambda z,z0=0.5: z**2 * numpy.exp(-(z/z0)**1.5)
    zlim = (0,3)
    xi = xi_plus(n,zlim,Nz=20,Nell=5E4,Or=0)
    C = compute_correlation_matrix(xi,N,dtheta,
                                   ngal = ngal_mean,
                                   sigma = sigma,
                                   whiten = whiten)
    evals,evecs = compute_KL(C)

    #----------------------------------------------------------------------
    #plot eigenvalues
    pylab.figure(1)
    if whiten:
        pylab.semilogy(range(1,N*N+1),evals,'-k')
        pylab.semilogy(range(1,N*N+1),evals-1,'--k')
        pylab.semilogy(range(1,N*N+1),numpy.ones(N*N),':k')
        pylab.xlim(0,N*N)
    else:
        pylab.loglog(range(1,N*N+1),evals,'-k')
        pylab.xlim(1,N*N)
    pylab.title('Eigenvalues of Shear Covariance')
    pylab.text(0.7,0.9,r"$\langle N_{gal}\rangle = %.1f$" % ngal_mean,
               fontsize=14,
               transform = pylab.gca().transAxes)
    pylab.text(0.7,0.8,r"$\sigma_\gamma = %.2f$" % sigma,
               fontsize=14,
               transform = pylab.gca().transAxes)
    
    #----------------------------------------------------------------------
    #create function to plot the mask
    def plotmask():
        pylab.imshow(mask.reshape(N,N).T,
                     cmap=GreyWhite,
                     origin='lower',interpolation='nearest',
                     extent=[-0.5*dtheta,dtheta*(N-0.5),
                             -0.5*dtheta,dtheta*(N-0.5)])

    #----------------------------------------------------------------------
    #plot noiseless & noisy input shear
    pylab.figure(2,figsize=(8,11))

    pylab.subplot(321)
    plotmask()
    whiskerplot(shear_true,
                dtheta,dtheta)
    pylab.title('true shear')
    pylab.xlabel('')

    pylab.subplot(322)
    plotmask()
    pylab.colorbar()

    whiskerplot(shear,
                dtheta,dtheta)
    pylab.title('noisy shear')
    pylab.xlabel('')
    
    mask.resize(N*N)
    shear.resize(N*N)
    Ngal.resize(N*N)
    
    #----------------------------------------------------------------------
    #plot four reconstructions

    #construct whitened shear
    if whiten:
        shear_w = shear*numpy.sqrt(Ngal)/sigma
    else:
        shear_w = shear

    for i in range(4):
        n = neigs[i]

        #compute shear reconstruction
        shear_i,Mn,an = reconstruct_shear(shear_w, mask,
                                          evals, evecs,
                                          n, alpha,
                                          full_output = True)
        if whiten:
            shear_i *= sigma/numpy.sqrt(Ngal)

        #plot reconstruction
        pylab.subplot(323+i)
        plotmask()
        whiskerplot(shear_i.reshape((N,N)),
                    dtheta,dtheta)
        pylab.title('%i modes' % n)

        if i in (0,1):
            pylab.xlabel('') 
        

def compute_rms(filename = '../shear_out.dat',
                mask_frac = 0.1,
                sigma = 0.3,
                alpha = numpy.logspace(-5,0,6),
                neigs = numpy.arange(100,1000,100),
                whiten = True):
    #----------------------------------------------------------------------
    #load shear
    shear,dtheta,Ngal = read_shear_out(filename,return_N=True)

    ngal_mean = numpy.mean(Ngal)
    shot_noise = sigma**2/ngal_mean

    print "Noise: sigma = %.2f" % sigma
    print "       |ngal| = %.1f/arcmin^2" % (ngal_mean/dtheta**2)
    
    N = shear.shape[0]
    assert N==shear.shape[1]
    
    print N,N,"dtheta =",dtheta
    
    #----------------------------------------------------------------------   
    #create mask
    assert mask_frac>=0 and mask_frac<=1
                
    numpy.random.seed(0)
    mask = create_mask(N,N,mask_frac)
    
    #----------------------------------------------------------------------
    #add noise to shear, with intrinsic ellipticity given by sigma
    if sigma:
        noise = numpy.zeros((N,N),dtype=complex)
        noise += sigma/numpy.sqrt(Ngal)*numpy.random.normal(size=(N,N))
        noise *= numpy.exp(2j*numpy.pi*numpy.random.random((N,N)))
        shear_true = shear.copy()
        shear += noise
    else:
        shear_true = shear
    
        
    #----------------------------------------------------------------------
    # construct correlation matrix
    print "constructing correlation matrix"
    n = lambda z,z0=0.5: z**2 * numpy.exp(-(z/z0)**1.5)
    zlim = (0,3)
    xi = xi_plus(n,zlim,Nz=20,Nell=5E4,Or=0)
    C = compute_correlation_matrix(xi,N,dtheta,
                                   ngal = ngal_mean,
                                   sigma = sigma,
                                   whiten = whiten)
    evals,evecs = compute_KL(C)

    #construct whitened shear
    mask.resize(N*N)
    shear.resize(N*N)
    shear_true.resize(N*N)
    Ngal.resize(N*N)

    if whiten: shear_w = shear*numpy.sqrt(Ngal)/sigma
    else: shear_w = shear

    #reference rms is the difference between noisy shear and true shear
    diff = shear-shear_true
    rms0 = numpy.sqrt(numpy.mean(diff**2))

    #----------------------------------------------------------------------
    #compute all the RMS values
    rms = numpy.zeros( (len(neigs),len(alpha)),dtype=float )

    for i in range(len(neigs)):
        n = neigs[i]
        print n
        for j in range(len(alpha)):
            a = alpha[j]
            print '  ',a
            shear_ij = reconstruct_shear(shear_w, mask,
                                         evals, evecs,
                                         n, a)
            if whiten:
                shear_ij *= sigma/numpy.sqrt(Ngal)
            
            diff = shear_ij-shear_true
            #rms[i,j] = numpy.sqrt(numpy.mean(mask*diff**2))
            rms[i,j] = numpy.sqrt(numpy.mean(diff**2))
    

    #pylab.imshow(numpy.log10(rms.T),
    #             origin='lower',
    #             interpolation='nearest',
    #             aspect = (neigs[-1]-neigs[0])/numpy.log10(alpha[-1]/alpha[0]),
    #             extent = (neigs[0],neigs[-1],
    #                       numpy.log10(alpha[0]),
    #                       numpy.log10(alpha[-1])) )
    #pylab.xlabel('neigs')
    #pylab.ylabel(r'$\rm{log}(\alpha)$')
    #pylab.colorbar().set_label('log10(rms)')

    return rms
               

if __name__ == '__main__':
    """
    alpha = 0
    neigs = numpy.arange(200,401)

    rms = compute_rms(filename = '../shear_fields/shear_out_40.dat',
                      mask_frac = 0.1,
                      sigma = 0.3,
                      alpha = [alpha],
                      neigs = neigs,
                      whiten = False)

    pylab.plot(neigs,
               rms.reshape(rms.size))
    pylab.show()
    exit()
    """
    

    """
    #alpha = numpy.logspace(-5,0,11)
    #neigs = numpy.arange(50,1050,50)

    #alpha = numpy.logspace(-2,0,20)
    #neigs = numpy.arange(100,450,10)
    
    alpha = numpy.logspace(-6,-4,11)
    neigs = numpy.arange(200,400,10) 

    rms = compute_rms(filename = '../shear_fields/shear_out.dat',
                      mask_frac = 0.1,
                      sigma = 0.3,
                      alpha = alpha,
                      neigs = neigs,
                      whiten = False)
    
    OF = open('output/rms.npz','w')
    numpy.savez(OF,alpha,neigs,rms)

    pylab.show()
    """


    
    
    alpha = 1.E-4

    plot_masked_shear_reconstructions(filename='../shear_out.dat',
                                      alpha=alpha,
                                      sigma=0.3,
                                      whiten=False)
    pylab.figure(1)
    pylab.savefig('fig/shear_evals.pdf')
    pylab.figure(2)
    pylab.savefig('fig/shear_masked.pdf')
    pylab.show()
    
