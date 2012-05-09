"""
cosmological parameter estimation using KL modes of shear observations
"""
import numpy
import pylab


from ..shear_correlation import xi_plus
from ..shear_KL import compute_correlation_matrix, compute_KL
from ..cosmology import Cosmology



def plot_signal_to_noise(N=32, dtheta=5.0,
                         ngal=40,
                         sigma=0.3):
    """
    plot signal to noise of the correlation matrix eigenmodes
    """
    #----------------------------------------------------------------------
    # construct correlation matrix
    print "constructing correlation matrix"
    n = lambda z,z0=0.5: z**2 * numpy.exp(-(z/z0)**1.5)
    zlim = (0,3)
    xi = xi_plus(n,zlim,Nz=20,Nell=5E4,Or=0)
    
    C = compute_correlation_matrix(xi,N,dtheta,
                                   ngal=ngal*dtheta**2,
                                   sigma = sigma,
                                   whiten = True)
    evals,evecs = compute_KL(C)

    pylab.semilogy(range(1,N*N+1),evals,'-k',
                   label='signal+noise')
    pylab.semilogy(range(1,N*N+1),evals-1,'--k',
                   label='signal')
    pylab.semilogy(range(1,N*N+1),numpy.ones(N*N),':k',
                   label='noise')

    pylab.xlim(0,N*N)
    
    pylab.legend(loc=0)
    pylab.xlabel('mode number')
    pylab.ylabel(r'$\langle a_n^2\rangle$',
                 fontsize=16)

    pylab.text(0.05,0.9,r'$\bar{n}_{gal} = %i/\rm{arcmin}$' % ngal,
               fontsize=16,
               transform=pylab.gca().transAxes)
    pylab.text(0.05,0.85,
               r'$\rm{field\ size\ =\ }%i \times %i\rm{\ pixels}$' \
                   % (N,N),
               fontsize=16,
               transform=pylab.gca().transAxes)
    pylab.text(0.05,0.79,
               r'$\rm{pixel\ size\ =\ }%.1f \times %.1f\rm{\ arcmin}$' \
                   % (dtheta,dtheta),
               fontsize=16,
               transform=pylab.gca().transAxes)

    pylab.title('K-L "Power Spectrum"')


def compute_likelihood( outfile = 'output/likelihood.dat',
                        N=32, dtheta=2.0,
                        nmodes = [300],
                        ngal=40,
                        sigma=0.3,
                        Om_range = numpy.linspace(0.2,0.4,10),
                        s8_range = numpy.linspace(0.75,0.95,10),
                        cosmo = None,
                        **kwargs):
    """
    compute the likelihood for all points in range
    """
    #----------------------------------------------------------------------
    # construct correlation matrix
    print "constructing fiducial basis"

    if cosmo is None:
        if 'Or' not in kwargs:
            kwargs['Or'] = 0
        cosmo = Cosmology(**kwargs)

    n = lambda z,z0=0.5: z**2 * numpy.exp(-(z/z0)**1.5)
    zlim = (0,3)
    xi = xi_plus(n,zlim,Nz=20,Nell=5E4,cosmo=cosmo)
    
    C = compute_correlation_matrix(xi,N,dtheta,
                                   ngal=ngal*dtheta**2,
                                   sigma = sigma,
                                   whiten = True)
    evals,evecs = compute_KL(C)

    #----------------------------------------------------------------------
    # create a fiducial realization of the data
    a_fid = numpy.random.normal(scale = numpy.sqrt(evals))

    #----------------------------------------------------------------------
    # compute likelihoods at each point
    OF = open(outfile,'w')
    OF.write('#Om s8 ncut log(L)\n')

    for i in range(len(Om_range)):
        Om = Om_range[i]
        for j in range(len(s8_range)):
            s8 = s8_range[j]
            print '(Om,s8) = (%.2g,%.2g)' % (Om,s8)
            cosmo_i = Cosmology(Om = Om,
                                Ol = 1.-Om,
                                Or = 0,
                                sigma8 = s8)
            xi = xi_plus(n,zlim,Nz=20,Nell=5E4,cosmo=cosmo_i)
            R =  compute_correlation_matrix(xi,N,dtheta,
                                            ngal=ngal*dtheta**2,
                                            sigma = sigma,
                                            whiten = True)
            for k in range(len(nmodes)):
                ncut = nmodes[k]
                evecs_n = evecs[:,:ncut]
                a_n = a_fid[:ncut]
                C_n = numpy.dot(evecs_n.T,numpy.dot(R,evecs_n))
                
                #model predicts <a>=0
                chi2 = numpy.dot(a_n,numpy.linalg.solve(C_n,a_n) )
                detC = numpy.linalg.det(C_n)

                X0 = -0.5 * ncut * numpy.log(2*numpy.pi)
                X1 = -0.5 * numpy.log( abs(detC) )
                X2 = -chi2/2 
                s = '%.2g  %.2g  %i  %.8g' % (Om,s8,ncut,X0+X1+X2)
                print s
                OF.write(s+'\n')
            ###
        ###
    OF.close()

if __name__ == '__main__':
    """
    plot_signal_to_noise(N=32, dtheta=5.0,
                         ngal=40,
                         sigma=0.3)
    pylab.show()
    exit()
    """

    compute_likelihood( outfile = 'output/likelihood.dat',
                        N=32, dtheta=5.0,
                        nmodes = [50,100,150,200,300,400,500,600,700,800],
                        ngal=40,
                        sigma=0.3,
                        Om_range = numpy.linspace(0.2,0.35,10),
                        s8_range = numpy.linspace(0.6,1.0,10) )
