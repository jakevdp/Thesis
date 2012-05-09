import sys
import os

import numpy as np

path1 = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
path2 = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

sys.path.append(path1)
from read_shear_catalog import read_shear_catalog

sys.path.append(path2)
from shear_KL_source.shear_correlation import xi_plus
from shear_KL_source.shear_KL import compute_correlation_matrix, compute_correlation_matrix_nonsquare, compute_KL

def bootstrap_resample(gamma, RA, DEC,
                       RAmin, dRA, nbinsRA,
                       DECmin, dDEC, nbinsDEC,
                       N_bootstraps,
                       rseed=1):
    """
    Bootstrap resample a shear catalog
    
    Parameters
    ----------
    gamma : array-like, complex, size = Ngal
        The individual shear measurements
    RA, DEC : array-like, float, size = Ngal
        The positions of the individual shear measurements
    RAmin, dRA, nbinsRA : floating point
        These define the right-ascention bounds of the pixel regions.
        The i^th pixel in RA has RAmin + dRA*i <= RA < RAmin + dRA*(i+1)
    DECmin, dDEC, nbinsDEC : floating point
        These define the declination bounds of the pixel regions.
        The i^th pixel in DEC has DECmin + dDEC*i <= DEC < DECmin + dDEC*(i+1)
    N_bootstraps : integer
        Number of bootstrap resamples to perform

    Returns
    -------
    gamma_mean : np.ndarray, complex, shape = (nbinsRA, nbinsDEC)
        the mean value of shear within each pixel
    N_per_bin : np.ndarray, integer, shape = (nbinsRA, nbinsDEC)
        number of galaxies in each bin
    gamma_noise : np.ndarray, float, shape = (nbinsRA, nbinsDEC)
        the squared error of shear within each pixel
    sigma_estimate : float
        global estimate of intrinsic ellipticity for the dataset
    """
    np.random.seed(rseed)
    
    Ngal = len(gamma)
    assert len(RA) == Ngal
    assert len(DEC) == Ngal

    #find RA indices and DEC indices
    iRA = np.floor((RA - RAmin) / dRA).astype(int)
    iDEC = np.floor((DEC - DECmin) / dDEC).astype(int)

    #compute the mean and number of galaxies in each shear bin
    gamma_mean = np.zeros((nbinsRA, nbinsDEC), dtype=complex)
    Ngal_bins = np.zeros((nbinsRA, nbinsDEC), dtype=int)

    for i in range(Ngal):
        gamma_mean[iRA[i], iDEC[i]] += gamma[i]
        Ngal_bins[iRA[i], iDEC[i]] += 1

    i_zero = np.where(Ngal_bins == 0)
    Ngal_bins[i_zero] = 1
    gamma_mean /= Ngal_bins
    Ngal_bins[i_zero] = 0

    #create a hash based on RA,DEC position to sort the galaxies into
    # a linear array, sorted by bins.
    power = 10 ** -(np.ceil(np.log10(nbinsDEC)))
    hash_list = iRA + power * iDEC
    i_sort = np.argsort(hash_list)
    #i_sort = np.arange(len(hash_list))
    gamma_list = gamma[i_sort]

    #create a list of indices telling where the shear values for each
    # bin start and end.
    # for the bin (i,j), the associated shear values are in
    # gamma_list[i * nbinsDEC + j, i * nbinsDEC + j + 1]
    indices = np.concatenate([[0], Ngal_bins.reshape(nbinsRA*nbinsDEC).cumsum()])
    
    #now compute the mean and standard deviation in a bootstrap resampling
    gamma_sum = np.zeros(nbinsRA*nbinsDEC, dtype=complex)
    gamma2_sum = np.zeros(nbinsRA*nbinsDEC, dtype=float)

    print "performing %i bootstrap resamplings of the data" % N_bootstraps
    for i in range(N_bootstraps):
        if (i+1)%100 == 0:
            print " > %i" % (i+1)
        for j in range(nbinsRA*nbinsDEC):
            i1 = indices[j]
            i2 = indices[j+1]
            if i1==i2: continue
            
            ind = np.random.randint(i1, i2, i2-i1)
            g = np.mean(gamma_list[ind])
            
            gamma_sum[j] += g
            gamma2_sum[j] += abs(g)**2

    gamma_sum /= N_bootstraps
    gamma2_sum /= N_bootstraps

    d2gamma = gamma2_sum - abs(gamma_sum**2)

    d2gamma.resize((nbinsRA, nbinsDEC))

    sigma2 = Ngal_bins * d2gamma
    sigma2 = sigma2[np.where(Ngal_bins > 2)]
    sigma_estimate = np.sqrt(np.mean(sigma2))

    return gamma_mean, Ngal_bins, d2gamma, sigma_estimate


def get_gamma_vector(catalog_file, dtheta,
                     RAmin = None, DECmin = None,
                     NRA = None, NDEC = None,
                     remove_z_problem = True,
                     N_bootstraps=0):
    """
    return the mask vector for a 2D shear field.

    Parameters
    ----------
    catalog_file : file of COSMOS shear catalog or list of files
                
    dtheta : pixel size in arcmin

    Other Parameters
    ----------------
    If these are unspecified, they will be determined from the data
    RAmin : minimum of RA bins (degrees).
    NRA : number of RA bins
    DECmin : minimum of DEC bins (degrees)
    NDEC : number of DEC bins
    N_bootstraps : number of bootstraps to perform.
        If zero, don't return noise

    Returns
    -------
    (gamma, Ngal, RArange, DECrange)
    gamma : array[complex], shape = (NRA, NDEC)
        average shear within each bin
    Ngal : array[integer], shape = (NRA, NDEC)
        number of galaxies in each bin
    RArange : array[float], shape = (NRA + 1,)
        edges of bins in RA
    DECrange : array[float], shape = (NDEC + 1,)
        edges of bins in DEC
    """
    RA, DEC, gamma1, gamma2 = \
        read_shear_catalog(catalog_file,
                           ('Ra', 'Dec',
                            'e1iso_rot4_gr_snCal',
                            'e2iso_rot4_gr_snCal'),
                           None,
                           remove_z_problem)

    if RAmin is None:
        RAmin = RA.min()
    if DECmin is None:
        DECmin = DEC.min()

    if NRA is None:
        NRA = int(np.ceil((RA.max() - RAmin + 1E-8) * 60. / dtheta))
    if NDEC is None:
        NDEC = int(np.ceil((DEC.max() - DECmin + 1E-8) * 60. / dtheta))

    RArange = RAmin + (dtheta / 60.) * np.arange(NRA + 1)
    DECrange = DECmin + (dtheta / 60.) * np.arange(NDEC + 1)

    gamma = gamma1 - 1j*gamma2

    gamma_mean, Ngal_bins, d2gamma, sigma_estimate = bootstrap_resample(
        gamma, RA, DEC,
        RAmin, dtheta/60., NRA,
        DECmin, dtheta/60., NDEC, N_bootstraps)

    if N_bootstraps > 0:
        return gamma_mean, Ngal_bins, d2gamma, RArange, DECrange
    else:
        return gamma_mean, Ngal_bins, RArange, DECrange


def shear_mask_vector(catalog_file, dtheta,
                      RAmin = None, DECmin = None,
                      NRA = None, NDEC = None):
    """
    return the mask vector for a 2D shear field.

    Parameters
    ----------
    catalog_file : file of COSMOS shear catalog
    dtheta : pixel size in arcmin

    Other Parameters
    ----------------
    If these are unspecified, they will be determined from the data
    RAmin : minimum of RA bins (degrees).
    NRA : number of RA bins
    DECmin : minimum of DEC bins (degrees)
    NDEC : number of DEC bins

    Returns
    -------
    (Ngal, RArange, DECrange)
    Ngal : array, shape = (NRA, NDEC)
        number of galaxies in each bin
    RArange : array, shape = (NRA + 1,)
        edges of bins in RA
    DECrange : array, shape = (NDEC + 1,)
        edges of bins in DEC
    """
    RA, DEC = read_shear_catalog(catalog_file, ('Ra', 'Dec'), None)

    if RAmin is None:
        RAmin = RA.min()
    if DECmin is None:
        DECmin = DEC.min()

    if NRA is None:
        NRA = int(np.ceil((RA.max() - RAmin + 1E-8) * 60. / dtheta))
    if NDEC is None:
        NDEC = int(np.ceil((DEC.max() - DECmin + 1E-8) * 60. / dtheta))

    RArange = RAmin + (dtheta / 60.) * np.arange(NRA + 1)
    DECrange = DECmin + (dtheta / 60.) * np.arange(NDEC + 1)

    Ngal, xedges, yedges = np.histogram2d(RA, DEC, bins=(RArange, DECrange))

    return Ngal, xedges, yedges


def zdist(z, z0=0.5, a=2, b=1.5):
    return z ** a * np.exp(-(z * 1. / z0) ** b)


def shear_correlation_matrix(sigma, RArange, DECrange, Ngal,
                             n_z = zdist,
                             whiten = True,
                             cosmo=None, **kwargs):
    """
    Compute the correlation matrix based on the COSMOS geometry

    Parameters
    ----------
    sigma : float
        intrinsic ellipticity in each pixel
    RArange : float array, size = NRA + 1
        RA bin edges
    DECrange : float array, size = NDEC + 1
        DEC bin edges
    Ngal : integer array, shape = (NRA, NDEC)
        number of galaxies in each bin. This can be determined using
        the routine `shear_mask_vector`
    z0, nz_a, nz_b : float
        specify the galaxy redshift distribution:
            n(z) ~ z^a * exp[-(z/z0)^b]
    cosmo : cosmology object
        if unspecified, a Cosmology object will be initialized from kwargs

    Returns
    -------
    C : the theoretical covariance matrix of the COSMOS data
    """
    xi = xi_plus(n_z, Nz=20, cosmo=cosmo, **kwargs)

    dpix1 = 60. * (RArange[1] - RArange[0])
    dpix2 = 60. * (DECrange[1] - DECrange[0])
    
    Npix1 = len(RArange) - 1
    Npix2 = len(DECrange) - 1

    #compute correlation, no shot noise
    if (Npix1 == Npix2) and (dpix1 == dpix2):
        C = compute_correlation_matrix(xi, Npix=Npix1, dpix=dpix1,
                                       ngal=1, sigma=0, whiten=False)
    else:
        C = compute_correlation_matrix_nonsquare(xi, Npix1=Npix1, dpix1=dpix1,
                                                 Npix2=Npix2, dpix2=dpix2,
                                                 ngal=1, sigma=0,
                                                 whiten=False)

    Ngal = Ngal.reshape(Npix1 * Npix2)

    i_zero = np.where(Ngal==0)[0]

    Ngal[i_zero] = 1

    noise_squared_diag = sigma**2 / Ngal.reshape(Npix1*Npix2)

    Ngal[i_zero] = 0

    #add shot noise
    C.flat[::Npix1 * Npix2 + 1] += noise_squared_diag

    #zero-out correlation matrix where there is no data
    C[i_zero] = 0
    C[:,i_zero] = 0

    if whiten:
        noise_inv = noise_squared_diag**-0.5
        C *= noise_inv[:,None]
        C *= noise_inv

    return C


def plot_covariance_matrix():
    from params import BRIGHT_CAT_NPZ
    Ngal, RArange, DECrange = shear_mask_vector(BRIGHT_CAT_NPZ, 4.0)

    C = shear_correlation_matrix(0.3, RArange, DECrange, Ngal)

    pylab.imshow(np.log10(C).T,
                 origin='upper',
                 interpolation='nearest')
    pylab.colorbar()


def plot_gal_distribution():
    from params import BRIGHT_CAT_NPZ
    gamma, Ngal, RArange, DECrange = get_gamma_vector(BRIGHT_CAT_NPZ, 4.0)
    #Ngal, RArange, DECrange = shear_mask_vector(BRIGHT_CAT_NPZ, 4.0)

    pylab.imshow(Ngal.T,
                 origin='lower',
                 interpolation='nearest',
                 extent = (RArange[0], RArange[-1],
                           DECrange[0], DECrange[-1]))
    pylab.xlabel('RA (deg)')
    pylab.ylabel('DEC (deg)')
    pylab.colorbar().set_label('galaxies per pixel')

    print "NRA:", len(RArange) - 1
    print "NDEC:", len(DECrange) - 1
    print Ngal.shape

if __name__ == '__main__':
    import pylab
    #plot_covariance_matrix()
    plot_gal_distribution()
    pylab.show()
    
