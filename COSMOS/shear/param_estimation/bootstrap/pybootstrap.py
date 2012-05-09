"""
bootstrap
This contains a cythonized routine for performing bootstrap resampling
of COSMOS shear.
"""
import numpy as np

def bootstrap_resample(gamma, RA, DEC,
                       RAmin, dRA, nbinsRA,
                       DECmin, dDEC, nbinsDEC,
                       N_bootstraps):
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
    
    for i in range(N_bootstraps):
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
