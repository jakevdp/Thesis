"""
Compute likelihoods for a range of cosmological parameters
"""
import sys
import os

try:
    from itertools import product as iter_product
except ImportError: #not available in python version < 2.6
    from my_itertools import iter_product

import numpy as np

sys.path.append(os.path.abspath('../../'))
from shear_KL_source.cosmology import Cosmology
from shear_KL_source.zdist import zdist_from_file, zdist_parametrized

from tools import get_gamma_vector, shear_correlation_matrix, zdist
from estimate_uncertainty import bootstrap_catalog

def compute_likelihoods(catalog_file,
                        out_file,
                        cosmo_dict,
                        nmodes,
                        dtheta,
                        n_z,
                        sigma=0.3,
                        RAmin = None, DECmin = None,
                        NRA = None, NDEC = None,
                        remove_z_problem=True,
                        fiducial_cosmology = None,
                        **kwargs):
    """
    compute the likelihoods for a range of cosmologies

    Parameters
    ----------
    catalog_file : string or list of strings
        location of the COSMOS catalog(s) to use
    out_file : string
        location to save likelihood output
        Output will be a text file, with columns labeled
    cosmo_dict : Dictionary
        keys are arguments for cosmology object
        values are the corresponding range
    nmodes : 
        number of modes to use.  This should be less than NRA*NDEC
        alternatively, a list of integers can be supplied
    dtheta : float
        size of (square) pixels in arcmin
    sigma : float (default = 0.3)
        intrinsic ellipticity of galaxies
    fiducial_cosmology : Cosmology object
        the fiducial cosmology used for determination of KL vectors
        if unspecified, it will be initialized from remaining kwargs

    Other Parameters
    ----------------
    n_z : `shear_KL_source.zdist.zdist` object
        n_z(z) returns the galaxy distribution
            
    If the following are unspecified, they will be determined from the data
    RAmin/DECmin : float
        minimum of RA/DEC bins (degrees).
    NRA/NDEC : int
        number of RA/DEC bins
    """
    gamma, Ngal, noise, RArange, DECrange = get_gamma_vector(
        catalog_file, dtheta, RAmin, DECmin, NRA, NDEC, N_bootstraps=10,
        remove_z_problem=True)
    
    if fiducial_cosmology is None:
        fiducial_cosmology = Cosmology(**kwargs)

    # use results of bootstrap resampling to compute
    # the noise on observed shear
    gamma = gamma.reshape(gamma.size)
    Ngal = Ngal.reshape(Ngal.size)
    noise = noise.reshape(noise.size)

    i_sigma = np.where(Ngal > 1)
    sigma_estimate = np.sqrt(np.mean(noise[i_sigma] * Ngal[i_sigma]))
    print "average sigma: %.2g" % sigma_estimate

    izero = np.where(Ngal <= 1)
    noise[izero] = sigma_estimate ** 2
    N_m_onehalf = noise ** -0.5

    # noise = sigma^2 / Ngal
    # correlation matrix takes a constant sigma and a variable
    # ngal. So we'll encode the noise as an "effective Ngal"
    # using the user-defined sigma.
    Ngal_eff = sigma**2 / noise
    Ngal_eff[np.where(Ngal == 0)] = 0

    # construct fiducial correlation matrix
    print ">> fiducial correlation matrix"
    R_fid = shear_correlation_matrix(sigma, RArange, DECrange, Ngal_eff,
                                     n_z,
                                     whiten=True,
                                     cosmo=fiducial_cosmology)

    evals, evecs = np.linalg.eigh(R_fid)
    isort = np.argsort(evals)[::-1]
    evals = evals[isort]
    evecs = evecs[:,isort]

    #compute KL transform of data
    a_data = np.dot(evecs.T, N_m_onehalf * gamma)

    #iterate through all nmodes requested
    if not hasattr(nmodes, '__iter__'):
        nmodes = [nmodes]

    cosmo_keys = cosmo_dict.keys()
    cosmo_vals = [cosmo_dict[k] for k in cosmo_keys]
    cosmo_kwargs = fiducial_cosmology.get_dict()
    
    log2pi = np.log(2*np.pi)

    OF = open(out_file, 'w')
    OF.write('# fiducial cosmology: %s\n' % str(cosmo_kwargs))
    OF.write('# ncut ')
    OF.write(' '.join(cosmo_keys))
    OF.write(' chi2 log|det(C)| log(Likelihood)\n')

    for cosmo_tup in iter_product(*cosmo_vals):
        cosmo_kwargs.update(dict(zip(cosmo_keys,cosmo_tup)))

        #flat universe prior
        cosmo_kwargs['Ol'] = 1. - cosmo_kwargs['Om']
        
        print ">>", cosmo_keys, ['%.2g' % v for v in cosmo_tup]
        R = shear_correlation_matrix(sigma, RArange, DECrange, Ngal_eff,
                                     n_z,
                                     whiten=True,
                                     **cosmo_kwargs)
        cosmo_args = (len(cosmo_keys) * " %.6g") % cosmo_tup
        for ncut in nmodes:
            evecs_n = evecs[:,:ncut]
            a_n = a_data[:ncut]
            C_n = np.dot(evecs_n.T, np.dot(R, evecs_n))

            # compute chi2 = (a_n-<a_n>)^T C_n^-1 (a_n-<a_n>)
            # model predicts <a> = 0 so this simplifies:
            chi2_raw = np.dot(a_n.conj(), np.linalg.solve(C_n, a_n))

            #chi2_raw is complex because a_n is complex.  The imaginary
            # part of chi2 should be zero (within machine precision), because
            # C_n is Hermitian.  We'll skip checking that this is the case.
            chi2 = chi2_raw.real
            s, logdetC = np.linalg.slogdet(C_n)
            
            X0 = -0.5 * ncut * log2pi
            X1 = -0.5 * logdetC
            X2 = -0.5 * chi2
            print chi2, logdetC, X0, X1, X2
            OF.write("%i %s %.6g %.6g %.6g\n" % (ncut, cosmo_args, chi2,
                                                 logdetC, X0+X1+X2))
        ###
    ###
    OF.close()

if __name__ == '__main__':
    from params import BRIGHT_CAT_NPZ, FAINT_CAT_NPZ, \
         COMBINED_ZDIST_ZPROB0, BRIGHT_ZDIST_ZPROB0

    s8_range = np.linspace(0.2, 1.4, 13)
    s8_range += 0.5 * (s8_range[1] - s8_range[0])
    Om_range = np.linspace(0.05, 0.95, 10)
    
    if 0:
        n_z = zdist_from_file(BRIGHT_ZDIST_ZPROB0)
        
        compute_likelihoods(BRIGHT_CAT_NPZ,
                            out_file='likelihoods/test_bright_a.out',
                            cosmo_dict={'Om' : Om_range,
                                        'sigma8' : s8_range},
                            nmodes=(100,200,400),
                            sigma=0.39,
                            n_z = n_z,
                            dtheta = 4.0,
                            Or=0.0)
    
    else:
        n_z = zdist_from_file(COMBINED_ZDIST_ZPROB0)
        
        compute_likelihoods((BRIGHT_CAT_NPZ,FAINT_CAT_NPZ),
                            out_file='likelihoods/test_both.out',
                            cosmo_dict={'Om' : Om_range,
                                        'sigma8' : s8_range},
                            nmodes=(100,200,400),
                            sigma=0.39,
                            n_z = n_z,
                            dtheta=4.0,
                            Or=0.0)
    
