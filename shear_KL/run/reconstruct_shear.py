"""
script to reconstruct masked shear
"""

import os
import sys
import numpy
from time import time
import optparse

DIR = os.path.dirname( os.path.abspath(__file__) )
add_path = lambda x: sys.path.append(os.path.abspath(os.path.join(DIR,x)))

add_path('../')
from shear_KL_source.DES_KL.reconstruct import reconstruct_shear_and_Map
from shear_KL_source.DES_tile.tools import get_random_field


def main():
    #options with arguments
    parser = optparse.OptionParser()
    parser.add_option("-R", "--RAmin", dest="RAmin",type='float',
                      help="specify minimum RA value",
                      metavar="RAMIN")
    parser.add_option("-D", "--DECmin", dest="DECmin",type='float',
                      help="specify minimum DEC value",
                      metavar="DECMIN")
    parser.add_option("-b", "--basis_file", dest="basis_file",type='str',
                      help="specify KL basis file",
                      metavar="BASISFILE")
    parser.add_option("-s", "--shear_dir", dest="shear_dir",type='str',
                      help="specify shear directory",
                      metavar="SHEARDIR")
    parser.add_option("-m", "--mask_dir", dest="mask_dir",type='str',
                      help="specify mask directory (or 'none')",
                      metavar="MASKDIR",
                      default='none')
    parser.add_option("-o", "--out_dir", dest="out_dir",type='str',
                      help="specify output directory",
                      metavar="OUTDIR")
    parser.add_option("-n", "--nmodes", dest="nmodes",type='int',
                      help="specify number of modes for reconstruction",
                      metavar="NMODES")
    parser.add_option("-a", "--alpha", dest="alpha",type='float',
                      help="specify Wiener filter level alpha",
                      metavar="ALPHA")
    parser.add_option("-r", "--rseed", dest="rseed",type='int',
                      help="specify random seed for noise generation",
                      metavar="RSEED",
                      default = 0)
    parser.add_option("-S", "--sigma", dest="sigma",type='float',
                      help="specify sigma: shape noise level",
                      metavar="SIGMA",
                      default = None)
    parser.add_option("-V", "--rmax", dest="rmax",type='float',
                      help="specify rmax for aperture mass (arcmin)",
                      metavar="RMAX",
                      default = None)
    parser.add_option("-X", "--xc", dest="xc",type='float',
                      help="specify xc for aperture mass (unitless)",
                      metavar="XC",
                      default = None)
    parser.add_option("-B", "--RAmin_field", dest="RAmin_field",type='float',
                      help="specify lower limit of RA for entire field (deg)",
                      metavar="RAMIN_FIELD",
                      default = None)
    parser.add_option("-C", "--DECmin_field", dest="DECmin_field",type='float',
                      help="specify lower limit of DEC for entire field (deg)",
                      metavar="DECMIN_FIELD",
                      default = None)
    parser.add_option("-E", "--NRA_field", dest="NRA_field",type='float',
                      help="specify number of RA pixels for entire field",
                      metavar="NRA_FIELD",
                      default = None)
    parser.add_option("-F", "--NDEC_field", dest="NDEC_field",type='float',
                      help="specify number of DEC pixels in entire field",
                      metavar="NDEC_FIELD",
                      default = None)
    
    #options without arguments
    parser.add_option("-w", "--wbn", dest="wbn",
                      action='store_true',default=False,
                      help="specify whether to weight by noise")
    parser.add_option("-x", "--no_signal", dest="no_signal",
                      action="store_true",default=False,
                      help="specify whether to use zero input signal")
    parser.add_option("-c", "--compute_shear_noise",dest="compute_shear_noise",
                      action='store_true',default=False,
                      help="specify whether to compute shear noise")
    parser.add_option("-M", "--compute_Map", dest="compute_Map",
                      action='store_true',default=False,
                      help="specify whether to compute Aperture mass")
    parser.add_option("-N", "--compute_Map_noise", dest="compute_Map_noise",
                      action='store_true',default=False,
                      help="specify whether to compute Aperture mass noise")
    
    (options, args) = parser.parse_args()
    
    X = numpy.load(options.basis_file)
    evecs = X['evecs']
    evals = X['evals']
    dtheta = X['dtheta']
    sigma = X['sigma']
    ngal_pix = X['ngal']

    if options.sigma is not None:
        sigma = options.sigma
    
    N = int( numpy.sqrt(len(evals)) )
    N2 = N/2
    
    rand_field = get_random_field(options.RAmin,N,
                                  options.DECmin,N,
                                  rseed = options.rseed,
                                  RAmin_field = options.RAmin_field,
                                  DECmin_field = options.DECmin_field,
                                  NRA_field = options.NRA_field,
                                  NDEC_field = options.NDEC_field,
                                  dtheta = dtheta)
    shear = numpy.zeros( (N,N),dtype=complex )
    mask = numpy.ones( (N,N),dtype=int )
    Ngal = numpy.zeros( (N,N),dtype=int )
    
    fw = N2*dtheta/60.
    
    for i in range(2):
        RA = options.RAmin+i*fw
        for j in range(2):
            DEC = options.DECmin+j*fw
            X = numpy.load(os.path.join(options.shear_dir,
                                        'shear_out_%.1f_%.1f.npz' % (RA,DEC)))
            if not options.no_signal:
                shear[i*N2:(i+1)*N2,
                      j*N2:(j+1)*N2] = X['gamma']
            
            Ngal[i*N2:(i+1)*N2,
                 j*N2:(j+1)*N2] = X['Ngal']

            X.close()

            if options.mask_dir != 'none':
                X = numpy.load(os.path.join(options.mask_dir,
                                            'mask_%.1f_%.1f.npz' % (RA,DEC)))
                mask[i*N2:(i+1)*N2,
                     j*N2:(j+1)*N2] = X['mask']
                X.close()
            ###
            
    print "reconstructing shear with %i/%i modes" % (options.nmodes,N*N)
    
    mask.resize(N*N)
    shear.resize(N*N)
    Ngal.resize(N*N)
    
    #create shear noise based on sigma and number of galaxies
    i = numpy.where(Ngal==0)
    Ngal[i] = ngal_pix
    mask[i]=0
    if sigma>0:
        noise = numpy.zeros(N*N, dtype=complex)
        noise += sigma/numpy.sqrt(Ngal)
        noise *= numpy.exp(1j*2*numpy.pi*rand_field.reshape(N*N))
    else:
        noise = 0
        
    #mask is contained within Ngal: zero where Ngal=0
    j = numpy.where(mask==0)
    Ngal[j]=0
    
    shear_observed = shear + noise
    
    shear_observed[j]=0
    
    t0 = time()
    
    ret = reconstruct_shear_and_Map( shear_observed,
                                     Ngal,
                                     sigma,
                                     ngal_pix,
                                     evals,
                                     evecs,
                                     options.nmodes,
                                     options.alpha,
                                     dtheta = dtheta,
                                     rmax = options.rmax,
                                     xc = options.xc,
                                     full_output=False,
                                     weight_by_noise=options.wbn,
                                     compute_shear_noise = options.compute_shear_noise,
                                     compute_Map = options.compute_Map,
                                     compute_Map_noise = options.compute_Map_noise)
    
    shear_out = ret[0]
    shear_out.resize( (N,N) )
    i=1
    if options.compute_shear_noise:
        Nshear = ret[i].reshape((N,N))
        i+=1
    else:
        Nshear = None
    
    if options.compute_Map:
        Map = ret[i].reshape((N/2,N/2))
        i+=1
    else:
        Map = None
    
    if options.compute_Map_noise:
        NMap = ret[i].reshape((N/2,N/2))
    else:
        NMap = None
        
    print "shear reconstructed in %.2g sec" % (time()-t0)
    t0 = time()
    
    outfile = os.path.join(options.out_dir,
                           'reconstruct_%.1f_%.1f.npz' % (options.RAmin,
                                                          options.DECmin))
    
    print "saving to",outfile
    
    if sigma==0:
        noise = None
    else:
        noise.resize( (N,N) )

    numpy.savez( outfile,
                 shear = shear_out,
                 Nshear = Nshear,
                 Map = Map,
                 NMap = NMap,
                 alpha = options.alpha,
                 sigma = sigma,
                 noise = noise)
                              
                              
if __name__ == '__main__':
    main()
