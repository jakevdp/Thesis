import sys
import os
import numpy

from ..Map_peaks import *
from ..DES_tile.tools import *
from .. import params
from ..tools import gamma_to_kappa

def tiled_Map(gamma,dtheta,rmax,
              chunk_size=5,
              normed=False):
    """
    compute aperture mass map for gamma.
    Break this into 64x64 pixel chunks, with a border of 2*rmax
    """
    border_size = int(numpy.ceil(1.8*rmax/dtheta))

    Map_E = numpy.zeros( gamma.shape )
    Map_B = numpy.zeros( gamma.shape )

    Nx,Ny = gamma.shape

    pos_x = dtheta*numpy.arange(Nx)
    pos_y = dtheta*numpy.arange(Ny)

    pos = pos_x[:,None] + 1j*pos_y[None,:]

    Nchunks_x = int(numpy.ceil(Nx*1./chunk_size))
    Nchunks_y = int(numpy.ceil(Ny*1./chunk_size))

    print Nchunks_x, Nchunks_y

    for ix in range(Nchunks_x):
        if ix%50==0: print "%i/%i" % (ix,Nchunks_x)
        ix = ix*chunk_size
        
        ixmin = max(0,ix-border_size)
        ixmax = min(Nx,ix+border_size)

        for iy in range(Nchunks_y):
            iy = iy*chunk_size
            
            iymin = max(0,iy-border_size)
            iymax = min(Ny,iy+border_size)

            if normed:
                E,B = Map_map_normed( gamma[ixmin:ixmax,iymin:iymax],
                                      pos[ixmin:ixmax,iymin:iymax],
                                      pos[ix:ix+chunk_size,iy:iy+chunk_size],
                                      rmax = rmax )
            else:
                E,B = Map_map( gamma[ixmin:ixmax,iymin:iymax],
                               pos[ixmin:ixmax,iymin:iymax],
                               pos[ix:ix+chunk_size,iy:iy+chunk_size],
                               rmax = rmax )
            Map_E[ix:ix+chunk_size,iy:iy+chunk_size] = E
            Map_B[ix:ix+chunk_size,iy:iy+chunk_size] = B

    return Map_E,Map_B



def tiled_Map_noise(gamma,noise,dtheta,rmax,
                    chunk_size=5,
                    normed=False):
    """
    compute aperture mass map for gamma.
    Break this into 64x64 pixel chunks, with a border of 2*rmax

    noise is an array of noise corresponding to each value of gamma.
    """
    assert gamma.shape == noise.shape
    
    border_size = int(numpy.ceil(1.8*rmax/dtheta))

    Map_E = numpy.zeros( gamma.shape )
    Map_B = numpy.zeros( gamma.shape )
    Map_noise = numpy.zeros( gamma.shape )

    Nx,Ny = gamma.shape

    pos_x = dtheta*numpy.arange(Nx)
    pos_y = dtheta*numpy.arange(Ny)

    pos = pos_x[:,None] + 1j*pos_y[None,:]

    Nchunks_x = int(numpy.ceil(Nx*1./chunk_size))
    Nchunks_y = int(numpy.ceil(Ny*1./chunk_size))

    print Nchunks_x, Nchunks_y

    for ix in range(Nchunks_x):
        if ix%50==0: print "%i/%i" % (ix,Nchunks_x)
        ix = ix*chunk_size
        
        ixmin = max(0,ix-border_size)
        ixmax = min(Nx,ix+border_size)

        for iy in range(Nchunks_y):
            iy = iy*chunk_size
            
            iymin = max(0,iy-border_size)
            iymax = min(Ny,iy+border_size)

            if normed:
                E,B,N = Map_map_noise_normed( gamma[ixmin:ixmax,iymin:iymax],
                                              noise[ixmin:ixmax,iymin:iymax],
                                              pos[ixmin:ixmax,iymin:iymax],
                                              pos[ix:ix+chunk_size,
                                                  iy:iy+chunk_size],
                                              rmax = rmax )
            else:
                E,B,N = Map_map_noise( gamma[ixmin:ixmax,iymin:iymax],
                                       noise[ixmin:ixmax,iymin:iymax],
                                       pos[ixmin:ixmax,iymin:iymax],
                                       pos[ix:ix+chunk_size,iy:iy+chunk_size],
                                       rmax = rmax )
            Map_E[ix:ix+chunk_size,iy:iy+chunk_size] = E
            Map_B[ix:ix+chunk_size,iy:iy+chunk_size] = B
            Map_noise[ix:ix+chunk_size,iy:iy+chunk_size] = N

    return Map_E,Map_B,Map_noise


def compute_Map_input(add_signal = True,
                      add_noise = True,
                      usemask = 'y',
                      normed = False):
    """
    compute the aperture mass map for the input shear
    """
    RAlim = params.RAlim #(10,12)
    DEClim = params.DEClim #(35,37)
    dtheta = params.dtheta
    sigma = params.sigma
    r = params.rmax
    
    gamma,Ngal = get_from_parsed( 'gamma','Ngal',
                                  RAlim=RAlim,
                                  DEClim=DEClim )
    
    if normed and (usemask=='n'):
        return

    if usemask=='y':
        mask = get_mask(RAlim=RAlim,
                        DEClim=DEClim,
                        maskdir=params.mask_outdir)
    else:
        mask = get_mask(blank=True,
                        RAlim=RAlim,
                        DEClim=DEClim)
            
            
    i = numpy.where(Ngal==0)
    Ngal[i]=1
    
    noise = numpy.zeros(gamma.shape,dtype=complex)
    noise += sigma/numpy.sqrt(Ngal)
    noise *= numpy.exp( 2j*numpy.pi*get_random_field(params.RAlim[0],
                                                     params.NRA,
                                                     params.DEClim[0],
                                                     params.NDEC))
    epsilon = 0
    if add_signal: epsilon = epsilon+gamma
    if add_noise:  epsilon = epsilon+noise
    
    epsilon_mask = epsilon * mask
    
    outdir = params.shear_recons_dir
    
    Map_E,Map_B,Map_noise = tiled_Map_noise(epsilon_mask,
                                            sigma/numpy.sqrt(Ngal),
                                            dtheta,r,
                                            normed=normed)

    if add_noise:
        if add_signal: tag = 'noisy'
        else:          tag = 'noiseonly'
    else:
        if add_signal: tag = 'perfect'
        else:          tag = 'zero'

    tag = 'Map_%s_%s' % (tag,usemask)

    if normed: tag += '_normed'
    
    tag += '.npz'
    
    outfile = os.path.join(outdir,tag)
    
    print "saving %s" % outfile
    numpy.savez(outfile,Map_E=Map_E,Map_B=Map_B,Map_noise=Map_noise)




    
def compute_Map_recons(F,
                       use_tbt = True,
                       include_noise = True):
    """
    compute the aperture mass map from the reconstructed shear

    use_tbt : use value computed tile-by-tile,
              rather than re-computing aperture mass globally
    """
    RAlim = params.RAlim #(10,12)
    DEClim = params.DEClim #(35,37)
    dtheta = params.dtheta
    
    r = params.rmax

    dir = params.shear_recons_dir

    if F not in os.listdir(dir):
        raise ValueError, "%s not in %s" % (F,dir)

    outfile = os.path.join(dir,'Map_%s.npz'%F)

    print "saving to", outfile
        
    if use_tbt:
        Map,Map_N = get_Map_reconstructed(F,return_noise=True)
        numpy.savez(outfile,
                    Map_E = Map.real,
                    Map_B = Map.imag,
                    Map_noise = numpy.sqrt(0.5*Map_N) )

    elif include_noise:
        gamma,noise = get_gamma_reconstructed(F,return_noise=True)
        Map_E,Map_B,Map_N = tiled_Map_noise(gamma,noise,dtheta,r)
        numpy.savez(outfile,Map_E=Map_E,Map_B=Map_B,Map_noise=Map_N)
        
    else:
        gamma = get_gamma_reconstructed(F)
        Map_E,Map_B = tiled_Map(gamma,dtheta,r)
        numpy.savez(outfile,Map_E=Map_E,Map_B=Map_B)
                       
            
def compute_kappa_input_true(filename = 'kappa_true.npz'):
    filename = os.path.join(params.shear_recons_dir,filename)
    
    kappa = get_from_parsed('kappa')
    
    print "saving to %s" % filename
    numpy.savez(filename,
                kappa=kappa)
    
def compute_kappa_input_perfect(filename = 'kappa_perfect.npz'):
    filename = os.path.join(params.shear_recons_dir,filename)
    
    gamma = get_from_parsed('gamma')
    kappa = gamma_to_kappa(gamma,
                           params.dtheta)

    print "saving to %s" % filename
    numpy.savez(filename,
                kappa=kappa)

def compute_kappa_input_masked(filename = 'kappa_masked.npz',
                               sigma = None):
    filename = os.path.join(params.shear_recons_dir,filename)
    
    gamma,Ngal = get_from_parsed('gamma','Ngal')
    if sigma is None: sigma = params.sigma

    i = numpy.where(Ngal==0)
    Ngal[i]=1
    
    mask = get_mask(RAlim=params.RAlim,
                    DEClim=params.DEClim,
                    maskdir=params.mask_outdir)
            
    noise = numpy.zeros(gamma.shape,dtype=complex)
    noise += sigma/numpy.sqrt(Ngal)
    noise *= numpy.exp( 2j*numpy.pi*get_random_field(params.RAlim[0],
                                                     params.NRA,
                                                     params.DEClim[0],
                                                     params.NDEC))

    gamma_masked = mask * (gamma+noise)

    kappa = gamma_to_kappa(gamma_masked,
                           params.dtheta)
    
    print "saving to %s" % filename
    numpy.savez(filename,
                kappa=kappa)
    
def compute_kappa_input_noisy(filename = 'kappa_noisy.npz',
                              sigma = None):
    filename = os.path.join(params.shear_recons_dir,filename)
    
    gamma,Ngal = get_from_parsed('gamma','Ngal')
    if sigma is None: sigma = params.sigma

    i = numpy.where(Ngal==0)
    Ngal[i]=1
            
    noise = numpy.zeros(gamma.shape,dtype=complex)
    noise += sigma/numpy.sqrt(Ngal)
    noise *= numpy.exp( 2j*numpy.pi*get_random_field(params.RAlim[0],
                                                     params.NRA,
                                                     params.DEClim[0],
                                                     params.NDEC))

    kappa = gamma_to_kappa(gamma+noise,
                           params.dtheta)
    
    print "saving to %s" % filename
    numpy.savez(filename,
                kappa=kappa)
    
def compute_kappa_input(noisy=True,
                        perfect=True,
                        true=True,
                        masked=True):
    """
    compute kappa from input shear in one or all of three ways.
    """
    if true:
        compute_kappa_input_true()
    if noisy:
        compute_kappa_input_perfect()
    if perfect:
        compute_kappa_input_noisy()
    if masked:
        compute_kappa_input_masked()


def compute_kappa_recons(F):
    """
    compute kappa map from reconstructed shear
    """
    RAlim = params.RAlim #(10,12)
    DEClim = params.DEClim #(35,37)
    dtheta = params.dtheta
    dir = params.shear_recons_dir

    if F not in os.listdir(dir):
        raise ValueError, "%s not in %s" % (F,dir)

    
    outfile = os.path.join(dir,'kappa_%s.npz'%F)
    gamma = get_gamma_reconstructed(F)

    print "computing kappa for",outfile
        
    kappa = gamma_to_kappa(gamma,dtheta)
        
    numpy.savez(outfile,kappa=kappa)
        
