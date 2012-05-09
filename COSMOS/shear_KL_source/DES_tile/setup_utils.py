
import os
import sys
import numpy

from ..shear_correlation import xi_plus
from ..shear_KL import compute_correlation_matrix, compute_KL
from .. import params
from .tools import get_basis_filename

def compute_eigenbasis( Npix = None,
                        field_width = None ):
    """
    compute and save the shear KL eigenbasis
    All the parameters are given in params
    """
    outfile = get_basis_filename(Npix,field_width)
    
    if Npix is None:
        Npix = 2*params.Npix
    if field_width is None:
        field_width = 2*params.field_width

    print 'saving to:'
    print ' ',outfile
    
    dpix = field_width/Npix
    
    xi = xi_plus(params.n_z,
                 params.zlim,
                 Nz=20,Or=0)

    C = compute_correlation_matrix(xi,
                                   Npix = Npix,
                                   dpix = dpix,
                                   ngal = params.ngal,
                                   sigma = params.sigma,
                                   whiten=True)

    evals,evecs = compute_KL(C)

    numpy.savez(outfile,
                evals=evals,
                evecs=evecs,
                dtheta=dpix,
                ngal=params.ngal,
                sigma=params.sigma)



def create_cfg_file(cfg_file = 'DES_tile.cfg',
                    alphas = (0.15,),
                    NMODES = (500,),
                    use_noise = (True,),
                    use_mask = (False,),
                    noise_only = False,
                    weight_by_noise = False,
                    compute_shear_noise = False,
                    compute_Map = True,
                    compute_Map_noise = False,
                    RAlim = None,
                    DEClim = None,
                    append_file = False):
    """
    create a cfg file based on params in order to compute tiled DES
    quantities using condor
    """
    if RAlim is None:
        RAlim = params.RAlim
    RAmin,RAmax = RAlim
    dRA = params.field_width * 1./60

    if DEClim is None: DEClim = params.DEClim
    DECmin, DECmax = DEClim
    dDEC = params.field_width * 1./60

    print "writing to %s" % cfg_file
    if append_file:
        OF = open(cfg_file,'a')
    else:
        OF = open(cfg_file,'w')

    pyexec = params.pyexec

    basis = get_basis_filename()
    outdir = params.shear_recons_dir
    sheardir = params.shear_in_dir

    if not append_file:
        OF.write('Executable = /astro/apps/pkg/python64/bin/python\n')
        OF.write('Notification = never\n')
        OF.write('getenv = true\n')
        OF.write('Universe = vanilla\n')
    #---
    OF.write('Initialdir = %s\n' % params.initialdir)
    OF.write('Log = %s\n\n' % os.path.join(params.condorlog,'log.txt'))

    #create directories
    for alpha in alphas:
        for mask in use_mask:
            if mask: mask='y'
            else: mask='n'
            for nmodes in NMODES:
                for noise in use_noise:
                    if noise: noise='y'
                    else: noise='n'
                    Dir = os.path.join(outdir,'%i_%s_a%.2f%s'\
                                       %(nmodes,mask,alpha,noise))
                    if noise_only:
                        Dir += '_no'
                    if not os.path.exists(Dir):
                        os.system('mkdir %s' % Dir)

    NQueues = 0

    for RA in numpy.arange(RAmin,RAmax-dRA,dRA):
        for DEC in numpy.arange(DECmin,DECmax-dDEC,dDEC):
            for alpha in alphas:
                for mask in use_mask:
                    #--------------------
                    if mask:
                        mask = 'y'
                        maskdir = params.mask_outdir
                    else:
                        mask = 'n'
                        maskdir = 'none'
                    #--------------------
                    for noise in use_noise:
                        #--------------------
                        if noise:
                            noise = 'y'
                            sigma = params.sigma
                        else:
                            noise = 'n'
                            sigma = 0
                        #--------------------
                        for nmodes in NMODES:
                            #--------------------
                            Dir = os.path.join(outdir,'%i_%s_a%.2f%s'\
                                               % (nmodes,mask,alpha,noise))
                            ID = '%.1f_%.1f_%i_%s_a%.2f%s'\
                                 % (RA,DEC,nmodes,mask,alpha,noise)
                            if noise_only:
                                Dir += '_no'
                                ID += '_no'
                            #--------------------

                        
                            OF.write('Output = %s\n' \
                                     % os.path.join(params.condorlog,
                                                    'run_%s.out'%ID) )
                            OF.write('Error = %s\n' \
                                     % os.path.join(params.condorlog,
                                                    'run_%s.err'%ID) )
                            
                            OF.write('Arguments = %s ' % pyexec)
                            OF.write('-R %.1f ' % RA)
                            OF.write('-D %.1f ' % DEC)
                            OF.write('-b %s ' % basis)
                            OF.write('-s %s ' % sheardir)
                            OF.write('-m %s ' % maskdir)
                            OF.write('-o %s ' % Dir)
                            OF.write('-n %i ' % nmodes)
                            OF.write('-a %.2g ' % alpha)
                            OF.write('-r %i ' % params.rseed)
                            OF.write('-S %.2g ' % sigma)
                            OF.write('-V %.2g ' % params.rmax)
                            OF.write('-X %.2g ' % params.xc)
                            OF.write('-B %.2g ' % params.RAmin)
                            OF.write('-C %.2g ' % params.DECmin)
                            OF.write('-E %i ' % params.NRA)
                            OF.write('-F %i ' % params.NDEC)

                            opts = ''
                            if noise_only: opts += 'x'
                            if weight_by_noise: opts+='w'
                            if compute_shear_noise: opts+='c'
                            if compute_Map: opts += 'M'
                            if compute_Map_noise: opts += 'N'
                            
                            if opts:
                                OF.write('-%s '%opts)
                            OF.write('\nQueue\n\n')
                            NQueues += 1

    print "%i Queues" % NQueues

    OF.close()
                
def create_mask_tiles( outdir,
                       fmask = None):
    """
    create mask tiles across the field
    """
    from .tools import Hikage_mask

    if fmask is None:
        fmask = params.fmask

    print "creating mask tiles: fmask = %.2f" % fmask
    
    field_width = params.field_width* 1./60
    RA_range = numpy.arange(params.RAmin,
                            params.RAmax,
                            field_width)
    DEC_range = numpy.arange(params.DECmin,
                             params.DECmax,
                             field_width)

    NRA = len(RA_range) * params.Npix
    NDEC = len(DEC_range) * params.Npix

    mask = Hikage_mask(params.NRA,
                       params.NDEC,
                       params.dtheta,
                       fmask=fmask)

    for i in range(len(RA_range)):
        RA_min = RA_range[i]
        for j in range(len(DEC_range)):
            DEC_min = DEC_range[j]
            outfile = os.path.join(outdir,
                                   'mask_%.1f_%.1f.npz' % (RA_min,
                                                           DEC_min) )
            numpy.savez(outfile,mask = mask[i*params.Npix:(i+1)*params.Npix,
                                            j*params.Npix:(j+1)*params.Npix])
        #end DEC loop
    #end RA loop

def parse_DES_shear( outdir = None,
                     field_width = None, #arcmin
                     Npix = None, #pixels per side
                     RAlim = None,
                     DEClim = None,
                     fits_dir = None,
                     fits_file = None):
    """
    parse the shear from the DES mock catalog.  This will split it into
    tiles using the parameters specified in ..params.
    """
    import pyfits
    import gc

    if outdir is None:
        outdir = params.shear_in_dir
    if field_width is None:
        field_width = params.field_width
    if Npix is None:
        Npix = params.Npix
    if RAlim is None:
        RAlim = params.RAlim
    if DEClim is None:
        DEClim = params.DEClim
    if fits_dir is None:
        fits_dir = params.fits_dir
    if fits_file is None:
        fits_file = params.fits_file

    
    field_width = field_width* 1./60
    RA_range = numpy.arange(RAlim[0],RAlim[1],field_width)
    DEC_range = numpy.arange(DEClim[0],DEClim[1],field_width)

    print "RA: %i values in range (%.1f,%.1f)" % (len(RA_range),
                                                  RAlim[0],
                                                  RAlim[1])

    print "DEC: %i values in range (%.1f,%.1f)" % (len(DEC_range),
                                                   DEClim[0],
                                                   DEClim[1])
    
    gamma  = numpy.zeros( (Npix,Npix), dtype=complex )
    kappa = numpy.zeros( (Npix,Npix), dtype=float )
    epsilon  = numpy.zeros( (Npix,Npix), dtype=complex )
    Ngal = numpy.zeros( (Npix,Npix), dtype=int )

    hdulists = [pyfits.open(os.path.join(fits_dir,
                                         fits_file%i)) for i in range(10)]

    dPix = field_width * 1./Npix

    for RA_min in RA_range:
        RA_max = RA_min + field_width
        
        #compute which hdu files are needed for this RA range
        i_min = int( numpy.floor(0.5*(RA_min-10)) )
        i_max = int( numpy.ceil(0.5*(RA_min+field_width-10)) )
        
        for DEC_min in DEC_range:
            DEC_max = DEC_min+field_width
            
            gamma *= 0
            kappa *= 0
            epsilon *= 0
            Ngal *= 0
            outfile = os.path.join(outdir,
                                   'shear_out_%.1f_%.1f.npz' % (RA_min,
                                                                DEC_min) )
            print "parsing RA=%.1f, DEC=%.1f" % (RA_min,DEC_min)
            #loop through hdu files
            for hdulist in hdulists[i_min:i_max]:
                RA = hdulist[1].data.field('RA')
                DEC = hdulist[1].data.field('DEC')
                
                i = numpy.where( (RA>=RA_min) & (RA<RA_max) \
                                 & (DEC>=DEC_min) & (DEC<DEC_max) )
                N = len(i[0])
                if N==0: continue
                
                RA = RA[i]
                DEC = DEC[i]
                GAMMA1  = hdulist[1].data.field('GAMMA1')[i]
                GAMMA2  = hdulist[1].data.field('GAMMA2')[i]
                KAPPA   = hdulist[1].data.field('KAPPA')[i]
                EPSILON = numpy.asarray( hdulist[1].data.field('EPSILON')[i] )

                RA -= RA_min
                RA /= dPix
                i_x = numpy.array(RA,dtype=int)
                i_x[numpy.where(i_x==Npix)] = Npix-1

                DEC -= DEC_min
                DEC /= dPix
                i_y = numpy.array(DEC,dtype=int)
                i_y[numpy.where(i_y==Npix)] = Npix-1

                for j in range(len(i_x)):
                    ind = (i_x[j],i_y[j])
                    Ngal[ind] += 1
                    gamma[ind] += -GAMMA1[j] + 1j*GAMMA2[j]
                    epsilon[ind] += -EPSILON[j,0] + 1j*EPSILON[j,1]
                    kappa[ind] += KAPPA[j]
                #end i_x loop
                gc.collect() #garbage collection - issue with pyfits
            #end hdulist loop
            i = numpy.where(Ngal==0)
            Ngal[i] = 1
            gamma /= Ngal
            epsilon /= Ngal
            kappa /= Ngal
            Ngal[i] = 0

            numpy.savez(outfile,
                        RA = numpy.arange(RA_min,RA_max,dPix),
                        DEC = numpy.arange(DEC_min,DEC_max,dPix),
                        gamma = gamma,
                        kappa = kappa,
                        epsilon = epsilon,
                        Ngal = Ngal)

        #end DEC loop
    #end RA loop
