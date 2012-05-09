import numpy
import pylab

import sys
import os
from .. import params

def get_basis_filename(Npix = None,
                       field_width = None):
    if Npix is None:
        Npix = 2*params.Npix
    if field_width is None:
        field_width = 2*params.field_width

    return os.path.join(params.scratch_dir,
                        'basis_%i_%i.npz' % (Npix,field_width) )
                        

def get_random_field(RAmin,NRA,
                     DECmin,NDEC,
                     rseed=None,
                     RAmin_field=None,
                     DECmin_field=None,
                     NRA_field=None,
                     NDEC_field=None,
                     dtheta=None):
    """
    get random field
    RAmin,DECmin in degrees
    NRA,NDEC = size of sub-field

    rseed: random seed for field generation.
           If None, obtain from params.rseed
    RAmin_field: RA minimum of the entire field. (in degrees)
           If None, obtain from params.RAmin
    DECmin_field: DEC minimum of the entire field. (in degrees)
           If None, obtain from params.DECmin
    NRA_field: number of RA pixels in entire field.
           If None, obtain from params.NRA
    NDEC_field: number of RA pixels in entire field.
           If None, obtain from params.NDEC
    dtheta: size of a pixel (in arcmin)
           If None, obtain from params.dtheta
    """
    if rseed is None: rseed = params.rseed
    if RAmin_field is None: RAmin_field = params.RAmin
    if DECmin_field is None: DECmin_field = params.DECmin
    if NRA_field is None: NRA_field = params.NRA
    if NDEC_field is None: NDEC_field = params.NDEC
    if dtheta is None: dtheta = params.dtheta

    numpy.random.seed(rseed)
    r = numpy.random.random( (NRA_field,NDEC_field) )

    dRA = dtheta/60.
    dDEC = dRA

    iRA = int(numpy.round((RAmin-RAmin_field)/dRA))
    iDEC = int(numpy.round((DECmin-DECmin_field)/dDEC))

    return r[iRA:iRA+NRA,
             iDEC:iDEC+NDEC]

def get_from_parsed(*keys,
                    **kwargs):
    """
    get different variables from the parsed (pixelized) DES mock files,
    as defined in params
    """

    if kwargs.get('shear_in_dir') is not None:
        filenames = os.path.join(kwargs['shear_in_dir'],
                                 'shear_out_%.1f_%.1f.npz')
    else:
        filenames = os.path.join(params.shear_in_dir,
                                 'shear_out_%.1f_%.1f.npz')

    if kwargs.get('RAlim') is None:
        RAlim = params.RAlim
    else:
        RAlim = kwargs['RAlim']
    if kwargs.get('DEClim') is None:
        DEClim = params.DEClim
    else:
        DEClim = kwargs['DEClim']
    
    Npix = params.Npix
    fw = params.field_width / 60.

    NRA = int( (RAlim[1]-RAlim[0]) / fw )
    NDEC = int( (DEClim[1]-DEClim[0]) / fw )

    RArange = RAlim[0] + fw * numpy.arange(NRA)
    DECrange = DEClim[0] + fw * numpy.arange(NDEC)

    output = None
    
    for RA in RArange:
        for DEC in DECrange:
            i = Npix*(RA-RAlim[0])/fw
            j = Npix*(DEC-DEClim[0])/fw
            filename = filenames % (RA,DEC)
            if not os.path.exists(filename):
                raise ValueError, "get_from_parsed: %s does not exist" % filename
            X = numpy.load(filename)
            if output is None:
                for key in keys:
                    if key not in X:
                        raise ValueError, \
                              "get_from_parsed : unrecognized key %s" % key
                    ###
                ###
                output = [numpy.zeros((NRA*Npix,NDEC*Npix),dtype=X[key].dtype)\
                          for key in keys]
            ###
            for k in range(len(keys)):
                output[k][i:i+Npix,j:j+Npix] = X[keys[k]]
            ###
            X.close()    
        ###
    ###
    if len(keys)==1:
        return output[0]
    else:
        return tuple(output)


def get_gamma_true(return_Ngal=False):
    if return_Ngal:
        return get_from_parsed('gamma','Ngal')
    else:
        return get_from_parsed('gamma')


def get_kappa_true():
    return get_from_parsed('kappa')

def get_Ngal():
    return get_from_parsed('Ngal')

def get_epsilon():
    return get_from_parsed('epsilon')

def show_kappa(kappa,clabel=r'\kappa',part='r',clim=None,logplot=True):
    if part.lower()=='r':
        kappa = kappa.real
    elif part.lower()=='i':
        kappa = kappa.imag
    elif part.lower() in ['a','n']:
        kappa = abs(kappa)
    else:
        raise ValueError, "show_kappa : unrecognized part %s" % part
    
    pylab.figure(figsize=(14,9))

    if logplot:
        kappa = numpy.log(1+kappa)
    
    pylab.imshow(kappa.T,
                 origin = 'lower',
                 interpolation = 'nearest',
                 extent = params.RAlim+params.DEClim)
    cb = pylab.colorbar()
    if logplot:
        cb.set_label(r'$\rm{log}(1+%s)$' % clabel,
                     fontsize=14)
    else:
        cb.set_label(r'$%s$' % clabel,
                     fontsize=14)

    if clim is not None:
        pylab.clim(clim)
    
    pylab.xlabel('RA (deg)')
    pylab.ylabel('DEC (deg)')

def get_gamma_reconstructed(dir,
                            RAlim = None,
                            DEClim = None,
                            return_noise = False):
    """
    get the KL-reconstructed shear
    """

    filenames = os.path.join( params.shear_recons_dir,
                              dir,
                              'reconstruct_%.1f_%.1f.npz' )

    if RAlim is None:
        RAlim = params.RAlim
    if DEClim is None:
        DEClim = params.DEClim
    Npix = params.Npix
    fw = params.field_width / 60.

    NRA = int( (RAlim[1]-RAlim[0]) / fw )
    NDEC = int( (DEClim[1]-DEClim[0]) / fw )
    
    gamma = numpy.zeros((NRA*Npix,NDEC*Npix),dtype=complex)
    if return_noise:
        noise = numpy.zeros((NRA*Npix,NDEC*Npix),dtype=float)

    RArange = RAlim[0] + fw * numpy.arange(NRA-1)
    DECrange = DEClim[0] + fw * numpy.arange(NDEC-1)

    #for each image, the central 1/4 of the area is kept.
    # i_in,i_out, etc. keep track of whether we're at the
    # edge, where we need to keep more than the central 1/4
    # in order to fill the field.

    i_out = 0
    j_out = 0

    for RA in RArange:
        i_in = Npix/2
        iw = Npix
        if RA == RArange[0]:
            i_out=0
            i_in=0
            iw = 3*Npix/2
        elif RA == RArange[-1]:
            iw = 3*Npix/2
        for DEC in DECrange:
            j_in = Npix/2
            jw = Npix
            if DEC == DECrange[0]:
                j_out = 0
                j_in=0
                jw = 3*Npix/2
            elif DEC == DECrange[-1]:
                jw = 3*Npix/2
            
            X = numpy.load(filenames % (RA,DEC))
            s = X['shear']
            if numpy.any(numpy.isnan(s)):
                print "nans in file at", RA,DEC
            gamma[i_out:i_out+iw,
                  j_out:j_out+jw] = s[i_in:i_in+iw,
                                      j_in:j_in+jw]
            if return_noise:
                n = X['Nshear'].reshape(s.shape) #XXX reshape is a bandaid for typo in reconstruct_shear
                noise[i_out:i_out+iw,
                      j_out:j_out+jw] = n[i_in:i_in+iw,
                                          j_in:j_in+jw]
            j_out += jw
            X.close()
        ###
        i_out += iw
    ###
    if return_noise:
        return gamma,noise
    else:
        return gamma

def get_Map_reconstructed(dir,
                          RAlim = None,
                          DEClim = None,
                          return_noise = False):
    """
    get the Map value computed within reconstruct_shear.py
    note that there will be a border of zeros if the full RAlim
    and DEClim are used: this is not caculated field-by-field
    """

    filenames = os.path.join( params.shear_recons_dir,
                              dir,
                              'reconstruct_%.1f_%.1f.npz' )

    if RAlim is None:
        RAlim = params.RAlim
    if DEClim is None:
        DEClim = params.DEClim
    Npix = params.Npix
    fw = params.field_width / 60.

    NRA = int( (RAlim[1]-RAlim[0]) / fw )
    NDEC = int( (DEClim[1]-DEClim[0]) / fw )
    
    Map = numpy.zeros((NRA*Npix,NDEC*Npix),dtype=complex)
    if return_noise:
        noise = numpy.zeros((NRA*Npix,NDEC*Npix),dtype=float)

    RArange = RAlim[0] + fw * numpy.arange(NRA-1)
    DECrange = DEClim[0] + fw * numpy.arange(NDEC-1)

    #for each image, the central 1/4 of the area is kept.
    # i_in,i_out, etc. keep track of whether we're at the
    # edge, where we need to keep more than the central 1/4
    # in order to fill the field.

    i_out = Npix/2
    iw = Npix
    jw = Npix

    for RA in RArange:
        j_out = Npix/2
        for DEC in DECrange:
            X = numpy.load(filenames % (RA,DEC))
            m = X['Map']
            try:
                numpy.isnan(m)
            except:
                raise ValueError("No field 'Map' in %s" 
                                 % (filenames %(RA,DEC)))
            if numpy.any(numpy.isnan(m)):
                print "nans in file at", RA,DEC
            Map[i_out:i_out+iw,
                j_out:j_out+jw] = m
            if return_noise:
                n = X['NMap']
                noise[i_out:i_out+iw,
                      j_out:j_out+jw] = n
            j_out += jw
            X.close()
        ###
        i_out += iw
    ###
    if return_noise:
        return Map,noise
    else:
        return Map

def get_mask(blank=False,
             RAlim=None,
             DEClim=None,
             maskdir = None):
    if maskdir is None:
        maskdir = params.mask_outdir
    else:
        maskdir = maskdir

    filenames = os.path.join(maskdir,'mask_%.1f_%.1f.npz')

    if RAlim is None:
        RAlim = params.RAlim
    if DEClim is None:
        DEClim = params.DEClim
    Npix = params.Npix
    fw = params.field_width / 60.

    NRA = int( (RAlim[1]-RAlim[0]) / fw )
    NDEC = int( (DEClim[1]-DEClim[0]) / fw )

    RArange = RAlim[0] + fw * numpy.arange(NRA)
    DECrange = DEClim[0] + fw * numpy.arange(NDEC)
    
    mask = numpy.ones( (Npix*NRA,Npix*NDEC),
                        dtype = bool )

    if not blank:
        for RA in RArange:
            for DEC in DECrange:
                i = Npix*(RA-RAlim[0])/fw
                j = Npix*(DEC-DEClim[0])/fw
                X = numpy.load(filenames % (RA,DEC))
            
                mask[i:i+Npix,j:j+Npix] = X['mask']

                X.close()
            ###
        ###
    ###

    return mask




def Hikage_mask(N1,N2,
                dtheta1, #arcmin
                dtheta2 = None,
                hmin=0.2, #arcmin
                hmax=2.0, #arcmin
                xrec=0.2, #unitless
                yrec=5.0, #unitless
                rsmin=0.3, #arcmin
                fmask=0.1, #unitless
                ):
    """
    stars of radius between hmin and hmax
    saturation spike is xrec*radius by yrec*radius, for stars of radius
    greater than rsmin
    go until fmask = fraction of masked pixels
    """
    if dtheta2 is None:
        dtheta2 = dtheta1
    
    mask = numpy.ones( (N1,N2),dtype=bool )
    nmask = int(N1*N2*fmask) #number of pixels to mask
    
    next=0
    while( next < nmask):
        p1 = numpy.random.random()*N1
        p2 = numpy.random.random()*N2
        hsize = (hmin+(hmax-hmin)*numpy.random.random()**4)
        imin = max( 0, int(p1-hsize/dtheta1) )
        imax = min( N1, int(p1+hsize/dtheta1)+1 )
        jmin = max( 0, int(p2-hsize/dtheta2) )
        jmax = min( N2, int(p2+hsize/dtheta2)+1 )

        for i in range(imin,imax):
            for j in range(jmin,jmax):
                if not mask[i,j]:continue
                if ((i-p1)*dtheta1)**2+((j-p2)*dtheta2)**2<hsize**2:
                    mask[i,j]=0
                    next += 1
        if hsize>rsmin:
            s1 = hsize*xrec/dtheta1
            s2 = hsize*yrec/dtheta2
            imin=int(numpy.ceil(p1-s1))
            imax=int(numpy.ceil(p1+s1))
            jmin=int(numpy.ceil(p2-s2))
            jmax=int(numpy.ceil(p2+s2))
            M = mask[imin:imax,jmin:jmax]
            next += numpy.sum(M)
            M[:,:] = 0

    return mask

            
if __name__ == '__main__':
    from ..tools import gamma_to_kappa
    
    print "reading reconstruction"
    gamma = get_gamma_reconstructed(mask=False)
    dtheta = params.field_width / params.Npix

    print "computing kappa via FFT"
    kappa = gamma_to_kappa(gamma,dtheta)
    show_kappa(kappa,part='a',logplot=True)

    print "reading true gamma"
    gamma_true = get_gamma_true()

    print "computing kappa via FFT"
    kappa_true = gamma_to_kappa(gamma_true,dtheta)
    show_kappa(kappa_true,part='a',logplot=True)
    
    pylab.show()
