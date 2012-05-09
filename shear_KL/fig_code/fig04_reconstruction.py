import numpy
import pylab
import sys
import os

from matplotlib.ticker import MultipleLocator
import matplotlib
matplotlib.use('Agg')

sys.path.append(os.path.abspath('../'))

from shear_KL_source.DES_tile.tools import get_mask, get_from_parsed
from shear_KL_source.DES_KL.tools import whiskerplot, GreyWhite

from shear_KL_source import params
params.load('../run/base_params.dat')

def get_reconstruction(nmodes,
                       alpha,
                       mask=True,
                       RAmin = 10,
                       DECmin = 35 ):
    if mask:
        dir = '%i_y_a%.2fy' % (nmodes,alpha)
    else:
        dir = '%i_n_a%.2fy' % (nmodes,alpha)

    if dir not in os.listdir(params.shear_recons_dir):
        raise ValueError, "%s not present in %s" % (dir, params.shear_recons_dir)

    dir = os.path.join(params.shear_recons_dir,dir)

    filename = "reconstruct_%.1f_%.1f.npz" % (RAmin,DECmin)

    if filename not in os.listdir(dir):
        print dir
        print filename
        raise ValueError, "RAmin=%.1f DECmin=%.1f not present" % (RAmin,DECmin)

    X = numpy.load(os.path.join(dir,filename))

    return X['shear'],X['noise']


def plot_with_mask_4(nmodes=900,
                     alpha=0.15,
                     RAmin = 10,
                     DECmin = 35,
                     xlim=(0,60),
                     ylim=(0,60),
                     plot_SN=False):
    RAlim = (RAmin,RAmin+1)
    DEClim = (DECmin,DECmin+1)
    
    gamma_out_masked, noise = get_reconstruction(nmodes,
                                                 alpha,
                                                 True,
                                                 RAmin,
                                                 DECmin)
    
    gamma_out_unmasked, noise = get_reconstruction(nmodes,
                                                   alpha,
                                                   False,
                                                   RAmin,
                                                   DECmin)
    

    if plot_SN:
        gamma_out_masked /= numpy.sqrt(noise_masked)
        gamma_out_unmasked /= numpy.sqrt(noise_unmasked)
    
    diff1 = gamma_out_masked-gamma_out_unmasked
    rms1 = numpy.sqrt(numpy.mean(abs(diff1[10:50,10:50])**2))
    
    mask = get_mask(False,RAlim,DEClim)

    gamma_true = get_from_parsed('gamma',RAlim=RAlim,DEClim=DEClim)
    
    diff2 = gamma_out_unmasked-gamma_true
    rms2 = numpy.sqrt(numpy.mean(abs(diff2[10:50,10:50])**2))
    
    diff3 = gamma_out_masked-gamma_true
    rms3 = numpy.sqrt(numpy.mean(abs(diff3[10:50,10:50])**2))

    assert gamma_out_unmasked.shape == gamma_true.shape

    dtheta = params.dtheta

    RAside = (RAlim[1]-RAlim[0])*60
    DECside = (DEClim[1]-DEClim[0])*60

    pylab.figure( figsize=(10,10) )
    #----------------------------------------
    
    pylab.subplot(221)
    pylab.imshow(mask.T*0,
                 origin='lower',
                 extent=(0,RAside,0,DECside),
                 interpolation='nearest',
                 cmap=pylab.cm.binary)
    pylab.clim(0,1)
    whiskerplot(gamma_true,params.dtheta,params.dtheta)
    pylab.xlim(xlim)
    pylab.ylim(ylim)
    pylab.title('shear field (noiseless)')

    #----------------------------------------

    pylab.subplot(222)
    pylab.imshow(mask.T*0,
                 origin='lower',
                 extent=(0,RAside,0,DECside),
                 interpolation='nearest',
                 cmap=pylab.cm.binary)
    pylab.clim(0,1)
    whiskerplot(gamma_true+noise,params.dtheta,params.dtheta)
    pylab.xlim(xlim)
    pylab.ylim(ylim)
    pylab.title('shear field (with noise)')

    #----------------------------------------
    
    pylab.subplot(223)
    pylab.imshow(mask.T*0,
                 origin='lower',
                 extent=(0,RAside,0,DECside),
                 interpolation='nearest',
                 cmap=pylab.cm.binary)
    pylab.clim(0,1)
    whiskerplot(gamma_out_unmasked,params.dtheta,params.dtheta)
    pylab.xlim(xlim)
    pylab.ylim(ylim)
    pylab.title('unmasked reconstruction')

    #----------------------------------------
    
    pylab.subplot(224)
    pylab.imshow(mask.T,
                 origin='lower',
                 extent=(0,RAside,0,DECside),
                 interpolation='nearest',
                 cmap=GreyWhite)
    pylab.clim(0,1)
    whiskerplot(gamma_out_masked,params.dtheta,params.dtheta)
    pylab.xlim(xlim)
    pylab.ylim(ylim)
    pylab.title('masked reconstruction')

    #----------------------------------------
    
    print "%i %.2f %.4f %.4f %.4f" % (nmodes,alpha,rms1,rms2,rms3)


def plot_with_mask_6(nmodes=900,
                     alpha=0.15,
                     RAmin = 10,
                     DECmin = 35,
                     xlim=(0,60),
                     ylim=(0,60),
                     plot_SN=False,
                     additional_n = [500,1500],
                     additional_a = [0.0,0.0] ):
    RAlim = (RAmin,RAmin+1)
    DEClim = (DECmin,DECmin+1)

    combine = 1
    if combine==1:
        scale = 5
    elif combine==2:
        scale = 4

    assert len(additional_n) == 2
    assert len(additional_a) == 2
    
    gamma_out_masked, noise = get_reconstruction(nmodes,
                                                 alpha,
                                                 True,
                                                 RAmin,
                                                 DECmin)
    
    gamma_out_unmasked, noise = get_reconstruction(nmodes,
                                                   alpha,
                                                   False,
                                                   RAmin,
                                                   DECmin)
    
    g1,n1 = get_reconstruction(additional_n[0],
                               additional_a[0],
                               True,
                               RAmin,
                               DECmin)
    
    g2,n2 = get_reconstruction(additional_n[1],
                               additional_a[1],
                               True,
                               RAmin,
                               DECmin)
                               

    if plot_SN:
        gamma_out_masked /= numpy.sqrt(noise_masked)
        gamma_out_unmasked /= numpy.sqrt(noise_unmasked)
        g1 /= numpy.sqrt(n1)
        g2 /= numpy.sqrt(n2)

    
    diff1 = gamma_out_masked-gamma_out_unmasked
    rms1 = numpy.sqrt(numpy.mean(abs(diff1[10:50,10:50])**2))
    
    mask = get_mask(False,RAlim,DEClim)

    gamma_true = get_from_parsed('gamma',RAlim=RAlim,DEClim=DEClim)
    
    diff2 = gamma_out_unmasked-gamma_true
    rms2 = numpy.sqrt(numpy.mean(abs(diff2[10:50,10:50])**2))
    
    diff3 = gamma_out_masked-gamma_true
    rms3 = numpy.sqrt(numpy.mean(abs(diff3[10:50,10:50])**2))

    assert gamma_out_unmasked.shape == gamma_true.shape

    dtheta = params.dtheta/60.

    RAside = (RAlim[1]-RAlim[0])
    DECside = (DEClim[1]-DEClim[0])

    pylab.figure( figsize=(8,11) )

    offset = (RAmin,DECmin)
    dtheta = params.dtheta/60.
    
    xlim = (offset[0] + xlim[0]/60.,
            offset[0] + xlim[1]/60.)
    ylim = (offset[1] + ylim[0]/60.,
            offset[1] + ylim[1]/60.)

    extent=(offset[0],offset[0]+RAside,
            offset[1],offset[1]+DECside)

    #----------------------------------------
    
    ax = pylab.subplot(321)
    pylab.imshow(mask.T*0,
                 origin='lower',
                 extent=extent,
                 interpolation='nearest',
                 cmap=pylab.cm.binary)
    pylab.clim(0,1)
    whiskerplot(gamma_true,dtheta,dtheta,
                combine=combine,scale=scale,offset=offset)
    pylab.xlim(xlim)
    pylab.ylim(ylim)
    pylab.title('shear field (noiseless)')
    pylab.xlabel('')
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))

    #----------------------------------------

    ax = pylab.subplot(322)
    pylab.imshow(mask.T*0,
                 origin='lower',
                 extent=extent,
                 interpolation='nearest',
                 cmap=pylab.cm.binary)
    pylab.clim(0,1)
    whiskerplot(gamma_true+noise,dtheta,dtheta,
                combine=combine,scale=scale,offset=offset)
    pylab.xlim(xlim)
    pylab.ylim(ylim)
    pylab.title('shear field (with noise)')
    pylab.xlabel('')
    pylab.ylabel('')
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))

    #----------------------------------------
    
    ax = pylab.subplot(323)
    pylab.imshow(mask.T*0,
                 origin='lower',
                 extent=extent,
                 interpolation='nearest',
                 cmap=pylab.cm.binary)
    pylab.clim(0,1)
    whiskerplot(gamma_out_unmasked,dtheta,dtheta,
                combine=combine,scale=scale,offset=offset)
    pylab.xlim(xlim)
    pylab.ylim(ylim)
    pylab.title('unmasked KL (n=%i)' % nmodes)
    pylab.xlabel('')
    #pylab.text(0.97,0.97,'n=%i' % nmodes,
    #           transform = pylab.gca().transAxes,
    #           va = 'top',
    #           ha = 'right',
    #           bbox=dict(facecolor='w',edgecolor='w') )
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))

    #----------------------------------------
    
    ax = pylab.subplot(324)
    pylab.imshow(mask.T,
                 origin='lower',
                 extent=extent,
                 interpolation='nearest',
                 cmap=GreyWhite)
    pylab.clim(0,1)
    whiskerplot(g1,dtheta,dtheta,
                combine=combine,scale=scale,offset=offset)
    pylab.xlim(xlim)
    pylab.ylim(ylim)
    pylab.title(r'$\mathdefault{masked\ KL}$')
    pylab.xlabel('')
    pylab.ylabel('')
    pylab.text(0.97,0.97,'n=%i' % additional_n[0],
               transform = pylab.gca().transAxes,
               va = 'top',
               ha = 'right',
               bbox=dict(facecolor='w',edgecolor='w') )
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))

    #----------------------------------------
    
    ax = pylab.subplot(325)
    pylab.imshow(mask.T,
                 origin='lower',
                 extent=extent,
                 interpolation='nearest',
                 cmap=GreyWhite)
    pylab.clim(0,1)
    whiskerplot(gamma_out_masked,dtheta,dtheta,
                combine=combine,scale=scale,offset=offset)
    pylab.xlim(xlim)
    pylab.ylim(ylim)
    pylab.title(r'$\mathdefault{masked\ KL}$')
    #pylab.xlabel('')
    #pylab.ylabel('')
    pylab.text(0.97,0.97,'n=%i' % nmodes,
               transform = pylab.gca().transAxes,
               va = 'top',
               ha = 'right',
               bbox=dict(facecolor='w',edgecolor='w') )
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))

    #----------------------------------------
    
    ax = pylab.subplot(326)
    pylab.imshow(mask.T,
                 origin='lower',
                 extent=extent,
                 interpolation='nearest',
                 cmap=GreyWhite)
    pylab.clim(0,1)
    whiskerplot(g2,dtheta,dtheta,
                combine=combine,scale=scale,offset=offset)
    pylab.xlim(xlim)
    pylab.ylim(ylim)
    pylab.title(r'$\mathdefault{masked\ KL\ (n=%i)}$' % additional_n[1])
    pylab.ylabel('')
    #pylab.text(0.97,0.97,'n=%i' % additional_n[1],
    #           transform = pylab.gca().transAxes,
    #           va = 'top',
    #           ha = 'right',
    #           bbox=dict(facecolor='w',edgecolor='w') )
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))

    #----------------------------------------

    pylab.suptitle(r'$\mathdefault{Reconstructions\ with\ n_{gal}=20/arcmin^2}$',
                   fontsize=16)

    
def plot_with_mask_6_horiz(nmodes=900,
                           alpha=0.15,
                           RAmin = 10,
                           DECmin = 35,
                           xlim=(0,60),
                           ylim=(0,60),
                           plot_SN=False,
                           additional_n = [500,1500],
                           additional_a = [0.0,0.0] ):
    from six_squares import six_squares
    
    RAlim = (RAmin,RAmin+1)
    DEClim = (DECmin,DECmin+1)

    combine = 1
    if combine==1:
        scale = 5
    elif combine==2:
        scale = 4

    assert len(additional_n) == 2
    assert len(additional_a) == 2
    
    gamma_out_masked, noise = get_reconstruction(nmodes,
                                                 alpha,
                                                 True,
                                                 RAmin,
                                                 DECmin)
    
    gamma_out_unmasked, noise = get_reconstruction(nmodes,
                                                   alpha,
                                                   False,
                                                   RAmin,
                                                   DECmin)
    
    g1,n1 = get_reconstruction(additional_n[0],
                               additional_a[0],
                               True,
                               RAmin,
                               DECmin)
    
    g2,n2 = get_reconstruction(additional_n[1],
                               additional_a[1],
                               True,
                               RAmin,
                               DECmin)
                               

    if plot_SN:
        gamma_out_masked /= numpy.sqrt(noise_masked)
        gamma_out_unmasked /= numpy.sqrt(noise_unmasked)
        g1 /= numpy.sqrt(n1)
        g2 /= numpy.sqrt(n2)

    
    diff1 = gamma_out_masked-gamma_out_unmasked
    rms1 = numpy.sqrt(numpy.mean(abs(diff1[10:50,10:50])**2))
    
    mask = get_mask(False,RAlim,DEClim)

    gamma_true = get_from_parsed('gamma',RAlim=RAlim,DEClim=DEClim)
    
    diff2 = gamma_out_unmasked-gamma_true
    rms2 = numpy.sqrt(numpy.mean(abs(diff2[10:50,10:50])**2))
    
    diff3 = gamma_out_masked-gamma_true
    rms3 = numpy.sqrt(numpy.mean(abs(diff3[10:50,10:50])**2))

    assert gamma_out_unmasked.shape == gamma_true.shape

    dtheta = params.dtheta/60.

    RAside = (RAlim[1]-RAlim[0])
    DECside = (DEClim[1]-DEClim[0])

    #pylab.figure( figsize=(10,8) )
    ax_list = six_squares(11,8)

    offset = (RAmin,DECmin)
    dtheta = params.dtheta/60.
    
    xlim = (offset[0] + xlim[0]/60.,
            offset[0] + xlim[1]/60.)
    ylim = (offset[1] + ylim[0]/60.,
            offset[1] + ylim[1]/60.)

    extent=(offset[0],offset[0]+RAside,
            offset[1],offset[1]+DECside)

    #----------------------------------------

    ax = ax_list[0]
    pylab.sca(ax)
    #ax = pylab.subplot(231)
    pylab.imshow(mask.T*0,
                 origin='lower',
                 extent=extent,
                 interpolation='nearest',
                 cmap=pylab.cm.binary)
    pylab.clim(0,1)
    whiskerplot(gamma_true,dtheta,dtheta,
                combine=combine,scale=scale,offset=offset)
    pylab.xlim(xlim)
    pylab.ylim(ylim)
    pylab.title('shear field (noiseless)')
    pylab.xlabel('')
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))

    #----------------------------------------
 
    ax = ax_list[1]
    pylab.sca(ax)
    #ax = pylab.subplot(232)
    pylab.imshow(mask.T*0,
                 origin='lower',
                 extent=extent,
                 interpolation='nearest',
                 cmap=pylab.cm.binary)
    pylab.clim(0,1)
    whiskerplot(gamma_true+noise,dtheta,dtheta,
                combine=combine,scale=scale,offset=offset)
    pylab.xlim(xlim)
    pylab.ylim(ylim)
    pylab.title('shear field (with noise)')
    pylab.xlabel('')
    pylab.ylabel('')
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))

    #----------------------------------------
    
    ax = ax_list[2]
    pylab.sca(ax)
    #ax = pylab.subplot(233)
    pylab.imshow(mask.T*0,
                 origin='lower',
                 extent=extent,
                 interpolation='nearest',
                 cmap=pylab.cm.binary)
    pylab.clim(0,1)
    whiskerplot(gamma_out_unmasked,dtheta,dtheta,
                combine=combine,scale=scale,offset=offset)
    pylab.xlim(xlim)
    pylab.ylim(ylim)
    pylab.title('unmasked KL (n=%i)' % nmodes)
    pylab.xlabel('')
    pylab.ylabel('')
    #pylab.text(0.97,0.97,'n=%i' % nmodes,
    #           transform = pylab.gca().transAxes,
    #           va = 'top',
    #           ha = 'right',
    #           bbox=dict(facecolor='w',edgecolor='w') )
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))

    #----------------------------------------
    
    ax = ax_list[3]
    pylab.sca(ax)
    #ax = pylab.subplot(234)
    pylab.imshow(mask.T,
                 origin='lower',
                 extent=extent,
                 interpolation='nearest',
                 cmap=GreyWhite)
    pylab.clim(0,1)
    whiskerplot(g1,dtheta,dtheta,
                combine=combine,scale=scale,offset=offset)
    pylab.xlim(xlim)
    pylab.ylim(ylim)
    pylab.title(r'$\mathdefault{masked\ KL\ (n=%i)}$' % additional_n[0])
    #pylab.text(0.97,0.97,'n=%i' % additional_n[0],
    #           transform = pylab.gca().transAxes,
    #           va = 'top',
    #           ha = 'right',
    #           bbox=dict(facecolor='w',edgecolor='w') )
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))

    #----------------------------------------
    
    ax = ax_list[4]
    pylab.sca(ax)
    #ax = pylab.subplot(235)
    pylab.imshow(mask.T,
                 origin='lower',
                 extent=extent,
                 interpolation='nearest',
                 cmap=GreyWhite)
    pylab.clim(0,1)
    whiskerplot(gamma_out_masked,dtheta,dtheta,
                combine=combine,scale=scale,offset=offset)
    pylab.xlim(xlim)
    pylab.ylim(ylim)
    pylab.title(r'$\mathdefault{masked\ KL\ (n=%i)}$' % nmodes)
    pylab.ylabel('')
    #pylab.text(0.97,0.97,'n=%i' % nmodes,
    #           transform = pylab.gca().transAxes,
    #           va = 'top',
    #           ha = 'right',
    #           bbox=dict(facecolor='w',edgecolor='w') )
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))

    #----------------------------------------
    
    ax = ax_list[5]
    pylab.sca(ax)
    #ax = pylab.subplot(236)
    pylab.imshow(mask.T,
                 origin='lower',
                 extent=extent,
                 interpolation='nearest',
                 cmap=GreyWhite)
    pylab.clim(0,1)
    whiskerplot(g2,dtheta,dtheta,
                combine=combine,scale=scale,offset=offset)
    pylab.xlim(xlim)
    pylab.ylim(ylim)
    pylab.title(r'$\mathdefault{masked\ KL\ (n=%i)}$' % additional_n[1])
    pylab.ylabel('')
    #pylab.text(0.97,0.97,'n=%i' % additional_n[1],
    #           transform = pylab.gca().transAxes,
    #           va = 'top',
    #           ha = 'right',
    #           bbox=dict(facecolor='w',edgecolor='w') )
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))

    #----------------------------------------

    pylab.suptitle(r'$\mathdefault{Reconstructions\ with\ n_{gal}=20/arcmin^2}$',
                   fontsize=16)
    
    
    
def main_1():
    #this shows a nicely reconstructed peak that is near a large masked region
    nmodes = 900
    alpha = 0.15
    plot_with_mask_4(nmodes,
                     alpha,
                     RAmin=11.5,
                     DECmin=36,
                     xlim=(10,50),
                     ylim=(10,50))
    title = 'nmodes=%i alpha=%.2f' % (nmodes,alpha)
    pylab.suptitle(title)
    pylab.savefig('figs/shear_reconstruction.eps')
    pylab.savefig('figs/shear_reconstruction.pdf')
    pylab.show()

def main_2():
    #this shows a nicely reconstructed peak that is near a large masked region
    nmodes = 900
    alpha = 0.15
    add_n = (100,2000)
    add_a = (0,0)
    plot_with_mask_6_horiz(nmodes,
                           alpha,
                           RAmin=11.5,
                           DECmin=36,
                           xlim=(15,45),
                           ylim=(20,50),
                           additional_n = add_n,
                           additional_a = add_a)


    
if __name__ == '__main__':
    main_2()
    pylab.savefig('figs/fig04_reconstruction.eps')
    pylab.savefig('figs/fig04_reconstruction.pdf')
    pylab.show()
