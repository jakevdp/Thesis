import numpy
import pylab
import sys
import os

from matplotlib.ticker import MultipleLocator

sys.path.append(os.path.abspath('../'))

from shear_KL_source.DES_tile.tools import get_mask, get_from_parsed
from shear_KL_source.DES_KL.tools import whiskerplot, GreyWhite

from shear_KL_source import params

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
        print params.shear_recons_dir
        raise ValueError, "nmodes=%i not present" % nmodes

    dir = os.path.join(params.shear_recons_dir,dir)

    filename = "reconstruct_%.1f_%.1f.npz" % (RAmin,DECmin)

    if filename not in os.listdir(dir):
        print dir
        print filename
        raise ValueError, "RAmin=%.1f DECmin=%.1f not present" % (RAmin,DECmin)

    X = numpy.load(os.path.join(dir,filename))

    return X['shear'],X['noise']

def plot_reconstruction_2():
    nmodes=900
    alpha=0.15
    RAmin=11.5
    DECmin=36
    xlim=(15,45)
    ylim=(20,50)
    
    RAlim = (RAmin,RAmin+1)
    DEClim = (DECmin,DECmin+1)

    RAside = (RAlim[1]-RAlim[0])
    DECside = (DEClim[1]-DEClim[0])
    
    params.load('../run/base_params.dat')
    params.load('../run/params_sig0.1.dat')
    gamma_out_masked_1, noise = get_reconstruction(nmodes,
                                                 alpha,
                                                 True,
                                                 RAmin,
                                                 DECmin)
    gamma_out_unmasked_1, noise = get_reconstruction(nmodes,
                                                   alpha,
                                                   False,
                                                   RAmin,
                                                   DECmin)
    mask_1 = get_mask(False,RAlim,DEClim)
    
    params.load('../run/params_sig0.1_f0.35.dat')
    gamma_out_masked_2, noise = get_reconstruction(nmodes,
                                                 alpha,
                                                 True,
                                                 RAmin,
                                                 DECmin)
    mask_2 = get_mask(False,RAlim,DEClim)
    
    params.load('../run/params_sig0.1_f0.5.dat')
    gamma_out_masked_3, noise = get_reconstruction(nmodes,
                                                 alpha,
                                                 True,
                                                 RAmin,
                                                 DECmin)
    mask_3 = get_mask(False,RAlim,DEClim)

    offset = (RAmin,DECmin)
    dtheta = params.dtheta/60.
    
    xlim = (offset[0] + xlim[0]/60.,
            offset[0] + xlim[1]/60.)
    ylim = (offset[1] + ylim[0]/60.,
            offset[1] + ylim[1]/60.)

    extent=(offset[0],offset[0]+RAside,
            offset[1],offset[1]+DECside)
    
    #--------------------------------------------------
    pylab.figure(figsize=(8,8))
    ax = pylab.subplot(221)
    pylab.imshow(mask_1.T*0,
                 origin='lower',
                 extent=extent,
                 interpolation='nearest',
                 cmap=pylab.cm.binary)
    whiskerplot(gamma_out_unmasked_1,dtheta,dtheta,offset=offset)
    pylab.xlim(xlim)
    pylab.ylim(ylim)
    #pylab.title(r'$\mathdefault{unmasked\ KL}$')
    #pylab.xlabel('')
    pylab.text(0.97,0.97,'no mask',
               transform = ax.transAxes,
               va = 'top',
               ha = 'right',
               bbox=dict(facecolor='w',edgecolor='w') )
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))

    #--------------------------------------------------
    ax = pylab.subplot(222)
    pylab.imshow(mask_1.T,
                 origin='lower',
                 extent=extent,
                 interpolation='nearest',
                 cmap=GreyWhite)
    pylab.clim(0,1)
    whiskerplot(gamma_out_masked_1,dtheta,dtheta,offset=offset)
    pylab.xlim(xlim)
    pylab.ylim(ylim)
    #pylab.title(r'$\mathdefault{masked\ KL}$')
    #pylab.xlabel('')
    pylab.ylabel('')
    pylab.text(0.97,0.97,'20% mask',
               transform = ax.transAxes,
               va = 'top',
               ha = 'right',
               bbox=dict(facecolor='w',edgecolor='w') )
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))

    #--------------------------------------------------
    ax = pylab.subplot(223)
    pylab.imshow(mask_2.T,
                 origin='lower',
                 extent=extent,
                 interpolation='nearest',
                 cmap=GreyWhite)
    pylab.clim(0,1)
    whiskerplot(gamma_out_masked_2,dtheta,dtheta,offset=offset)
    pylab.xlim(xlim)
    pylab.ylim(ylim)
    #pylab.title(r'$\mathdefault{masked\ KL}$')
    pylab.text(0.97,0.97,'35% mask',
               transform = ax.transAxes,
               va = 'top',
               ha = 'right',
               bbox=dict(facecolor='w',edgecolor='w') )
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))

    #--------------------------------------------------
    ax = pylab.subplot(224)
    pylab.imshow(mask_3.T,
                 origin='lower',
                 extent=extent,
                 interpolation='nearest',
                 cmap=GreyWhite)
    pylab.clim(0,1)
    whiskerplot(gamma_out_masked_3,dtheta,dtheta,offset=offset)
    pylab.xlim(xlim)
    pylab.ylim(ylim)
    #pylab.title(r'$\mathdefault{masked\ KL}$')
    pylab.ylabel('')
    pylab.text(0.97,0.97,'50% mask',
               transform = ax.transAxes,
               va = 'top',
               ha = 'right',
               bbox=dict(facecolor='w',edgecolor='w') )
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    
    #--------------------------------------------------

    pylab.figtext(0.5,0.97,r'$\mathdefault{KL\ Reconstructions}$',
                  ha='center',va='top',
                  fontsize=16)
    pylab.figtext(0.5,0.94,r'$\mathdefault{n=900\ modes,\ \ n_{gal}=100/arcmin^2}$',
                  ha='center',va='top',
                  fontsize=14)


if __name__ == '__main__':
    plot_reconstruction_2()
    pylab.savefig('figs/fig05_reconstruction_2.eps')
    pylab.savefig('figs/fig05_reconstruction_2.pdf')
    pylab.show()
