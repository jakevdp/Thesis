import sys

import numpy as np
import pylab

sys.path.append('Map')
from peaks import Aperture_Mass

from read_shear_catalog import read_shear_catalog
from KS_method import gamma_to_kappa

def whiskerplot(shear,dRA=1.,dDEC=1.,scale=5, combine=1, offset=(0,0) ):
    if combine>1:
        s = (combine*int(shear.shape[0]/combine),
             combine*int(shear.shape[1]/combine))
        shear = shear[0:s[0]:combine, 0:s[1]:combine] \
                + shear[1:s[0]:combine, 0:s[1]:combine] \
                + shear[0:s[0]:combine, 1:s[1]:combine] \
                + shear[1:s[0]:combine, 1:s[1]:combine]
        shear *= 0.25

        dRA *= combine
        dDEC *= combine

    
    theta = shear**0.5
    RA = offset[0] + np.arange(shear.shape[0])*dRA
    DEC = offset[1] + np.arange(shear.shape[1])*dDEC

    pylab.quiver(RA,DEC,
                 theta.real.T,theta.imag.T,
                 pivot = 'middle',
                 headwidth = 0,
                 headlength = 0,
                 headaxislength = 0,
                 scale=scale)
    pylab.xlim(0,shear.shape[0]*dRA)
    pylab.ylim(0,shear.shape[1]*dDEC)
    pylab.xlabel('RA (arcmin)')
    pylab.ylabel('DEC (arcmin)')

def plot_shear(filename, dpix=1.0):
    """
    dpix is pixel size in arcmin
    """
    RA, DEC, gamma1, gamma2 = read_shear_catalog(filename,
                                                 ('Ra','Dec',
                                                  'e1iso_rot4_gr_snCal',
                                                  'e2iso_rot4_gr_snCal'),
                                                 None)
    Ngal = len(RA)

    #bin data into pixels
    RAmin, RAmax = min(RA), max(RA)
    DECmin, DECmax = min(DEC), max(DEC)
    
    i = np.floor( (RA-RAmin)*60./dpix ).astype(int)
    j = np.floor( (DEC-DECmin)*60./dpix ).astype(int)

    Nx = np.max(i)+1
    Ny = np.max(j)+1

    print Nx,Ny

    gamma = np.zeros((Nx,Ny), dtype=complex)
    ngal  = np.zeros((Nx,Ny), dtype=int)

    for ind in xrange(Ngal):
        gamma[i[ind], j[ind]] += gamma1[ind] - 1j*gamma2[ind]
        ngal[i[ind], j[ind]] += 1

    izero = np.where(ngal==0)
    ngal[izero] = 1
    gamma /= ngal
    ngal[izero] = 0

    kappa = gamma_to_kappa(gamma,dpix)

    extent = (-0.5*dpix, (Nx-0.5)*dpix,
               -0.5*dpix, (Ny-0.5)*dpix)

    pylab.figure()
    pylab.imshow(ngal.T, origin='lower', extent=extent, interpolation='nearest')
    pylab.colorbar().set_label(r'$\mathdefault{n_{gal} (per pixel)}$')
    whiskerplot(gamma, dpix, dpix)
    pylab.axis(extent)

    pylab.figure()
    pylab.imshow(kappa.real.T, origin='lower', 
                 extent=extent)
    cb1 = pylab.colorbar()
    clim = cb1.get_clim()
    pylab.title('kappa: E-mode')
    pylab.axis(extent)

    pylab.figure()
    pylab.imshow(kappa.imag.T, origin='lower', 
                 extent=extent)
    cb2 = pylab.colorbar()
    cb2.set_clim(clim)
    pylab.title('kappa: B-mode')
    pylab.axis(extent)

def plot_Map(filename, dpix=1.0):
    RA, DEC, gamma1, gamma2 = read_shear_catalog(filename,
                                                 ('Ra','Dec',
                                                  'e1iso_rot4_gr_snCal',
                                                  'e2iso_rot4_gr_snCal'),
                                                 None)
    RAmin, RAmax = min(RA), max(RA)
    DECmin, DECmax = min(DEC), max(DEC)

    dpix_deg = dpix/60.

    pos = RA + 1j*DEC
    gamma = gamma1 + 1j*gamma2

    RA_out = np.arange(RAmin,RAmax+dpix_deg,dpix_deg)
    DEC_out = np.arange(DECmin,DECmax+dpix_deg,dpix_deg)

    RA_out,DEC_out = np.meshgrid(RA_out,DEC_out)

    pos_out = RA_out + 1j*DEC_out
    
    shape = pos_out.shape

    Map = Aperture_Mass(gamma, pos, pos_out.reshape(-1))

    Map = Map.reshape(shape)

    pylab.figure()
    pylab.imshow(Map.real.T,
                 origin='lower')
    pylab.colorbar()

    pylab.figure()
    pylab.imshow(Map.imag.T,
                 origin='lower')
    pylab.colorbar()

if __name__ == '__main__':
    from params import BRIGHT_CAT, FAINT_CAT
    plot_shear(BRIGHT_CAT, 1.0)
    plot_shear(FAINT_CAT, 1.0)

    #plot_Map(BRIGHT_CAT, dpix=1.)
    pylab.show()
