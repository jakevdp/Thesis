"""
Tools for doing KL reconstructions of DES mock shear fields
"""
import numpy
import pylab

# Create a GreyWhite color map
from matplotlib.colors import LinearSegmentedColormap
L = ((0.0,0.5,0.5),
     (1.0,1.0,1.0))
GreyWhite = LinearSegmentedColormap('GreyWhite',
                                    dict([(color,L) for color in ['red',
                                                                  'blue',
                                                                  'green']]) )



def whiskerplot(shear,dRA=1.,dDEC=1.,scale=5, combine=1,offset=(0,0) ):
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
    RA = offset[0] + numpy.arange(shear.shape[0])*dRA
    DEC = offset[1] + numpy.arange(shear.shape[1])*dDEC

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
                 

def read_shear_out(filename='shear_out.dat',
                   return_N = False,
                   return_kappa = False):
    print "reading",filename
    N = None
    dtheta = None
    kappa = None
    
    shear = None
    Ngal = None
    i1 = None
    i2 = None
    ix = None
    iy = None
    ik = None
    
    for line in open(filename):
        if line.startswith('#'):
            if line.startswith('#NPIX'):
                N = int(line.split()[1])
                shear = numpy.zeros((N,N),dtype=complex)
                Ngal = numpy.zeros((N,N),dtype=int)
                kappa = numpy.zeros((N,N),dtype=float)
            elif line.startswith('#DTHETA'):
                dtheta = float(line.split()[1])
            elif line.startswith('#COLS'):
                COLS = line.split()[1:]
                i1 = COLS.index('shear1')
                i2 = COLS.index('shear2')
                ix = COLS.index('bin_x')
                iy = COLS.index('bin_y')
                iN = COLS.index('Ngal')
                ik = COLS.index('kappa')
        else:
            line = line.split()
            if len(line)==0:continue
            shear[int(line[ix]),int(line[iy])] \
                = float(line[i1])+1j*float(line[i2])
            kappa[int(line[ix]),int(line[iy])] = float(line[ik])
            Ngal[int(line[ix]),int(line[iy])] = int(line[iN])

    ret = [shear,dtheta]
    if return_N:
        ret.append(Ngal)
    if return_kappa:
        ret.append(kappa)
    return tuple(ret)

def create_mask(Nx,Ny,frac,
                rmin = 0.5,
                rmax = 2):
    """
    create a mask Nx by Ny pixels
    frac: 0 <= frac <= 1: fraction of pixels to be covered
    """
    mask = numpy.ones((Nx,Ny))

    ncovered = 0
    goal = frac*Nx*Ny

    while ncovered < goal:
        x = Nx*numpy.random.random()
        y = Ny*numpy.random.random()
        r = rmin + numpy.random.random()*(rmax-rmin)
        
        xmin = max(0,int(numpy.floor(x-r)))
        xmax = min(Nx,int(numpy.ceil(x+r)))
        ymin = max(0,int(numpy.floor(y-r)))
        ymax = min(Ny,int(numpy.ceil(y+r)))

        for ix in range(xmin,xmax):
            for iy in range(ymin,ymax):
                if (x-ix)**2 + (y-iy)**2 < r**2:
                    ncovered += mask[ix,iy]
                    mask[ix,iy] = 0
    
    return mask
