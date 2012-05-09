import numpy as np
import pylab
from scipy import interpolate

def contour_logL(xvals, yvals, logL, resample=False):
    """
    draw 1, 2, and 3-sigma contours from a gridded log-likelihood
    """
    #resample logL
    if resample:
        x,y = np.meshgrid(xvals, yvals)
        x.resize(x.size)
        y.resize(y.size)
        logL = logL.copy()
        logL.resize(logL.size)
        
        tck = interpolate.bisplrep(x, y, logL)
        
        xvals = np.linspace(xvals[0], xvals[-1], 20)
        yvals = np.linspace(yvals[0], yvals[-1], 20)
        
        logL = interpolate.bisplev(xvals, yvals, tck)
    
    #normalize logL: start by making the max logL=0
    L = np.exp(logL - logL.max())
    L /= L.sum()

    #assume a well-behaved peak.  Find 1-, 2-, and 3-sigma values
    Llist = L.copy().reshape(L.size)
    Llist.sort()
    levels = Llist.cumsum()

    i1 = levels.searchsorted(1 - 0.63)
    i2 = levels.searchsorted(1 - 0.95)
    i3 = levels.searchsorted(1 - 0.997)

    v1 = Llist[i1]
    v2 = Llist[i2]
    v3 = Llist[i3]

    pylab.contour(xvals, yvals, L, [v1,v2,v3])

def plot_likelihood(outfile,
                    nmodes,
                    xaxis = 'Om',
                    yaxis = 'sigma8',
                    key = 'log(Likelihood)',
                    normalize = True):
    F = open(outfile)
    cosmo_str = F.next()
    columns = np.asarray(F.next().strip('#').strip().split())
    F.close()

    i_ncut = np.where(columns == 'ncut')[0]
    i_xaxis = np.where(columns == xaxis)[0]
    i_yaxis = np.where(columns == yaxis)[0]
    i_logL = np.where(columns == key)[0]

    if len(i_ncut) == 0:
        raise ValueError("key 'ncut' not found in column list")
    if len(i_xaxis) == 0:
        raise ValueError("key '%s' not found in column list" % xaxis)
    if len(i_yaxis) == 0:
        raise ValueError("key '%s' not found in column list" % yaxis)
    if len(i_logL) == 0:
        raise ValueError("key '%s' not found in column list" % key)

    i_ncut = i_ncut[0]
    i_xaxis = i_xaxis[0]
    i_yaxis = i_yaxis[0]
    i_logL = i_logL[0]
    
    X = np.loadtxt(outfile)
    
    whr = np.where(X[:,i_ncut] == nmodes)

    if len(whr[0]) == 0:
        raise ValueError("nmodes=%i not contained in file %s"
                         % (nmodes,outfile))

    X = X[whr]
    
    xvals = np.unique(X[:,i_xaxis])
    yvals = np.unique(X[:,i_yaxis])

    #print xvals
    #print yvals

    Nx = len(xvals)
    Ny = len(yvals)

    logL = X[:,i_logL]

    if i_xaxis < i_yaxis:
        logL = logL.reshape((Nx,Ny))
    else:
        logL = logL.reshape((Ny,Nx)).T

    if normalize:
        logL -= logL.max()
        i = np.where(logL < -15)
        logL[i] = -15

    if True:
        dx = xvals[1] - xvals[0]
        dy = yvals[1] - yvals[0]
        extent = (xvals[0] - 0.5 * dx,
                  xvals[-1] + 0.5 * dx,
                  yvals[0] - 0.5 * dy,
                  yvals[-1] + 0.5 * dy)
        aspect = (dx + xvals[-1] - xvals[0]) / (dy + yvals[-1] - yvals[0])
        
        pylab.imshow(logL.T,
                     origin = 'lower',
                     interpolation = None,  # 'nearest',
                     extent=extent,
                     aspect=aspect,
                     cmap = pylab.cm.bone)
    elif True:
        x, y = np.meshgrid(xvals,yvals)
        pylab.contourf(x, y, logL.T, 40)

        pylab.colorbar().set_label(key)
        
    contour_logL(xvals, yvals, logL.T)
    
    pylab.xlabel(xaxis)
    pylab.ylabel(yaxis)
    pylab.title('nmodes = %i' % nmodes)

if __name__ == '__main__':
    key = 'log(Likelihood)'
    #key = 'chi2'
    #key = 'log|det(C)|'

    import sys

    filename = sys.argv[1]
    try:
        i = int(sys.argv[2])
    except:
        i = 400

    print filename
    print "%i modes" % i
    
    pylab.figure()
    plot_likelihood(filename, i, key=key, normalize=True)
    pylab.show()
    
