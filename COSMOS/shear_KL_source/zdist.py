"""
redshift distribution functions
"""
import numpy as np
from scipy import integrate, interpolate

class zdist_base(object):
    def __init__(self, zlim, func):
        self.zlim = zlim

    def _dist_func(self, z):
        raise NotImplementedError, "__dist_func"
    
    def __call__(self, z):
        return self._dist_func(z)

    zmin = property(lambda self: self.zlim[0])
    zmax = property(lambda self: self.zlim[1])

    def integrate(self, z0, z1):
        return integrate.quad(self._dist_func, z0, z1)[0]

    def integrate_weighted(self, w, z0, z1):
        return integrate.quad(lambda z: w(z) * self._dist_func(z),
                              z0, z1)[0]


class zdist_parametrized(zdist_base):
    def __init__(self, z0, a, b, zlim):
        self.zlim = zlim
        self.z0 = float(z0)
        self.a = float(a)
        self.b = float(b)

    def _dist_func(self, z):
        return z ** self.a * np.exp(-(z / self.z0) ** self.b)
    

class zdist_from_file(zdist_base):
    def __init__(self, filename, dz=None):
        """
        dz gives the window size through which to view the distribution
        """
        X = np.loadtxt(filename)
        self.z = X[:,0]
        self.n_z = X[:,1]

        self.zlim = (self.z[0], self.z[-1])

        if dz:
            n = int(np.ceil(dz / (self.z[1] - self.z[0])))
            print n
            self.n_z = np.correlate(self.n_z,
                                    (1./n) * np.ones(n, dtype=float),
                                    mode='same')

        self.tck = interpolate.splrep(self.z, self.n_z)

    
    def _dist_func(self, z):
        n_z = interpolate.splev(z, self.tck)

        if hasattr(z, '__len__'):
            i = np.where((z < self.zlim[0]) | (z > self.zlim[1]))
            n_z[i] = 0
        elif z < self.zlim[0] or z > self.zlim[1]:
            n_z = 0

        return n_z

    def integrate(self, z0, z1):
        return self.integrate_weighted(lambda z: z*0+1,
                                       z0,z1)

    def integrate_weighted(self, w, z0, z1):
        if z0==z1:
            return 0
        if z1 <= self.z[0] or z0 > self.z[-1]:
            return 0
        
        zmin = self.z[0]
        dz = self.z[1] - self.z[0]
        zmax = zmin + dz * len(self.z)

        i0 = int(np.floor((z0-zmin)/dz))
        i1 = int(np.ceil((z1-zmin)/dz))

        if i0 < 0: i0 = 0
        if i1 > len(self.z): i1 = len(self.z)

        I = dz * np.mean(w(self.z[i0:i1]) * self.n_z[i0:i1])

        #subtract left portion
        if z0 > self.z[i0]:
            wnz0 = w(z0) * (self.n_z[i0] * (self.z[i0+1] - z0)/dz +
                           self.n_z[i1] * (z0 - self.z[i0])/dz)
            I -= (0.5 * (z0 - self.z[i0])
                  * (w(self.z[i0]) * self.n_z[i0] + wnz0))

        #subtract right portion
        if z1 < self.z[i1-1] + dz:
            zi0 = self.z[i1-1]
            zi1 = zi0 + dz
            nzi0 = self.n_z[i1-1]
            try:
                nzi1 = self.n_z[i1]
            except:
                nzi1 = 0
                
            wnz1 = w(z1) * (nzi0 * (zi1 - z1)/dz + nzi1 * (z1 - zi0)/dz)
            I -= 0.5 * (zi1 - z1) * (wnz1 + w(zi1) * nzi1)

        return I
        
