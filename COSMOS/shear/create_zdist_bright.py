import numpy as np

from read_shear_catalog import read_shear_catalog

from params import BRIGHT_ZDIST_ZPROB0, BRIGHT_ZDIST, BRIGHT_CAT_NPZ

def create_zdist_bright():
    zphot_0 = \
            read_shear_catalog(BRIGHT_CAT_NPZ,
                               ('zphot',),
                               None,
                               remove_z_problem = False)
    zphot_1 = \
            read_shear_catalog(BRIGHT_CAT_NPZ,
                               ('zphot',),
                               None,
                               remove_z_problem = True)

    z = np.arange(0, 5, 0.01)

    i0 = z.searchsorted(zphot_0[0])
    i1 = z.searchsorted(zphot_1[0])

    OF_0 = open(BRIGHT_ZDIST, 'w')
    OF_1 = open(BRIGHT_ZDIST_ZPROB0, 'w')

    for i in range(len(z)):
        OF_0.write('%.6f %.6g\n' % (z[i], len(np.where(i0==i)[0])))
        OF_1.write('%.6f %.6g\n' % (z[i], len(np.where(i1==i)[0])))

    OF_0.close()
    OF_1.close()

if __name__ == '__main__':
    create_zdist_bright()
    
