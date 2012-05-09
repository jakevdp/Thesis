import numpy as np

def read_shear_catalog_npz(filename, cols=None,
                           Nmax=None, remove_z_problem=False,
                           zprob_cutoff=2):
    """
    cols can be either a list of integers, or a list of names
    """
    F = np.load(filename)
    colnames = F['colnames']
    data = F['data']

    x = np.arange(len(cols))
    i_cols = np.zeros(len(cols),dtype='i')

    if cols is None:
        i_cols = x
    else:
        for i,c in enumerate(cols):
            try:
                i_cols[i] = x[c]
            except:
                if c not in colnames:
                    raise ValueError("'%s' not among column names" % c)
                i_cols[i] = np.where(colnames==c)[0]

    if remove_z_problem and 'z_problem' in colnames:
        z_problem = data[:, np.where(colnames=='z_problem')]
        i_good = np.where(z_problem < zprob_cutoff)[0]
        print "removing %i problematic redshifts" % (data.shape[0]
                                                     - len(i_good))
        data = data[i_good]
    
    if Nmax is None:
        return (data.T)[i_cols]
    else:
        return (data[:Nmax].T)[i_cols]

def read_shear_catalog_asc(filename, cols=None,
                           Nmax=None, remove_z_problem=False,
                           zprob_cutoff = 2):
    """
    cols can be either a list of integers, or a list of names
    """
    F = open(filename)
    line = F.next()

    colnames = []
    
    while line.startswith('#'):
        colnames.append(line.split()[2])
        line = F.next()

    x = np.arange(len(cols))
    i_cols = np.zeros(len(cols),dtype='i')

    if cols is None:
        i_cols = x
    else:
        for i,c in enumerate(cols):
            try:
                i_cols[i] = x[c]
            except:
                if c not in colnames:
                    raise ValueError("'%s' not among column names" % c)
                i_cols[i] = colnames.index(c)

    if remove_z_problem and 'z_problem' in colnames:
        i_zprob = colnames.index('z_problem')
        removed = 0

    X = []
    i = 0

    if Nmax is None:
        Nmax = np.inf

    while True:
        line = map(float, line.split())
        if remove_z_problem and line[i_zprob] >= zprob_cutoff:
            removed += 1
            continue
        try:
            X.append( np.array(line)[i_cols] )
            i += 1
            line = F.next()
        except StopIteration:
            break
        if i==Nmax:
            break

    print "removed %i problematic redshifts" % removed

    return np.asarray(X).T


def read_shear_catalog(filenames, cols=None,
                       Nmax=None, remove_z_problem=False,
                       zprob_cutoff=2):
    X = []

    if type(filenames) == type('string'):
        filenames = [filenames]

    for f in filenames:
        try:
            Xf = read_shear_catalog_npz(f, cols, Nmax,
                                        remove_z_problem, zprob_cutoff)
        except IOError:
            Xf = read_shear_catalog_asc(f, cols, Nmax,
                                        remove_z_problem, zprob_cutoff)
        X.append(Xf)

    return np.concatenate(X, 1)


def get_colnames_npz(filename):
    F = np.load(filename)
    return F['colnames']


def get_colnames_asc(filename):
    F = open(filename)
    line = F.next()

    colnames = []
    
    while line.startswith('#'):
        colnames.append(line.split()[2])
        line = F.next()

    return colnames
    

def get_colnames(filenames):
    X = []

    if type(filenames) == type('string'):
        filenames = [filenames]

    for f in filenames:
        try:
            colnames = get_colnames_npz(f, cols, Nmax)
        except IOError:
            colnames = get_colnames_asc(f, cols, Nmax)
        X.append(Xf)

    return X

    
