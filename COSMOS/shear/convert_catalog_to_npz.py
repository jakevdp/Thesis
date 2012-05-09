import numpy as np

def convert_to_npz(filename,filename_out):
    data = np.loadtxt(filename)
    F = open(filename)
    line = F.next()
    colnames = []
    while line.startswith('#'):
        colnames.append(line.split()[2])
        line = F.next()
    colnames = np.asarray(colnames)

    np.savez(filename_out, 
             colnames=colnames, data=data)

if __name__ == '__main__':
    from params import BRIGHT_CAT, BRIGHT_CAT_NPZ, FAINT_CAT, FAINT_CAT_NPZ
    convert_to_npz(BRIGHT_CAT, BRIGHT_CAT_NPZ)
    convert_to_npz(FAINT_CAT, FAINT_CAT_NPZ)
