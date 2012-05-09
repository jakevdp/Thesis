#-append the correct paths for importing Lens3D modules--------------
import sys, os
Lens3D_path = os.environ.get('LENS3D_PATH')
if Lens3D_path is None:
    raise ValueError, 'must set environment variable LENS3D_PATH'
sys.path.append(Lens3D_path)
sys.path.append(os.path.join(Lens3D_path,'3D_shear_generation'))
#--------------------------------------------------------------------

from Lens3D import *
from Reconstructions import *
from time import time

def save_mode_profiles(filtertype='trans',
                       alpha = 0.1,
                       N = 16,
                       z_gamma = numpy.linspace(0.08,2.0,25),
                       z_delta = numpy.linspace(0.1,2.0,20),
                       theta_min = 0,
                       theta_max = None,
                       border_size = None,
                       border_noise = 1E3,
                       Ngal = 70,
                       z0 = 0.57,
                       sig = 0.3):
    if theta_max is None:
        theta_max = N-1
    if border_size is None:
        border_size = N/16
    
    if filtertype not in ['aitken','svd','trans','rad']:
        raise ValueError, "type must be one of ['aitken','svd','trans','rad']"
    

    #calculate all the transformation matrices
    P_gk = construct_P_gk( theta_min, theta_max, N,
                           theta_min, theta_max, N,
                           z_gamma )
    
    P_kd = construct_P_kd(N,N,z_gamma,z_delta)

    N_los = compute_N_los(z_gamma,z0,sig,Ngal)
    N_angular = compute_N_angular(N,border_size,border_noise)
        
    U,S,VH = compute_SVD(P_gk,
                         P_kd,
                         N_angular,
                         N_los)
    
    if filtertype=='aitken':
        rowfile = 'rows_aitken.txt'
        colfile = 'cols_aitken.txt'
        rows = abs(S.data_)**-1
        cols = rows
        
    else:
        rowfile = 'rows_%s_%.2g.txt' % (filtertype,alpha)
        colfile = 'cols_%s_%.2g.txt' % (filtertype,alpha)
        if filtertype=='svd':
            N_gg = create_N_gg_tensor(N_los,N_angular)
            
            R = create_SVD_Rmatrix(P_gk,
                                   P_kd,
                                   N_angular,
                                   N_los,
                                   alpha)
        else:
            N_gg = create_N_gg_constborder(N_los,
                                           N, border_size, border_noise)
            if filtertype == 'trans':
                S_dd = construct_angular_S_dd( theta_min, theta_max, N,
                                               theta_min, theta_max, N,
                                               z_delta)
            else:
                S_dd = construct_radial_S_dd(N, N, z_delta,
                                             (theta_max-theta_min)*1./(N-1) )
        
            R = create_WF_Rmatrix(P_gk,
                                  P_kd,
                                  S_dd,
                                  N_gg,
                                  alpha,
                                  verbose = 0)
            
        N_gg_1_2 = N_gg.copy()
        N_gg_1_2.data_ **= 0.5
        
        V = VH.H
        UH = U.H
        RH = R.H
        
        rows = numpy.zeros(U.shape[1])
        cols = numpy.zeros(U.shape[1])
        
        t0 = time()
        
        print UH.shape
        
        for i in range(U.shape[1]):
            if i%100 == 0:
                print "finished %i out of %i rows:" % (i,U.shape[1]),
                print " time elapsed: %.2g sec" % (time()-t0)
            vi = numpy.zeros(U.shape[1],dtype=complex)
            vi[i] = 1.0
            vi = U.view_as_Lens3D_vec(vi)
            row = U.matvec(vi)
            row = N_gg_1_2.matvec(row)
            row = R.matvec(row)
            row = VH.matvec(row)
            rows[i] = numpy.linalg.norm(V.view_as_normal_vec(row))

        for i in range(V.shape[1]):
            if i%100 == 0:
                print "finished %i out of %i cols:" % (i,V.shape[1]),
                print " time elapsed: %.2g sec" % (time()-t0)
            vi = numpy.zeros(V.shape[1],dtype=complex)
            vi[i] = 1.0
            vi = V.view_as_Lens3D_vec(vi)
            col = V.matvec(vi)
            col = RH.matvec(col)
            col = N_gg_1_2.matvec(col)
            col = UH.matvec(col)
            cols[i] = numpy.linalg.norm(U.view_as_normal_vec(col))

    OF = open(rowfile,'w')
    print "writing rows to %s" % rowfile
    for row in rows:
        OF.write('%.6g\n' % row)
    OF.close()
    
    OF = open(colfile,'w')
    print "writing cols to %s" % colfile
    for col in cols:
        OF.write('%.6g\n' % col)
    OF.close()
    
if __name__ == '__main__':
    for alpha in [0.1,0.001]:
        for filtertype in ['aitken','svd','trans','rad']:
            v = save_mode_profiles(filtertype,
                                   alpha = alpha,
                                   N = 16,
                                   border_size = 1,
                                   z_gamma = numpy.linspace(0.08,2.0,25),
                                   z_delta = numpy.linspace(0.1,2.0,20)  )
