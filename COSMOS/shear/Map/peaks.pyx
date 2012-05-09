import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

DTYPEC = np.complex128
ctypedef np.complex128_t DTYPEC_t

#define inline max and min functions
cdef inline DTYPE_t dtype_max(DTYPE_t a, DTYPE_t b): return a if a >= b else b
cdef inline DTYPE_t dtype_min(DTYPE_t a, DTYPE_t b): return a if a <= b else b

cdef inline int int_max(int a, int b): return a if a >= b else b
cdef inline int int_min(int a, int b): return a if a <= b else b

#get c math functions
cdef extern from "math.h":
    DTYPE_t tanh(DTYPE_t theta)
    DTYPE_t sqrt(DTYPE_t x)
    DTYPE_t exp(DTYPE_t x)

cdef Q_NFW(DTYPE_t x,
           DTYPE_t xc):
    cdef DTYPE_t r = x/xc
    return 1. / (1. + exp(6.-150.*x) + exp(-47.0+50.*x) ) * tanh(r) / r

def pyQ_NFW(x,xc):
    r = x*1./xc
    return 1. / (1. + np.exp(6.-150.*x) + np.exp(-47.0+50.*x) ) * np.tanh(r) / r

def Aperture_Mass(np.ndarray[DTYPEC_t,ndim=1] gamma not None,
                  np.ndarray[DTYPEC_t,ndim=1] pos not None,
                  np.ndarray[DTYPEC_t,ndim=1] posM not None,
                  DTYPE_t rmax = 5.6,
                  DTYPE_t xc = 0.15):
    assert gamma.shape[0] == pos.shape[0]

    cdef int N_in = gamma.shape[0]
    cdef int N_out = posM.shape[0]

    cdef np.ndarray[DTYPEC_t,ndim=1] Map = np.zeros(N_out, dtype=DTYPEC)

    cdef np.ndarray[DTYPE_t,ndim=1] gamma_r = gamma.real
    cdef np.ndarray[DTYPE_t,ndim=1] gamma_i = gamma.imag
    cdef np.ndarray[DTYPE_t,ndim=1] pos_x = pos.real
    cdef np.ndarray[DTYPE_t,ndim=1] pos_y = pos.imag
    cdef np.ndarray[DTYPE_t,ndim=1] posM_x = posM.real
    cdef np.ndarray[DTYPE_t,ndim=1] posM_y = posM.imag

    cdef int i, j

    cdef DTYPE_t r, x, cos1, sin1, cos2, sin2, sep_x, sep_y
    cdef DTYPE_t gamma_T, gamma_X, Q, sumQ, sumE, sumB

    for i in range(N_out):
        sumE = 0
        sumB = 0
        sumQ = 0
        for j in range(N_in):
            sep_x = pos_x[j]-posM_x[i]
            sep_y = pos_y[j]-posM_y[i]
            r = sqrt(sep_x*sep_x + sep_y*sep_y)
            x = r/rmax
            
            if x==0: continue
            if x>1.8: continue
            #empirically, I found r/rmax > 1.8 gives contributions
            # below double precision

            cos1 = sep_x/r
            sin1 = sep_y/r
            
            cos2 = cos1*cos1-sin1*sin1
            sin2 = 2*cos1*sin1
            
            gamma_T = - gamma_r[j] * cos2 - gamma_i[j] * sin2
            gamma_X = gamma_r[j] * sin2 - gamma_i[j] * cos2
    
            Q = Q_NFW(x,xc)

            sumQ += Q
            sumE += Q*gamma_T
            sumB += Q*gamma_X
            if sumQ==0:
                Map[i] = 0
            else:
                Map[i] = sumE/sumQ + 1j*sumB/sumQ
    return Map
    

def find_peaks(np.ndarray[DTYPE_t, ndim=2] A not None,
               DTYPE_t r=2.0):
    """
    A is a 2D array of real values
    r is the radius (in pixels) of the peak finder
    """
    cdef int imax = A.shape[0]
    cdef int jmax = A.shape[1]
    cdef int ri = int( np.floor(r) )

    cdef unsigned int i,j,i_loc,j_loc,imin_loc,imax_loc,jmin_loc,jmax_loc

    cdef DTYPE_t Amax

    cdef int npeaks = int( imax*jmax / (np.pi*r*r) )

    cdef np.ndarray[unsigned char,ndim=2] peaks = \
         np.zeros((imax,jmax),
                     dtype=np.uint8)
    cdef DTYPE_t r2 = r*r

    for i in range(imax):
        imin_loc = int_max(0,i-ri)
        imax_loc = int_min(imax,i+ri+1)
        for j in range(jmax):
            jmin_loc = int_max(0,j-ri)
            jmax_loc = int_min(jmax,j+ri+1)

            Amax = -np.inf

            for i_loc in range(imin_loc,imax_loc):
                for j_loc in range(jmin_loc,jmax_loc):
                    if (i_loc-i)*(i_loc-i)+(j_loc-j)*(j_loc-j) > r2:
                        continue
                    Amax = dtype_max(A[i_loc,j_loc],Amax)

            if A[i,j] == Amax:
                peaks[i,j] = 1

    return peaks

def Map_map(np.ndarray[DTYPEC_t,ndim=2] gamma not None,
            np.ndarray[DTYPEC_t,ndim=2] pos not None,
            np.ndarray[DTYPEC_t,ndim=2] posM not None,
            DTYPE_t rmax,
            DTYPE_t xc=0.15 ):
    assert (gamma.shape[0] == pos.shape[0]) and (gamma.shape[1]==pos.shape[1])

    cdef int igmax = pos.shape[0]
    cdef int jgmax = pos.shape[1]

    cdef int iMmax = posM.shape[0]
    cdef int jMmax = posM.shape[1]

    cdef np.ndarray[DTYPE_t,ndim=2] Map_E = np.zeros((iMmax,jMmax),
                                                           dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=2] Map_B = np.zeros((iMmax,jMmax),
                                                           dtype=DTYPE)

    cdef np.ndarray[DTYPE_t,ndim=2] gamma_r = gamma.real
    cdef np.ndarray[DTYPE_t,ndim=2] gamma_i = gamma.imag
    cdef np.ndarray[DTYPE_t,ndim=2] pos_x = pos.real
    cdef np.ndarray[DTYPE_t,ndim=2] pos_y = pos.imag
    cdef np.ndarray[DTYPE_t,ndim=2] posM_x = posM.real
    cdef np.ndarray[DTYPE_t,ndim=2] posM_y = posM.imag

    cdef int iM,jM,ig,jg

    cdef DTYPE_t r, x, cos1, sin1, cos2, sin2, sep_x, sep_y
    cdef DTYPE_t gamma_T, gamma_X, Q, sumQ, sumE, sumB

    for iM in range(iMmax):
        for jM in range(jMmax):
            sumE = 0
            sumB = 0
            sumQ = 0
            for ig in range(igmax):
                for jg in range(jgmax):
                    sep_x = pos_x[ig,jg]-posM_x[iM,jM]
                    sep_y = pos_y[ig,jg]-posM_y[iM,jM]
                    r = sqrt(sep_x*sep_x + sep_y*sep_y)
                    x = r/rmax

                    if x==0: continue
                    if x>1.8: continue
                    #empirically, I found r/rmax > 1.8 gives contributions
                    # below double precision

                    cos1 = sep_x/r
                    sin1 = sep_y/r

                    cos2 = cos1*cos1-sin1*sin1
                    sin2 = 2*cos1*sin1

                    gamma_T = - gamma_r[ig,jg] * cos2 \
                              - gamma_i[ig,jg] * sin2
                    gamma_X = gamma_r[ig,jg] * sin2 \
                              - gamma_i[ig,jg] * cos2
    
                    Q = Q_NFW(x,xc)

                    sumQ += Q
                    sumE += Q*gamma_T
                    sumB += Q*gamma_X
            if sumQ==0:
                Map_E[iM,jM] = 0
                Map_B[iM,jM] = 0
            else:
                Map_E[iM,jM] = sumE/sumQ
                Map_B[iM,jM] = sumB/sumQ
    return Map_E, Map_B



def Map_map_normed(np.ndarray[DTYPEC_t,ndim=2] gamma not None,
                   np.ndarray[DTYPEC_t,ndim=2] pos not None,
                   np.ndarray[DTYPEC_t,ndim=2] posM not None,
                   DTYPE_t rmax,
                   DTYPE_t xc=0.15 ):
    """
    Normalize over areas where gamma = 0
    """
    assert (gamma.shape[0] == pos.shape[0]) and (gamma.shape[1]==pos.shape[1])

    cdef int igmax = pos.shape[0]
    cdef int jgmax = pos.shape[1]

    cdef int iMmax = posM.shape[0]
    cdef int jMmax = posM.shape[1]

    cdef np.ndarray[DTYPE_t,ndim=2] Map_E = np.zeros((iMmax,jMmax),
                                                           dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=2] Map_B = np.zeros((iMmax,jMmax),
                                                           dtype=DTYPE)

    cdef np.ndarray[DTYPE_t,ndim=2] gamma_r = gamma.real
    cdef np.ndarray[DTYPE_t,ndim=2] gamma_i = gamma.imag
    cdef np.ndarray[DTYPE_t,ndim=2] pos_x = pos.real
    cdef np.ndarray[DTYPE_t,ndim=2] pos_y = pos.imag
    cdef np.ndarray[DTYPE_t,ndim=2] posM_x = posM.real
    cdef np.ndarray[DTYPE_t,ndim=2] posM_y = posM.imag

    cdef int iM,jM,ig,jg

    cdef DTYPE_t r, x, cos1, sin1, cos2, sin2, sep_x, sep_y
    cdef DTYPE_t gamma_T, gamma_X, Q, sumQ, sumE, sumB

    for iM in range(iMmax):
        for jM in range(jMmax):
            sumE = 0
            sumB = 0
            sumQ = 0
            for ig in range(igmax):
                for jg in range(jgmax):
                    sep_x = pos_x[ig,jg]-posM_x[iM,jM]
                    sep_y = pos_y[ig,jg]-posM_y[iM,jM]
                    r = sqrt(sep_x*sep_x + sep_y*sep_y)
                    x = r/rmax

                    if x==0: continue
                    if x>1.8: continue
                    #empirically, I found r/rmax > 1.8 gives contributions
                    # below double precision

                    cos1 = sep_x/r
                    sin1 = sep_y/r

                    cos2 = cos1*cos1-sin1*sin1
                    sin2 = 2*cos1*sin1

                    gamma_T = - gamma_r[ig,jg] * cos2 \
                              - gamma_i[ig,jg] * sin2
                    gamma_X = gamma_r[ig,jg] * sin2 \
                              - gamma_i[ig,jg] * cos2
    
                    Q = Q_NFW(x,xc)

                    if (gamma_T == 0) and (gamma_X==0):
                        pass
                    else:
                        sumQ += Q
                        sumE += Q*gamma_T
                        sumB += Q*gamma_X
            if sumQ==0:
                Map_E[iM,jM] = 0
                Map_B[iM,jM] = 0
            else:
                Map_E[iM,jM] = sumE/sumQ
                Map_B[iM,jM] = sumB/sumQ
    return Map_E, Map_B


def Map_map_noise(np.ndarray[DTYPEC_t,ndim=2] gamma not None,
                  np.ndarray[DTYPE_t,ndim=2] noise not None,
                  np.ndarray[DTYPEC_t,ndim=2] pos not None,
                  np.ndarray[DTYPEC_t,ndim=2] posM not None,
                  DTYPE_t rmax,
                  DTYPE_t xc=0.15 ):
    assert (gamma.shape[0] == pos.shape[0]) and (gamma.shape[1]==pos.shape[1])
    assert (noise.shape[0] == pos.shape[0]) and (noise.shape[1]==pos.shape[1])

    cdef int igmax = pos.shape[0]
    cdef int jgmax = pos.shape[1]

    cdef int iMmax = posM.shape[0]
    cdef int jMmax = posM.shape[1]

    cdef np.ndarray[DTYPE_t,ndim=2] Map_E = np.zeros((iMmax,jMmax),
                                                           dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=2] Map_B = np.zeros((iMmax,jMmax),
                                                           dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=2] Map_noise = np.zeros((iMmax,jMmax),
                                                               dtype=DTYPE)

    cdef np.ndarray[DTYPE_t,ndim=2] gamma_r = gamma.real
    cdef np.ndarray[DTYPE_t,ndim=2] gamma_i = gamma.imag
    cdef np.ndarray[DTYPE_t,ndim=2] pos_x = pos.real
    cdef np.ndarray[DTYPE_t,ndim=2] pos_y = pos.imag
    cdef np.ndarray[DTYPE_t,ndim=2] posM_x = posM.real
    cdef np.ndarray[DTYPE_t,ndim=2] posM_y = posM.imag

    cdef int iM,jM,ig,jg

    cdef DTYPE_t r, x, cos1, sin1, cos2, sin2, sep_x, sep_y
    cdef DTYPE_t gamma_T, gamma_X, Q, sumQ, sumE, sumB, sumN

    for iM in range(iMmax):
        for jM in range(jMmax):
            sumE = 0
            sumB = 0
            sumN = 0
            sumQ = 0
            for ig in range(igmax):
                for jg in range(jgmax):
                    sep_x = pos_x[ig,jg]-posM_x[iM,jM]
                    sep_y = pos_y[ig,jg]-posM_y[iM,jM]
                    r = sqrt(sep_x*sep_x + sep_y*sep_y)
                    x = r/rmax

                    if x==0: continue
                    if x>1.8: continue
                    #empirically, I found r/rmax > 1.8 gives contributions
                    # below double precision

                    cos1 = sep_x/r
                    sin1 = sep_y/r

                    cos2 = cos1*cos1-sin1*sin1
                    sin2 = 2*cos1*sin1

                    gamma_T = - gamma_r[ig,jg] * cos2 \
                              - gamma_i[ig,jg] * sin2
                    gamma_X = gamma_r[ig,jg] * sin2 \
                              - gamma_i[ig,jg] * cos2
    
                    Q = Q_NFW(x,xc)

                    sumQ += Q
                    sumE += Q*gamma_T
                    sumB += Q*gamma_X
                    sumN += 0.5 * (Q*noise[ig,jg])**2
            if sumQ==0:
                Map_E[iM,jM] = 0
                Map_B[iM,jM] = 0
                Map_noise[iM,jM] = 0
            else:
                Map_E[iM,jM] = sumE/sumQ
                Map_B[iM,jM] = sumB/sumQ
                Map_noise[iM,jM] = np.sqrt(sumN)/sumQ
    return Map_E, Map_B, Map_noise



def Map_map_noise_normed(np.ndarray[DTYPEC_t,ndim=2] gamma not None,
                         np.ndarray[DTYPE_t,ndim=2] noise not None,
                         np.ndarray[DTYPEC_t,ndim=2] pos not None,
                         np.ndarray[DTYPEC_t,ndim=2] posM not None,
                         DTYPE_t rmax,
                         DTYPE_t xc=0.15 ):
    """
    Normalize over areas where gamma = 0
    """
    assert (gamma.shape[0] == pos.shape[0]) and (gamma.shape[1]==pos.shape[1])
    assert (noise.shape[0] == pos.shape[0]) and (noise.shape[1]==pos.shape[1])

    cdef int igmax = pos.shape[0]
    cdef int jgmax = pos.shape[1]

    cdef int iMmax = posM.shape[0]
    cdef int jMmax = posM.shape[1]

    cdef np.ndarray[DTYPE_t,ndim=2] Map_E = np.zeros((iMmax,jMmax),
                                                           dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=2] Map_B = np.zeros((iMmax,jMmax),
                                                           dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=2] Map_noise = np.zeros((iMmax,jMmax),
                                                               dtype=DTYPE)

    cdef np.ndarray[DTYPE_t,ndim=2] gamma_r = gamma.real
    cdef np.ndarray[DTYPE_t,ndim=2] gamma_i = gamma.imag
    cdef np.ndarray[DTYPE_t,ndim=2] pos_x = pos.real
    cdef np.ndarray[DTYPE_t,ndim=2] pos_y = pos.imag
    cdef np.ndarray[DTYPE_t,ndim=2] posM_x = posM.real
    cdef np.ndarray[DTYPE_t,ndim=2] posM_y = posM.imag

    cdef int iM,jM,ig,jg

    cdef DTYPE_t r, x, cos1, sin1, cos2, sin2, sep_x, sep_y
    cdef DTYPE_t gamma_T, gamma_X, Q, sumQ, sumE, sumB, sumN

    for iM in range(iMmax):
        for jM in range(jMmax):
            sumE = 0
            sumB = 0
            sumN = 0
            sumQ = 0
            for ig in range(igmax):
                for jg in range(jgmax):
                    sep_x = pos_x[ig,jg]-posM_x[iM,jM]
                    sep_y = pos_y[ig,jg]-posM_y[iM,jM]
                    r = sqrt(sep_x*sep_x + sep_y*sep_y)
                    x = r/rmax

                    if x==0: continue
                    if x>1.8: continue
                    #empirically, I found r/rmax > 1.8 gives contributions
                    # below double precision

                    cos1 = sep_x/r
                    sin1 = sep_y/r

                    cos2 = cos1*cos1-sin1*sin1
                    sin2 = 2*cos1*sin1

                    gamma_T = - gamma_r[ig,jg] * cos2 \
                              - gamma_i[ig,jg] * sin2
                    gamma_X = gamma_r[ig,jg] * sin2 \
                              - gamma_i[ig,jg] * cos2
    
                    Q = Q_NFW(x,xc)

                    if (gamma_T == 0) and (gamma_X==0):
                        pass
                    else:
                        sumQ += Q
                        sumE += Q*gamma_T
                        sumB += Q*gamma_X
                        sumN += 0.5 * (Q*noise[ig,jg])**2
            if sumQ==0:
                Map_E[iM,jM] = 0
                Map_B[iM,jM] = 0
                Map_noise[iM,jM] = 0
            else:
                Map_E[iM,jM] = sumE/sumQ
                Map_B[iM,jM] = sumB/sumQ
                Map_noise[iM,jM] = np.sqrt(sumN)/sumQ
    return Map_E, Map_B, Map_noise
