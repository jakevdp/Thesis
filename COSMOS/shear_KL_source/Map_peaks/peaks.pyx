import numpy
cimport numpy

DTYPE = numpy.float64
ctypedef numpy.float64_t DTYPE_t

DTYPEC = numpy.complex128
ctypedef numpy.complex128_t DTYPEC_t

#define inline max and min functions
cdef inline DTYPE_t f64_max(DTYPE_t a, DTYPE_t b): return a if a >= b else b
cdef inline DTYPE_t f64_min(DTYPE_t a, DTYPE_t b): return a if a <= b else b

cdef inline int int_max(int a, int b): return a if a >= b else b
cdef inline int int_min(int a, int b): return a if a <= b else b

#get c math functions
cdef extern from "math.h":
    DTYPE_t tanh(DTYPE_t theta)
    DTYPE_t sqrt(DTYPE_t x)
    DTYPE_t exp(DTYPE_t x)


def find_peaks(numpy.ndarray[DTYPE_t, ndim=2] A not None,
               DTYPE_t r=2.0):
    """
    A is a 2D array of real values
    r is the radius (in pixels) of the peak finder
    """
    cdef int imax = A.shape[0]
    cdef int jmax = A.shape[1]
    cdef int ri = int( numpy.floor(r) )

    cdef unsigned int i,j,i_loc,j_loc,imin_loc,imax_loc,jmin_loc,jmax_loc

    cdef DTYPE_t Amax

    cdef int npeaks = int( imax*jmax / (numpy.pi*r*r) )

    cdef numpy.ndarray[unsigned char,ndim=2] peaks = \
         numpy.zeros((imax,jmax),
                     dtype=numpy.uint8)
    cdef DTYPE_t r2 = r*r

    for i in range(imax):
        imin_loc = int_max(0,i-ri)
        imax_loc = int_min(imax,i+ri+1)
        for j in range(jmax):
            jmin_loc = int_max(0,j-ri)
            jmax_loc = int_min(jmax,j+ri+1)

            Amax = -numpy.inf

            for i_loc in range(imin_loc,imax_loc):
                for j_loc in range(jmin_loc,jmax_loc):
                    if (i_loc-i)*(i_loc-i)+(j_loc-j)*(j_loc-j) > r2:
                        continue
                    Amax = f64_max(A[i_loc,j_loc],Amax)

            if A[i,j] == Amax:
                peaks[i,j] = 1

    return peaks

cdef Q_NFW(DTYPE_t x,
           DTYPE_t xc):
    cdef DTYPE_t r = x/xc
    return 1. / (1. + exp(6.-150.*x) + exp(-47.0+50.*x) ) * tanh(r) / r

def pyQ_NFW(x,xc):
    r = x*1./xc
    return 1. / (1. + numpy.exp(6.-150.*x) + numpy.exp(-47.0+50.*x) ) * numpy.tanh(r) / r

def Map_map(numpy.ndarray[DTYPEC_t,ndim=2] gamma not None,
            numpy.ndarray[DTYPEC_t,ndim=2] pos not None,
            numpy.ndarray[DTYPEC_t,ndim=2] posM not None,
            DTYPE_t rmax,
            DTYPE_t xc=0.15 ):
    assert (gamma.shape[0] == pos.shape[0]) and (gamma.shape[1]==pos.shape[1])

    cdef int igmax = pos.shape[0]
    cdef int jgmax = pos.shape[1]

    cdef int iMmax = posM.shape[0]
    cdef int jMmax = posM.shape[1]

    cdef numpy.ndarray[DTYPE_t,ndim=2] Map_E = numpy.zeros((iMmax,jMmax),
                                                           dtype=DTYPE)
    cdef numpy.ndarray[DTYPE_t,ndim=2] Map_B = numpy.zeros((iMmax,jMmax),
                                                           dtype=DTYPE)

    cdef numpy.ndarray[DTYPE_t,ndim=2] gamma_r = gamma.real
    cdef numpy.ndarray[DTYPE_t,ndim=2] gamma_i = gamma.imag
    cdef numpy.ndarray[DTYPE_t,ndim=2] pos_x = pos.real
    cdef numpy.ndarray[DTYPE_t,ndim=2] pos_y = pos.imag
    cdef numpy.ndarray[DTYPE_t,ndim=2] posM_x = posM.real
    cdef numpy.ndarray[DTYPE_t,ndim=2] posM_y = posM.imag

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



def Map_map_normed(numpy.ndarray[DTYPEC_t,ndim=2] gamma not None,
                   numpy.ndarray[DTYPEC_t,ndim=2] pos not None,
                   numpy.ndarray[DTYPEC_t,ndim=2] posM not None,
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

    cdef numpy.ndarray[DTYPE_t,ndim=2] Map_E = numpy.zeros((iMmax,jMmax),
                                                           dtype=DTYPE)
    cdef numpy.ndarray[DTYPE_t,ndim=2] Map_B = numpy.zeros((iMmax,jMmax),
                                                           dtype=DTYPE)

    cdef numpy.ndarray[DTYPE_t,ndim=2] gamma_r = gamma.real
    cdef numpy.ndarray[DTYPE_t,ndim=2] gamma_i = gamma.imag
    cdef numpy.ndarray[DTYPE_t,ndim=2] pos_x = pos.real
    cdef numpy.ndarray[DTYPE_t,ndim=2] pos_y = pos.imag
    cdef numpy.ndarray[DTYPE_t,ndim=2] posM_x = posM.real
    cdef numpy.ndarray[DTYPE_t,ndim=2] posM_y = posM.imag

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


def Map_map_noise(numpy.ndarray[DTYPEC_t,ndim=2] gamma not None,
                  numpy.ndarray[DTYPE_t,ndim=2] noise not None,
                  numpy.ndarray[DTYPEC_t,ndim=2] pos not None,
                  numpy.ndarray[DTYPEC_t,ndim=2] posM not None,
                  DTYPE_t rmax,
                  DTYPE_t xc=0.15 ):
    assert (gamma.shape[0] == pos.shape[0]) and (gamma.shape[1]==pos.shape[1])
    assert (noise.shape[0] == pos.shape[0]) and (noise.shape[1]==pos.shape[1])

    cdef int igmax = pos.shape[0]
    cdef int jgmax = pos.shape[1]

    cdef int iMmax = posM.shape[0]
    cdef int jMmax = posM.shape[1]

    cdef numpy.ndarray[DTYPE_t,ndim=2] Map_E = numpy.zeros((iMmax,jMmax),
                                                           dtype=DTYPE)
    cdef numpy.ndarray[DTYPE_t,ndim=2] Map_B = numpy.zeros((iMmax,jMmax),
                                                           dtype=DTYPE)
    cdef numpy.ndarray[DTYPE_t,ndim=2] Map_noise = numpy.zeros((iMmax,jMmax),
                                                               dtype=DTYPE)

    cdef numpy.ndarray[DTYPE_t,ndim=2] gamma_r = gamma.real
    cdef numpy.ndarray[DTYPE_t,ndim=2] gamma_i = gamma.imag
    cdef numpy.ndarray[DTYPE_t,ndim=2] pos_x = pos.real
    cdef numpy.ndarray[DTYPE_t,ndim=2] pos_y = pos.imag
    cdef numpy.ndarray[DTYPE_t,ndim=2] posM_x = posM.real
    cdef numpy.ndarray[DTYPE_t,ndim=2] posM_y = posM.imag

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
                Map_noise[iM,jM] = numpy.sqrt(sumN)/sumQ
    return Map_E, Map_B, Map_noise



def Map_map_noise_normed(numpy.ndarray[DTYPEC_t,ndim=2] gamma not None,
                         numpy.ndarray[DTYPE_t,ndim=2] noise not None,
                         numpy.ndarray[DTYPEC_t,ndim=2] pos not None,
                         numpy.ndarray[DTYPEC_t,ndim=2] posM not None,
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

    cdef numpy.ndarray[DTYPE_t,ndim=2] Map_E = numpy.zeros((iMmax,jMmax),
                                                           dtype=DTYPE)
    cdef numpy.ndarray[DTYPE_t,ndim=2] Map_B = numpy.zeros((iMmax,jMmax),
                                                           dtype=DTYPE)
    cdef numpy.ndarray[DTYPE_t,ndim=2] Map_noise = numpy.zeros((iMmax,jMmax),
                                                               dtype=DTYPE)

    cdef numpy.ndarray[DTYPE_t,ndim=2] gamma_r = gamma.real
    cdef numpy.ndarray[DTYPE_t,ndim=2] gamma_i = gamma.imag
    cdef numpy.ndarray[DTYPE_t,ndim=2] pos_x = pos.real
    cdef numpy.ndarray[DTYPE_t,ndim=2] pos_y = pos.imag
    cdef numpy.ndarray[DTYPE_t,ndim=2] posM_x = posM.real
    cdef numpy.ndarray[DTYPE_t,ndim=2] posM_y = posM.imag

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
                Map_noise[iM,jM] = numpy.sqrt(sumN)/sumQ
    return Map_E, Map_B, Map_noise
