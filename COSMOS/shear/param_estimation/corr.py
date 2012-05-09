import numpy as np

def bootstrap_estimate(RA, DEC, gamma, RAbins, DECbins, Nbootstrap=1000):
    N = len(gamma)
    
    gamma_sum = 0
    gamma2_sum = 0

    for i in range(Nbootstrap):
        ind = np.random.randint(N, size=N)

        RAi = RA[ind]
        DECi = DEC[ind]
        gammai = gamma[ind]

        count = np.histogram2d(RAi, DECi, (RAbins, DECbins))
        shear = np.histogram2d(RAi, DECi, (RAbins, DECbins), weights=gammai)

        mask = (count == 0)
        count[mask] = 1

        shear /= count

        gamma_sum += shear
        gamma2_sum += abs(shear) ** 2

    gamma_sum /= Nbootstrap
    gamma2_sum /= Nbootstrap

    d2gamma = gamma2_sum - abs(gamma_sum ** 2)


def correlation_matrix(xi, RA, DEC, gamma, RAbins, DECbins):
    """

    """
    
