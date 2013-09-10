import numpy as np
import random as rnd

class ObservationOpError(Exception):
    pass

def pos2Idx(g, pos):
    if not isinstance(pos, np.ndarray):
        raise ObservationOpError("pos <numpy.ndarray>")
    if pos.ndim<>1:
        raise ObservationOpError("pos.ndim=1")
    N=len(pos)
    idx=np.zeros(N, dtype=int)
    for i in xrange(N):
        idx[i]=np.min(np.where(g.x>=pos[i]))
    return idx

def departure(xi, x_b, var, B_sqrt_op, H, argsH, obs, rCTilde_sqrt):
    x=B_sqrt_op(xi, var, rCTilde_sqrt)+x_b
    Hx=H(x, *argsH)
    return Hx-obs

def opObs_Idx_op(x, g, idxObs):
    nObs=len(idxObs)
    H=np.zeros(shape=(nObs,g.N))
    for i in xrange(nObs):
        H[i, idxObs[i]]=1.
    return np.dot(H,x)

def opObs_Idx_op_T(obs, g, idxObs):
    nObs=len(idxObs)
    H=np.zeros(shape=(nObs,g.N))
    for i in xrange(nObs):
        H[i, idxObs[i]]=1.
    return np.dot(H.T,obs)


def degrad(signal,mu,sigma,seed=0.7349156729):
    ''' 
    Degradation d'un signal vectoriel par l'addition de bruit
    gaussien

    degrad(u,mu,sigma,seed=...)

    u:      signal d'entree a bruiter
    mu: moyenne de la distribution gaussienne de bruit
    sigma:  variance de la distribution

    retourne:
        u_degrad        (u_degrad[i]=u[i]+rnd.gauss(mu, sigma))
    '''
    rnd.seed(seed)
    sig_degrad=signal.copy()
    for i in xrange(signal.size):
        sig_degrad[i]=signal[i]+rnd.gauss(mu, sigma)
    return sig_degrad
