import numpy as np
import matplotlib.pyplot as plt

import pyKdV as kdv
from pyKdV.pseudoSpec1D import conjT, fft2d, ifft2d

#----| Correlations |---------------------------------------

def fCorr_isoHomo(g, sig):
    return kdv.gauss(g.x, 0., sig)

def fSigma_uniform(g, var):
    return np.diag(var*np.ones(g.N))


def B_sqrt_isoHomo(g, var, corr):
    Sigma=np.diag(var)
    tfCDemi=np.diag(np.sqrt(corr))
    return np.dot(Sigma, np.fft.ifft(tfCDemi))

def B_sqrt_inv_isoHomo(g, var, corr):
    Sigma_inv=np.diag(var**-1)
    tfCDemi_inv=np.diag(np.sqrt(corr)**-1)
    return g.N*np.dot(conjT(np.fft.ifft(conjT(tfCDemi_inv))), Sigma_inv)
    
#----| Observations |---------------------------------------


def opObs_exactIdx(g, idxObs):
    """
    s   :   model state
    idxObs :  shape=(nObs,)
    """
    nObs=len(idxObs)
    H=np.zeros(shape=(nObs,g.N))
    for i in xrange(nObs):
        H[i, idxObs[i]]=1.
    return H


def obsCovar_independant(idxObs, var):
    nObs=len(idxObs)
    return np.diag(var)

def degrad(signal,mu,sigma,seed=0.7349156729):
    rnd.seed(seed)
    signal_degrad=signal.copy()
    for i in xrange(signal.size):
        signal_degrad[i]=signal[i]+rnd.gauss(mu, sigma)
    return signal_degrad

#----| Cost function |--------------------------------------

def costFunc(xi, d, R_inv, B_sqrt, H):
    return np.dot(xi.conj(), xi)+0.5*np.dot(d.conj(),np.dot(R_inv,d))

def gradCostFunc(xi, d, R_inv, B_sqrt, H):
    return xi+np.dot(conjT(B_sqrt),np.dot(conjT(H),np.dot(R_inv, d)))

