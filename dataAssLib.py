import numpy as np
import matplotlib.pyplot as plt

import pyKdV as kdv
from pyKdV.pseudoSpec1D import conjT, fft2d, ifft2d

#----------------------------------------------------------
def r2c(rsp):
    """
    Real to complex (hermitian signal)

        ordering convention
        -------------------
        c_i=a_i+j*b_i
        c_{-i}=(c_i)*

        csp=[c_0, c_1, c_2, ..., c_{N-1}]
        rsp=[a_0, a_1, b_1, a_2, b_2, ..., a_{(N-1)/2+1}, b_{(N-1)/2+1}]
        
        csp.shape=(N)
        rsp.shape=(N+2)
    """
    N=(len(rsp)-1)/2
    csp=np.zeros(2*N-1, dtype=complex)
    NCmpl=len(csp)
    csp[0]=rsp[0]
    for i in xrange(1,N):
        csp[i]     =0.5*(rsp[2*i-1]-1j*rsp[2*i])
        csp[NCmpl-i]   =0.5*(rsp[2*i-1]+1j*rsp[2*i])
    return csp

def c2r(csp):
    """
    Real to complex (hermitian signal)

        ordering convention
        -------------------
        c_i=a_i+j*b_i
        c_{-i}=(c_i)*

        csp=[c_0, c_1, c_2, ..., c_{N-1}]
        rsp=[a_0, a_1, b_1, a_2, b_2, ..., a_{(N-1)/2+1}, b_{(N-1)/2+1}]
        
        csp.shape=(N)
        rsp.shape=(N+2)
    """
    N=len(csp)
    NReal=(N-1)/2+1
    rsp=np.zeros(2*NReal+1)
    rsp[0]=csp[0].real
    for i in xrange(1,NReal+1):
        rsp[2*i-1]    =2.*csp[i].real
        rsp[2*i]      =-2.*csp[i].imag
    return rsp

#----| Correlations |---------------------------------------

def fCorr_isoHomo(g, sig):
    return kdv.gauss(g.x, 0., sig)

def fSigma_uniform(g, var):
    return np.diag(var*np.ones(g.N))


def B_sqrt_isoHomo(g, var, corr):
    Sigma=np.diag(var)
    tfC_sqrt=np.diag(np.sqrt(corr))
    C_sqrt=np.fft.ifft(tfC_sqrt)
    #return np.dot(Sigma, np.fft.ifft(tfCDemi))
    return np.dot(Sigma, C_sqrt)

def B_sqrt_inv_isoHomo(g, var, corr):
    Sigma_inv=np.diag(var**-1)
    tfC_sqrt_inv=np.diag(np.sqrt(corr)**-1)
    C_sqrt_inv=np.fft.ifft(tfC_sqrt_inv)
    return np.dot(C_sqrt_inv, Sigma_inv)
    #return g.N*np.dot(conjT(np.fft.ifft(conjT(tfC_sqrt_inv))).real,
    #                    Sigma_inv)
    #return g.N*np.dot(conjT(np.fft.ifft(conjT(tfCDemi_inv))),
    #                    Sigma_inv)
    

def spCorr_isoHomo(corr):
    return np.diag(np.sqrt(corr))


def B_sqrt_op(xi, N, var, rCTilde):
    rXiTilde=np.dot(rCTilde, xi)
    xiTilde=r2c(rXiTilde)
    xi2=np.fft.ifft(xiTilde).real
    Sigma=np.diag(var)
    return np.dot(xi2, Sigma)

def B_sqrt_op_T(x, N,  var, rCTilde):
    Sigma=np.diag(var)
    xTmp=np.dot(x, Sigma)
    xTilde=np.fft.fft(xTmp.T)*N
    rXTilde=c2r(xTilde)
    xi=np.dot(rCTilde, rXTilde)
    for i in xrange(1,g.N):
        xi[i]=0.5*xi[i]
    return xi

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

def costFunc(xi, d, R_inv, B_sqrt_op, H, N, var, rCTilde):
    J_xi=np.dot(xi, xi).real
    J_o=0.5*np.dot(d,np.dot(R_inv,d))
    return J_xi+J_o

def gradCostFunc(xi, d, R_inv, B_sqrt_op, H, N, var, rCTilde):
    return xi+B_sqrt_op_T(np.dot(H.T,np.dot(R_inv, d)), N, var, rCTilde )

