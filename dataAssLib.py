import numpy as np
import matplotlib.pyplot as plt

import pyKdV as kdv

#----------------------------------------------------------
def c2r(csp):
    """
    Real to complex (hermitian signal)

        ordering convention
        -------------------
        c_i=a_i+j*b_i
        c_{-i}=(c_i)*

        csp=[c_0, c_1, c_2, ..., c_{N-1}]
        rsp=[a_0, a_1, b_1, a_2, b_2, ..., a_{(N-1)/2+1}, b_{(N-1)/2+1}]
        
        csp.shape=(N+1)
        rsp.shape=(N+1)
    """
    N=len(csp)-1
    rsp=np.zeros(N+1)
    rsp[0]=csp[0].real
    for i in xrange(1,N/2+1):
        rsp[2*i-1]    =2.*csp[i].real
        rsp[2*i]      =2.*csp[i].imag
    return rsp

def r2c(rsp):
    N=len(rsp)-1
    csp=np.zeros(N+1, dtype=complex)
    csp[0]=rsp[0]
    for i in xrange(1,N/2+1):
        csp[i]     =0.5*(rsp[2*i-1]+1j*rsp[2*i])
        csp[N-i+1]   =0.5*(rsp[2*i-1]-1j*rsp[2*i])
    return csp

def r2c_Adj(csp):
    N=len(csp)-1
    rsp=np.zeros(N+1)
    for i in xrange(1, N/2+1):
        rsp[2*i-1]  =csp[i].real
        rsp[2*i]    =-csp[i].imag
    rsp[0]=csp[0].real
    return rsp

def ifft_Adj(x):
    N=len(x)
    xi=np.zeros(N)
    xi=np.fft.fft(x)
    xi=xi*N
    # --- traitement particulier pour le terme a0
    #  clarifier et voir les notes de Pierre
    # et verifier le test d'ajointitude
    #for i in xrange(1, N):
    #    xi[i]=xi[i]*0.5
    return xi

#----| Correlations |---------------------------------------

def fCorr_isoHomo(x, sig):
    return kdv.gauss(x, 0., sig)



def B_sqrt_op(xi, N, var, CTilde_sqrt):
    """
        B_{1/2} operator

        var             :   1D array of variances
                            (diagonal os Sigma matrix)
        CTilde_sqrt     :   1D array of the diagonal
                            of CTilde_sqrt
    """
    #rCTilde=c2r(CTilde_sqrt)
    rCTilde=CTilde_sqrt

    xiR=rCTilde*xi              #   1
    xiC=r2c(xiR)                #   2
    x1=np.fft.ifft(xiC).real    #   3
    return x1*var               #   4

def B_sqrt_op_T(x, N,  var, CTilde_sqrt):
    #rCTilde=c2r(CTilde_sqrt)
    rCTilde=CTilde_sqrt

    x1=x*var                    #   4.T
    xiC=ifft_Adj(x1)        #   3.T
    xiR=r2c_Adj(xiC)            #   2.T
    return rCTilde*xiR      #   1.T

#----| Observations |---------------------------------------


def opObs_exactIdx(N, idxObs):
    """
    s   :   model state
    idxObs :  shape=(nObs,)
    """
    nObs=len(idxObs)
    H=np.zeros(shape=(nObs,N))
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

