import numpy as np
import matplotlib.pyplot as plt
import random as rnd

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
        #rsp[2*i]    =-csp[i].imag
        rsp[2*i]    =csp[i].imag
        # attention, le produit dans l'espace complexe
        # demande la conjugaison
    rsp[0]=csp[0].real
    return rsp

def ifft_Adj(x):
    N=len(x)
    xi=np.zeros(N)
    xi=np.fft.fft(x)
    xi=xi/N
    return xi

#----| Correlations |---------------------------------------

def fCorr_isoHomo(x, sig):
    return kdv.gauss(x, 0., sig)

def rCTilde_sqrt_isoHomo(g, fCorr):
    """
        Construit la matrice CTilde_sqrt isotrope et homogene
        dans la base 'r'.


        Comme celle-ci est diagonale, on la representente comme
        un vecteur (sa diagonale).

        [c_0, c_1.real, c_1.imag, c_2.real, c_2.imag, ...]

        <!> Attention, elle reste un tenseur d'ordre 2,
            il faudra cependant etre coherent dans l'application
            des transformations et changement de base (de 'r' a 'c')
            des operateurs LCL* et non LC...
            les manipulations qui suivent resultent de cela

    """
    rFTilde=g.N*c2r(np.fft.fft(fCorr))

    rCTilde=np.zeros(g.N)
    rCTilde[0]=np.abs(rFTilde[0])
    for i in xrange(1, (g.N-1)/2+1):
        # rFTilde[idx pairs] contiennent les coefs reels
        # resultant de c2r.C.(c2r)*
        # (meme si on l'ecrit comme un vecteur, il s'agit de la diagonale
        #   d'une matrice - un tenseur d'ordre 2, donc il faut appliquer
        #   les operateurs de chaque cote)
        rCTilde[2*i-1]=np.abs(rFTilde[2*i-1])
        rCTilde[2*i]=np.abs(rFTilde[2*i-1])
    
    rCTilde_sqrt=np.sqrt(rCTilde)
    return rCTilde_sqrt



def B_sqrt_op(xi, N, var, rCTilde_sqrt):
    """
        B_{1/2} operator

        var             :   1D array of variances
                            (diagonal os Sigma matrix)
        rCTilde_sqrt    :   1D array of the diagonal
                            of CTilde_sqrt (in 'r' basis)
    """
    xiR=rCTilde_sqrt*xi              #   1
    xiC=r2c(xiR)                #   2
    x1=np.fft.ifft(xiC).real    #   3
    return x1*var               #   4

def B_sqrt_op_T(x, N,  var, rCTilde_sqrt):
    x1=x*var                    #   4.T
    xiC=ifft_Adj(x1)        #   3.T
    xiR=r2c_Adj(xiC)            #   2.T
    return rCTilde_sqrt*xiR      #   1.T

#----| Observations |---------------------------------------

def departure(xi, N, var, B_sqrt_op, H, obs, rCTilde_sqrt):
    x=B_sqrt_op(xi, N, var, rCTilde_sqrt)
    return np.dot(H, x)-obs

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

def costFunc(xi, N, var, B_sqrt_op, B_sqrt_op_T,
                H, obs, R_inv, rCTilde_sqrt):

    J_xi=0.5*np.dot(xi, xi).real
    d=departure(xi, N, var, B_sqrt_op, H, obs, rCTilde_sqrt)
    J_o=0.5*np.dot(d,np.dot(R_inv,d))
    return J_xi+J_o

def gradCostFunc(xi, N, var, B_sqrt_op, B_sqrt_op_T,
                    H, obs, R_inv, rCTilde_sqrt):

    d=departure(xi, N, var, B_sqrt_op, H, obs, rCTilde_sqrt)
    return xi+B_sqrt_op_T(
                            np.dot(H.T,np.dot(R_inv, d)), 
                            N, var, rCTilde_sqrt )


def gradTest(costFunc, gradCostFunc, xi, N, var, B_sqrt_op, B_sqrt_op_T,
                H, obs, R_inv, rCTilde_sqrt, verbose=False):
    
    maxPow=-10
    J0=costFunc(xi, N, var, B_sqrt_op, B_sqrt_op_T,
                H, obs, R_inv, rCTilde_sqrt)
    gradJ0=gradCostFunc(xi, N, var, B_sqrt_op, B_sqrt_op_T,
                        H, obs, R_inv, rCTilde_sqrt)
    result={}
    for power in xrange(-1, maxPow, -1):
        eps=10.**(power)
        Jeps=costFunc(xi-eps*gradJ0, N, var, B_sqrt_op, B_sqrt_op_T,
                        H, obs, R_inv, rCTilde_sqrt)

        res=((J0-Jeps)/(eps*np.dot(gradJ0, gradJ0)))
        result[power]=res

        if verbose : print(power, result[power])

    return result

    
    
