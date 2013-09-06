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

def fCorr_isoHomo(g, sig):
    return kdv.gauss(g.x, 0., sig)

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



def B_sqrt_op(xi, var, rCTilde_sqrt):
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

def B_sqrt_op_T(x, var, rCTilde_sqrt):
    x1=x*var                    #   4.T
    xiC=ifft_Adj(x1)        #   3.T
    xiR=r2c_Adj(xiC)            #   2.T
    return rCTilde_sqrt*xiR      #   1.T

#----| Observations |---------------------------------------

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


#----| Cost function |--------------------------------------

def costFunc(xi, x_b, var, B_sqrt_op, B_sqrt_op_T,
                H, H_T, argsH, obs, R_inv, rCTilde_sqrt):

    J_xi=0.5*np.dot(xi.T, xi)
    d=departure(xi, x_b, var, B_sqrt_op, H, argsH, obs, rCTilde_sqrt)
    J_o=0.5*np.dot(d.T,np.dot(R_inv,d))
    return J_xi+J_o

def gradCostFunc(xi, x_b, var, B_sqrt_op, B_sqrt_op_T,
                    H, H_T, argsH, obs, R_inv, rCTilde_sqrt):

    d=departure(xi, x_b, var, B_sqrt_op, H,  argsH, obs, rCTilde_sqrt)
    gradJ_o=B_sqrt_op_T(H_T(np.dot(R_inv.T, d), *argsH), 
                            var, rCTilde_sqrt)
    return xi+gradJ_o

def gradTest(costFunc, gradCostFunc, xi, *args):
    
    maxPow=-14
    J0=costFunc(xi, *args)
    gradJ0=gradCostFunc(xi, *args)
    result={}
    for power in xrange(-1, maxPow, -1):
        eps=10.**(power)
        Jeps=costFunc(xi-eps*gradJ0, *args)
        
        n2GradJ0=np.dot(gradJ0, gradJ0)
        res=((J0-Jeps)/(eps*n2GradJ0))
        result[power]=[Jeps,n2GradJ0, res]

        verbose=True
        if verbose : print(power, result[power])

    return result
