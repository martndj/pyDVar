
import numpy as np
import matplotlib.pyplot as plt

import pyKdV as kdv
from pyKdV.pseudoSpec1D import conjT, fft2d, ifft2d
from dataAssLib import *

Ntrc=4
L=10.
g=kdv.SpectralGrid(Ntrc, L, aliasing=1)



def fCorr(g, sig):
    return kdv.gauss(g.x, 0., sig)

def fSigma(g, var):
    return np.diag(var*np.ones(g.N))



corr=fCorr(g, 1.)

tfC=np.diag(corr)
C_direct=ifft2d(tfC)

tfC_inv=np.diag(corr**(-1))
C_inv_direct=g.N**2*conjT(ifft2d(conjT(tfC_inv)))


tfC_sqrt_inv=np.diag(np.sqrt(corr)**(-1))
C_sqrt_inv=g.N*conjT(np.fft.ifft(conjT(tfC_sqrt_inv)))

tfC_sqrt=np.diag(np.sqrt(corr))
C_sqrt=np.fft.ifft(tfC_sqrt)

#---| Validation
I=np.dot(C_direct, C_inv_direct)
I_sqrt=np.dot(C_sqrt, C_sqrt_inv)
# some residus, but minor: good!


#---| B matrix
var=0.2
Sigma=fSigma(g, var)
Sigma_inv=fSigma(g, var**-1)

#B_sqrt=np.dot(Sigma, C_sqrt)
#B_sqrt_inv=g.N*np.dot(conjT(np.fft.ifft(conjT(tfC_sqrt_inv))), Sigma_inv)
B_sqrt=B_sqrt_isoHomo(g, var*np.ones(g.N), fCorr_isoHomo(g, 1.))
B_sqrt_inv=B_sqrt_inv_isoHomo(g, var*np.ones(g.N), fCorr_isoHomo(g, 1.))

B=np.dot(B_sqrt,B_sqrt.T)
B_real=np.dot(B_sqrt.real,B_sqrt.real.T)