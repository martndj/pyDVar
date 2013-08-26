
import numpy as np
import matplotlib.pyplot as plt

import pyKdV as kdv
from pyKdV.pseudoSpec1D import conjT, fft2d, ifft2d

Ntrc=100
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


tfCDemi_inv=np.diag(np.sqrt(corr)**(-1))
CDemi_inv=g.N*conjT(np.fft.ifft(conjT(tfCDemi_inv)))

tfCDemi=np.diag(np.sqrt(corr))
CDemi=np.fft.ifft(tfCDemi)

#---| Validation
I=np.dot(C_direct, C_inv_direct)
IDemi=np.dot(CDemi, CDemi_inv)
# some residus, but minor: good!


#---| B matrix
var=0.2
Sigma=fSigma(g, var)
Sigma_inv=fSigma(g, var**-1)

BDemi=np.dot(Sigma, CDemi)
BDemi_inv=g.N*np.dot(conjT(np.fft.ifft(conjT(tfCDemi_inv))), Sigma_inv)

