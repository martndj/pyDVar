
import numpy as np
import matplotlib.pyplot as plt

import pyKdV as kdv
from pyKdV.pseudoSpec1D import conjT, fft2d, ifft2d

Ntrc=2
L=10.
g=kdv.SpectralGrid(Ntrc, L, aliasing=1)


def modMat(M):
    shape=M.shape
    mod=np.zeros(shape=shape)
    for i in xrange(shape[0]):
        for j in xrange(shape[1]):
            mod[i,j]=np.sqrt(M[i,j].real**2+M[i,j].imag**2)

def fCorr(x):
    sig=1.
    return kdv.gauss(x, 0., sig)

def Sigma(x):
    var=0.2
    return np.diag(var*np.ones(len(x)))



corr=fCorr(g.x)

tfC=np.diag(corr)
C_direct=ifft2d(tfC)

tfC_inv=np.diag(corr**(-1))
C_inv_direct=g.N**2*conjT(ifft2d(conjT(tfC_inv)))


tfCDemi_inv=np.diag(np.sqrt(corr)**(-1))
CDemi_inv=g.N*conjT(np.fft.ifft(conjT(tfCDemi_inv)))

tfCDemi=np.diag(np.sqrt(corr))
CDemi=np.fft.ifft(tfCDemi)

#---| Verification
I=np.dot(C_direct, C_inv_direct)
IDemi=np.dot(CDemi, CDemi_inv)
# some residus, but minor: good!

