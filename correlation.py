
import numpy as np
import matplotlib.pyplot as plt

import pyKdV as kdv

Ntrc=10
L=10.
g=kdv.SpectralGrid(Ntrc, L)

def fCorr(x):
    sig=0.1
    return kdv.gauss(x, 0., sig)


tfC=np.diag(fCorr(g.x))
C=np.fft.ifft2(tfC)

modC=np.zeros(shape=(g.N, g.N))
for i in xrange(g.N):
    for j in xrange(g.N):
        modC[i,j]=np.sqrt(C[i,j].real**2+C[i,j].imag**2)

