import numpy as np
import matplotlib.pyplot as plt
import random as rnd

import pyKdV as kdv
from pyKdV.pseudoSpec1D import conjT, fft2d, ifft2d

Ntrc=100
L=100.
g=kdv.SpectralGrid(Ntrc, L, aliasing=1)

def opObs(grid, s, idxObs):
    """
    s   :   model state
    idxObs :  shape=(nObs,)
    """
    nObs=len(idxObs)
    Hs=np.empty(nObs)
    for i in xrange(nObs):
        Hs[i]=s[idxObs[i]]
    return Hs


def obsCovar_independant(grid, idxObs, sig):
    nObs=len(idxObs)
    return np.eye(nObs)

    


idxObs=[30, 35,50,55, 57, 78, 89, 90]
signal=kdv.soliton(g.x, 0., 1., 0., 1., -1)
Hs=opObs(g, signal, idxObs)
plt.plot(g.x, signal)
plt.plot(g.x[idxObs], Hs, 'o')
    
    
    
    
    
