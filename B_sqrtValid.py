import numpy as np
from dataAssLib import *
import matplotlib.pyplot as plt
import pyKdV as kdv

Ng=100
sig=0.2
# pour sig=0.1 ca colle presque...
g=kdv.SpectralGrid(Ng, 100., aliasing=1)

xDirac=np.zeros(g.N)
xDirac[Ng/2+1]=1.

variances=sig*np.ones(g.N)
fCorr=fCorr_isoHomo(g.x, 5.)
rFTilde=c2r(np.fft.fft(fCorr))

# Attention, C est une matrice, un tenseur d'ordre 2,
# mais on l'ecrit comme un vecteur
# il faudra cependant etre coherent dans l'application
# des operateurs LCL* et non LC...
# les manipulations qui suivent resultent de cela
rCTilde=np.zeros(g.N)
rCTilde[0]=np.abs(rFTilde[0])
for i in xrange(1, (g.N-1)/2+1):
    # rFTilde[idx pairs] contiennent les coefs reels
    rCTilde[2*i-1]=np.abs(rFTilde[2*i-1])
    rCTilde[2*i]=np.abs(rFTilde[2*i-1])


CTilde_sqrt=np.sqrt(rCTilde)


xiTest=B_sqrt_op_T(xDirac, g.N,  variances, CTilde_sqrt)
xTest=B_sqrt_op(xiTest, g.N, variances, CTilde_sqrt)
plt.plot(g.x, xTest)
plt.plot(g.x, fCorr)
plt.show()
