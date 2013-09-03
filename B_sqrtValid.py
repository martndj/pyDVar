import numpy as np
import random as rnd
from dataAssLib import *
import matplotlib.pyplot as plt
import pyKdV as kdv

Ng=100
# pour sig=0.1 ca colle presque...
g=kdv.SpectralGrid(Ng, 100., aliasing=1)


sig=1.
lCorr=5.
variances=sig*np.ones(g.N)
fCorr=fCorr_isoHomo(g.x, lCorr)
rFTilde=g.N*c2r(np.fft.fft(fCorr))

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

# correlation test
xDirac=np.zeros(g.N)
xDirac[Ng/4]=1.
xiTest=B_sqrt_op_T(xDirac, g.N,  variances, CTilde_sqrt)
xTest=B_sqrt_op(xiTest, g.N, variances, CTilde_sqrt)

# adjoint test
rnd.seed(0.4573216806)
mu=0.; sigNoise=2.
xNoise=np.zeros(g.N)
yNoise=np.zeros(g.N)
for i in xrange(g.N):
    yNoise[i]=rnd.gauss(mu, sigNoise)
    xNoise[i]=rnd.gauss(mu, sigNoise)
testDirect=np.dot(xNoise,
                    B_sqrt_op(yNoise, g.N, variances, CTilde_sqrt).conj())
testAdjoint=np.dot(B_sqrt_op_T(xNoise, g.N, variances, CTilde_sqrt),
                    yNoise.conj())

#print(testDirect)
#print(testAdjoint)
print("Adjoint test with noise: <x,Gy>-<G*x,y>")
print(testDirect-testAdjoint)


#plt.plot(g.x, xTest/sig**2)
plt.plot(g.x, xTest)
plt.plot(g.x, fCorr)
plt.show()
