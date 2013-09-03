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
CTilde_sqrt=rCTilde_sqrt_isoHomo(g, fCorr)

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
