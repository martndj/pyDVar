import numpy as np
import random as rnd
from dataAssLib import r2c, c2r, r2c_Adj
rnd.seed(0.4573216806)

N=11
mu=0.
sig=1.

def ifft_Adj(x):
    N=len(x)
    xi=np.zeros(N)
    xi=np.fft.fft(x)
    #xi=xi*N
    # --- traitement particulier pour le terme a0
    #  clarifier et voir les notes de Pierre
    # et verifier le test d'ajointitude
    #for i in xrange(1, N):
    #    xi[i]=xi[i]*0.5
    return xi


x=np.empty(N, dtype='complex')
y=np.empty(N)

x[0]=rnd.gauss(mu, sig)
for i in xrange(1,(N-1)/2+1):
    x[i]=rnd.gauss(mu, sig)+1j*rnd.gauss(mu,sig)
    x[N-i]=x[i].real-1j*x[i].imag
for i in xrange(N):
    y[i]=rnd.gauss(mu, sig)


print(np.dot(x.conj(), r2c(y)))
print(np.dot(r2c_Adj(x),y))

Lx_y=np.dot(np.fft.ifft(x), y)
x_LAdjy=np.dot(x, ifft_Adj(y))
print(Lx_y-x_LAdjy)
