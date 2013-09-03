import numpy as np
import random as rnd
from dataAssLib import r2c, c2r, r2c_Adj, ifft_Adj
rnd.seed(0.4573216806)

N=11
mu=1.
sig=1.



x=np.empty(N, dtype='complex')
y=np.empty(N)

x[0]=rnd.gauss(mu, sig)
for i in xrange(1,(N-1)/2+1):
    x[i]=rnd.gauss(mu, sig)+1j*rnd.gauss(mu,sig)
    x[N-i]=x[i].real-1j*x[i].imag
for i in xrange(N):
    y[i]=rnd.gauss(mu, sig)

print("Testing adjoint of r2c()")
print(np.dot(x.conj(), r2c(y))-np.dot(r2c_Adj(x),y))

print("Testing adjoint of ifft()")
Lx_y=np.dot(np.fft.ifft(x), y.conj())
x_LAdjy=np.dot(x, ifft_Adj(y).conj())
print(Lx_y-x_LAdjy)
