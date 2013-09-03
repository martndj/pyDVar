import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sciOpt

from dataAssLib import *
import pyKdV as kdv

Ntrc=100
L=100.
g=kdv.SpectralGrid(Ntrc, L, aliasing=1)

    

x_truth=kdv.soliton(g.x, 0., 1., 0., 1., -1)
x_bkg=kdv.soliton(g.x, -4.2, 1.1, 0.3, 1.2, -1.76)


#----| Observations |---------
idxObs=[33, 47, 50, 55, 67, 80]
H=opObs_exactIdx(g.N, idxObs)
obs=np.dot(H,x_truth)


#----| Preconditionning |-----
Lc=10.
sig=2.
corr=fCorr_isoHomo(g.x, Lc)
rCTilde=rCTilde_sqrt_isoHomo(g, corr)
var=sig*np.ones(g.N)

xi=np.zeros(g.N)

#----| testing B_sqrt |-------
#xDirac=np.zeros(g.N)
#xDirac[g.N/2]=1.
#xiTest=B_sqrt_op_T(xDirac, g.N,  var, rCTilde)
#xTest=B_sqrt_op(xiTest, g.N, var, rCTilde)
#plt.figure()
#plt.plot(g.x, xTest/sig**2)
#plt.plot(g.x, corr)
#plt.show()


#----| Departure |------------
Hx=np.dot(H, x_bkg)
d=Hx-obs
sigR=0.1
R_inv=sigR**(-1)*np.eye(len(idxObs))

#----| Minimizing |-----------
args=(d, R_inv, B_sqrt_op_T, H, g.N, var, rCTilde)
xi_a, out=sciOpt.fmin_bfgs(costFunc, xi, fprime=gradCostFunc,  
                args=args, retall=True, maxiter=50000)

x_a=x_bkg+B_sqrt_op(xi_a, g.N, var, rCTilde)


plt.figure()
plt.plot(g.x, x_truth,'k--')
plt.plot(g.x, x_bkg,'m')
#plt.plot(g.x, signal)
plt.plot(g.x[idxObs], obs, 'go')
plt.plot(g.x[idxObs], Hx, 'mo')
plt.plot(g.x, x_a, 'r')
plt.plot(g.x[idxObs], x_a[idxObs],'ro')
plt.legend(['$x_t$', '$x_b$', '$y$', r'$H(x)$', r'$x_a$'])
plt.show()    

    
    
    
