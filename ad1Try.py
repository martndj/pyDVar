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
idxObs=[30,35,40,41,49,50,55,57,60,78,89,90]
H=opObs_exactIdx(g, idxObs)
obs=np.dot(H,x_truth)


#----| Preconditionning |-----
xi=np.zeros(g.N)
corr=fCorr_isoHomo(g, 5.)
rCTilde=spCorr_isoHomo(corr)
var=0.2*np.ones(g.N)

#----| Departure |------------
Hx=np.dot(H, x_bkg)
d=Hx-obs
R_inv=np.eye(len(idxObs))

#----| Minimizing |-----------
args=(d, R_inv, B_sqrt_op, H, g.N, var, rCTilde)
xi_a, out=sciOpt.fmin_bfgs(costFunc, xi, fprime=gradCostFunc,  
                args=args, retall=True, maxiter=1000)

x_a=x_bkg+B_sqrt_op(xi_a, N, var, rCTilde)


plt.plot(g.x, x_truth)
plt.plot(g.x, x_bkg)
#plt.plot(g.x, signal)
plt.plot(g.x[idxObs], obs, 'o')
plt.plot(g.x[idxObs], Hx, 'o')
plt.plot(g.x, x_a)
plt.show()    

    
    
    
