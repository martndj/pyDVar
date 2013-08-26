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
var=0.2*np.ones(g.N)
B_sqrt=B_sqrt_isoHomo(g, var, corr)

#----| Departure |------------
Hx=np.dot(H, x_bkg)
d=Hx-obs
R_inv=np.eye(len(idxObs))

#----| Minimizing |-----------
args=(d, R_inv, B_sqrt, H)
xi_a=sciOpt.fmin_bfgs(costFunc, xi, fprime=gradCostFunc,  
                args=args, maxiter=100)

x_a=x_bkg+np.dot(B_sqrt, xi_a)


plt.plot(g.x, x_truth)
plt.plot(g.x, x_bkg)
#plt.plot(g.x, signal)
plt.plot(g.x[idxObs], obs, 'o')
plt.plot(g.x[idxObs], Hx, 'o')
plt.plot(g.x, x_a)
plt.show()    

    
    
    
