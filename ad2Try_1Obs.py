import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.optimize as sciOpt

from dataAssLib import *
import pyKdV as kdv

Ntrc=100
L=100.
g=kdv.SpectralGrid(Ntrc, L, aliasing=1)
    
def reality(x):
    return 1.+0.3*kdv.gauss(x, 5., 8. )-0.4*kdv.gauss(x, -6., 14. )
x_truth=reality(g.x)
x_bkg=np.ones(g.N)

#----| Observations |---------
nObs=1
posObs=np.empty(nObs)
idxObs=np.empty(nObs, dtype=int)
posObs[0]=0.
idxObs[0]=np.min(np.where(g.x>=posObs[0]))
H=opObs_exactIdx(g.N, idxObs)
obs=np.dot(H,x_truth)
sigR=1.
R_inv=sigR**(-1)*np.eye(len(idxObs))

#----| Preconditionning |-----
Lc=10.
sig=1.
corr=fCorr_isoHomo(g.x, Lc)
rCTilde_sqrt=rCTilde_sqrt_isoHomo(g, corr)
var=sig*np.ones(g.N)
xi=np.zeros(g.N)

#----| Minimizing |-----------
args=(x_bkg, g.N, var, B_sqrt_op, B_sqrt_op_T, H, obs, R_inv, rCTilde_sqrt)
xi_a, out=sciOpt.fmin_bfgs(costFunc, xi, fprime=gradCostFunc,  
                args=args, retall=True, maxiter=100)

#----| Gradient test |--------
resultGradTest=gradTest(costFunc, gradCostFunc, xi_a, *args)


#----| Analysis |-------------
x_a=B_sqrt_op(xi_a, g.N, var, rCTilde_sqrt)+x_bkg

plt.figure()
gs = gridspec.GridSpec(2, 1,height_ratios=[4,1])
analGrph=plt.subplot(gs[0])
analGrph.plot(g.x, x_truth,'k--')
analGrph.plot(g.x, x_bkg,'m')
analGrph.plot(g.x[idxObs], obs, 'go')
analGrph.plot(g.x[idxObs], np.dot(H,x_bkg), 'mo')
analGrph.plot(g.x, x_a, 'r')
analGrph.plot(g.x[idxObs], x_a[idxObs],'ro')
analGrph.legend(['$x_t$', '$x_b$', '$y$', r'$H(x)$', r'$x_a$'])

corrGrph=plt.subplot(gs[1])
corrGrph.plot(g.x, corr)
corrGrph.legend(['$f(x)$'])

plt.show()    
