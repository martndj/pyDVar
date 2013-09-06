import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sciOpt

from dVar import *
import pyKdV as kdv

Ntrc=100
L=100.
g=kdv.SpectralGrid(Ntrc, L, aliasing=1)
    
def reality(x):
    return 1.+0.3*kdv.gauss(x, 5., 8. )-0.4*kdv.gauss(x, -6., 14. )
x_truth=reality(g.x)
x_bkg=np.ones(g.N)

#----| Observations |---------
nObs=10
posObs=np.empty(nObs)
idxObs=np.empty(nObs, dtype=int)
for i in xrange(nObs):
    posObs[i]=-L/2.+i*L/nObs
    idxObs[i]=np.min(np.where(g.x>=posObs[i]))
H=opObs_Idx_op
H_T=opObs_Idx_op_T
argsH=(g, idxObs)

sigR=.2
x_noise=degrad(reality(g.x), 0., sigR)
obs=H(x_noise, *argsH)
R_inv=sigR**(-1)*np.eye(len(idxObs))

#----| Preconditionning |-----
Lc=10.
sig=0.4
corr=fCorr_isoHomo(g, Lc)
rCTilde_sqrt=rCTilde_sqrt_isoHomo(g, corr)
var=sig*np.ones(g.N)
xi=np.zeros(g.N)



args=(x_bkg, var, B_sqrt_op, B_sqrt_op_T, 
        H, H_T, argsH, obs, R_inv, rCTilde_sqrt)
#----| Gradient test |--------
resultGradTest=gradTest(costFunc, gradCostFunc, xi, *args)
#----| Minimizing |-----------
xi_a, out=sciOpt.fmin_bfgs(costFunc, xi, fprime=gradCostFunc,  
                args=args, retall=True, maxiter=100)

#----| Final Gradient test |--
resultGradTest=gradTest(costFunc, gradCostFunc, xi_a, *args)


#----| Analysis |-------------
x_a=B_sqrt_op(xi_a, var, rCTilde_sqrt)+x_bkg

plt.figure()
plt.plot(g.x, x_truth,'k--')
plt.plot(g.x, x_bkg,'m')
plt.plot(g.x[idxObs], obs, 'go')
plt.plot(g.x[idxObs], H(x_bkg, *argsH), 'mo')
plt.plot(g.x, x_a, 'r')
plt.plot(g.x[idxObs], x_a[idxObs],'ro')
plt.legend(['$x_t$', '$x_b$', '$y$', r'$H(x)$', r'$x_a$'])
plt.show()    
