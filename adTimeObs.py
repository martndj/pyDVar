import numpy as np
import matplotlib.pyplot as plt
from dVar import pos2Idx, fCorr_isoHomo, B_sqrt_op, B_sqrt_op_Adj, \
                    rCTilde_sqrt_isoHomo, \
                    degrad, opObs_Idx, opObs_Idx_Adj
from kdVar import *
import pyKdV as kdv


Ntrc=100
L=300.
g=kdv.SpectralGrid(Ntrc, L)
    
kdvParam=kdv.Param(g, beta=1., gamma=-1.)
tInt=10.
maxA=4.

model=kdv.Launcher(kdvParam,tInt, maxA)

x0_truth_base=kdv.rndFiltVec(g, Ntrc=g.Ntrc/5,  amp=1.)
wave=kdv.soliton(g.x, 0., amp=5., beta=1., gamma=-1)\
            +1.5*kdv.gauss(g.x, 40., 20. )-1.*kdv.gauss(g.x, -20., 14. )
x0_truth=x0_truth_base+wave
x_truth=model.integrate(x0_truth)

x0_bkg=x0_truth_base
x_bkg=model.integrate(x0_bkg)

#----| Observations |---------
dObsPos={}
nTime=10
nPosObs=50
for i in xrange(nTime):
    dObsPos[tInt/(i+1.)]=np.linspace(-g.L/2., g.L/2., nPosObs)

H=kd_opObs
H_TL=kd_opObs_TL
H_TL_Adj=kd_opObs_TL_Adj
argsHcom=(g, dObsPos, kdvParam, maxA)

sigR=.5
x0_degrad=degrad(x0_truth, 0., sigR)                   
dObs_degrad=H(x0_degrad, *argsHcom) 
dObs_truth=H(x0_truth,  *argsHcom) 

dR_inv={}
for t in dObsPos.keys():
    dR_inv[t]=sigR**(-1)*np.eye(len(dObsPos[t]))

#----| Observations to be assimilated
dObs=dObs_truth
nTime=len(dObs_degrad.keys())
nSubRow=3
nSubLine=nTime/nSubRow+1
if nTime%nSubRow: nSubLine+=1
plt.figure(figsize=(12.,12.))
i=0
for t in np.sort(dObs_degrad.keys()):
    i+=1
    sub=plt.subplot(nSubLine, nSubRow, nSubRow+i)
    ti=whereTrajTime(x_truth, t)
    sub.plot(g.x, x_truth[ti], 'g')
    sub.plot(g.x[pos2Idx(g, dObsPos[t])], dObs[t], 'go')
    sub.plot(g.x, x_bkg[ti], 'b')
    sub.plot(g.x[pos2Idx(g, dObsPos[t])], 
                x_bkg[ti][pos2Idx(g, dObsPos[t])], 'bo')
    sub.set_title("$t=%f$"%t)
    if i==nTime:
        sub.legend(["$x_{t}$",  "$y$", "$x_b$", 
                    "$H(x_b)$"], loc="lower left")

#----| Validating H_TL_Adj |----
x_rnd=kdv.rndFiltVec(g, amp=0.5)
dY=dObs
Hx=H_TL(x_rnd, x_bkg, *argsHcom)
H_Adjy=H_TL_Adj(dY, x_bkg, *argsHcom)
prod1=0.
for t in Hx.keys():
    prod1+=np.dot(dY[t], Hx[t])
prod2=np.dot(H_Adjy, x_rnd)
print("Validating adjoint of observation TLM")
print(np.abs(prod1-prod2))
    

#----| Preconditionning |-----------
Lc=10.
sig=2.
corr=fCorr_isoHomo(g, Lc)
rCTilde_sqrt=rCTilde_sqrt_isoHomo(g, corr)
var=sig*np.ones(g.N)
xi=np.zeros(g.N)





#----| Assimilation |---------------
da=KdVDataAss(g, x_bkg, var, B_sqrt_op, B_sqrt_op_Adj,
                H, H_TL, H_TL_Adj, argsHcom, dObs, dR_inv, 
                rCTilde_sqrt)

da.minimize()
x0_a=da.analysis
sub=plt.subplot(nSubLine, 1,1)
sub.plot(g.x, x0_truth, 'k--')
sub.plot(g.x, x0_bkg, 'b')
sub.plot(g.x, x0_a, 'r')
sub.legend(["$x_t$","$x_b$","$x_a$"], loc='best')
plt.show()


