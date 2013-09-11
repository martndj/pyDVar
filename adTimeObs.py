import numpy as np
import matplotlib.pyplot as plt

from dVar import pos2Idx, fCorr_isoHomo, degrad, B_sqrt_op, \
                    rCTilde_sqrt_isoHomo, opObs_Idx
from kdVar import kd_opObs, kd_opObs_TL, whereTrajTime, kd_departure 
import pyKdV as kdv


Ntrc=100
L=300.
g=kdv.SpectralGrid(Ntrc, L)
    
kdvParam=kdv.Param(g, beta=1., gamma=-1.)
tInt=10.
maxA=5.

x0_truth_base=kdv.rndFiltVec(g, Ntrc=g.Ntrc/5,  amp=1.)
wave=kdv.soliton(g.x, 0., amp=5., beta=1., gamma=-1)\
            +1.5*kdv.gauss(g.x, 40., 20. )-1.*kdv.gauss(g.x, -20., 14. )
x0_truth=x0_truth_base+wave
launcher_truth=kdv.Launcher(kdvParam, x0_truth)
x_truth=launcher_truth.integrate(tInt, maxA)

x0_bkg=x0_truth_base
launcher_bkg=kdv.Launcher(kdvParam, x0_bkg)
x_bkg=launcher_bkg.integrate(tInt, maxA)

#----| Observations |---------
H=kd_opObs
H_TL=kd_opObs_TL
dObsPos={}
dObsPos[1.]=np.array([-30.,  70.])
dObsPos[3.]=np.array([-120., -34., -20., 2.,  80., 144.])
dObsPos[6.]=np.array([-90., -85, 4., 10.])
dObsPos[9.]=np.array([-50., 0., 50.])
argsH=(g, dObsPos, opObs_Idx, kdvParam, maxA)

sigR=.5
x0_degrad=degrad(x0_truth, 0., sigR)                   
dObs_degrad=H(x0_degrad, *argsH) 
dObs_truth=H(x0_truth, *argsH) 
                     

dR_inv={}
for t in dObsPos.keys():
    dR_inv[t]=sigR**(-1)*np.eye(len(dObsPos[t]))

#----| Preconditionning |-----
Lc=10.
sig=0.4
corr=fCorr_isoHomo(g, Lc)
rCTilde_sqrt=rCTilde_sqrt_isoHomo(g, corr)
var=sig*np.ones(g.N)
xi=np.zeros(g.N)

#----| Departures |-----------
dDepartures=kd_departure(xi, x0_bkg, var, B_sqrt_op, H, H_TL, argsH, 
                            dObs_degrad, rCTilde_sqrt)
for t in np.sort(dObs_degrad.keys()):
    print("t=%f"%t)
    print(dDepartures[t])

#----| Minimizing |-----------

#----| Post-processing |------
nTime=len(dObs_degrad.keys())
plt.figure(figsize=(10.,3.*nTime))
i=0
for t in np.sort(dObs_degrad.keys()):
    i+=1
    sub=plt.subplot(nTime, 1, i)
    ti=whereTrajTime(x_truth, t)
    sub.plot(g.x, x_truth[ti], 'g')
    sub.plot(g.x[pos2Idx(g, dObsPos[t])], dObs_truth[t], 'go')
    sub.plot(g.x[pos2Idx(g, dObsPos[t])], dObs_degrad[t], 'ro')
    sub.plot(g.x, x_bkg[ti], 'b')
    sub.plot(g.x[pos2Idx(g, dObsPos[t])], 
                x_bkg[ti][pos2Idx(g, dObsPos[t])], 'bo')
    sub.set_title("$t=%f$"%t)
    if i==nTime:
        sub.legend(["$x_{t}$", "$H(x_{t})$", "$y$", "$x_b$", "$H(x_b)$"],
                    loc="lower left")
plt.show()    
