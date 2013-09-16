import numpy as np
from pyKdV import SpectralGrid, Trajectory
import scipy.optimize as sciOpt

from dVar import gradTest,  printGradTest
from kd_costFunction import *


class KdVDataAss(object):   
    """
    Variational Data Assimilation
    for Augmented Korteweg-de Vries System

    KdVDataAss(self, s_grid, traj_bkg, s_var, B_sqrt, B_sqrt_Adj, 
                    H, H_TL, H_TL_Adj, argsH, dObs, dR_inv, rCTilde_sqrt,
                    maxiter=100)

    s_grid          :   grid <pyKdV.SpectralGrid>
    traj_bkg        :   background trajectory <pyKdV.Trajectory>
    s_var           :   model variances <numpy.ndarray>
    B_sqrt, 
        B_sqrt_Adj  :   B^{1/2} operator and its adjoint
    H,H_TL,
        H_TL_Adj    :   observation operator, its tangent linear and
                            adjoint
    argsH           :   observation operators common arguments <list>
    dObs            :   observations <dict>
                            {time <float>   :   values <np.array>, ...}
    dR_inv          :   observation correlation matrices <dict>
                            {time <float>   :   diagonal <np.array>, ...}
    rCTilde_sqrt    :   CTilde^{1/2} diagonal <numpy.ndarray>
    """
    class KdVDataAssError(Exception):
        pass


    #------------------------------------------------------
    #----| Init |------------------------------------------
    #------------------------------------------------------

    def __init__(self, s_grid, traj_bkg, s_var, B_sqrt, B_sqrt_Adj, 
                    H, H_TL, H_TL_Adj, argsH, dObs, dR_inv, rCTilde_sqrt,
                    maxiter=100, retall=False):

        if not (isinstance(s_grid, SpectralGrid)):
            raise self.KdVDataAssError("s_grid <SpectralGrid>")
        self.grid=s_grid

        if not (isinstance(traj_bkg,Trajectory)):
            raise self.KdVDataAssError("traj_bkg <pyKdV.Trajectory>")
        if not (traj_bkg.shape[1]==self.grid.N):
            raise self.KdVDataAssError("traj_bkg.shape[1]<>self.grid.N")
        self.traj_bkg=traj_bkg

        if not (isinstance(s_var,np.ndarray)):
            raise self.KdVDataAssError("s_var <numpy.ndarray>")
        if not (s_var.shape==(self.grid.N,)):
            raise self.KdVDataAssError("s_var.shape<>self.grid.N")
        self.s_var=s_var

        if (not callable(B_sqrt)) or (not callable(B_sqrt_Adj)):
            raise self.KdVDataAssError("B_sqrt[_Adj] <functions>")
        self.B_sqrt=B_sqrt
        self.B_sqrt_Adj=B_sqrt_Adj

        if ((not callable(H)) or (not callable(H_TL))
            or (not callable(H_TL_Adj))):
            raise self.KdVDataAssError("H[_TL[_Adj]] <functions>")
        self.H=H
        self.H_TL=H_TL
        self.H_TL_Adj=H_TL_Adj
        self.argsH=argsH

        if not (isinstance(dObs, dict)): 
            raise self.KdVDataAssError("dObs <dict>")
        for t in dObs.iterkeys():
            if not isinstance(dObs[t], np.ndarray):
                raise self.KdVDataAssError("dObs[t] <numpy.ndarray>")
        self.dObs=dObs

        if not (isinstance(dR_inv, dict)): 
            raise self.KdVDataAssError("dR_inv <dict>")
        if np.all(np.sort(dR_inv.keys())<>np.sort(dObs.keys())):
            raise self.KdVDataAssError("dR_inv.keys()==dObs.keys()")
        for t in dR_inv.iterkeys():
            if not (isinstance(dR_inv[t],np.ndarray)):
                raise self.KdVDataAssError("dR_inv[t] <numpy.ndarray>")
            if not (dR_inv[t].shape==self.dObs[t].shape,):
                raise self.KdVDataAssError(
                "dR_inv[t].shape==self.dObs[t].shape; dR_inv[t] is the diagonal of the full R inverse matrix.")
        self.dR_inv=dR_inv

        if (not isinstance(rCTilde_sqrt, np.ndarray)):
            raise self.KdVDataAssError("rCTilde_sqrt <numpy.ndarray>")
        self.rCTilde_sqrt=rCTilde_sqrt

        self.maxiter=maxiter
        self.retall=retall


    #------------------------------------------------------
    #----| Public methods |--------------------------------
    #------------------------------------------------------


    def minimize(self, gradientTest=True):
        self.costFuncArgs=(self.traj_bkg, self.s_var, 
                        self.B_sqrt, self.B_sqrt_Adj,
                        self.H, self.H_TL, self.H_TL_Adj, self.argsH, 
                        self.dObs, self.dR_inv,
                        self.rCTilde_sqrt)
       
        xi=np.zeros(self.grid.N)

        #----| Gradient test |--------------------
        if gradientTest:
            printGradTest(gradTest(costFunc, gradCostFunc, xi, 
                                    *self.costFuncArgs))

        #----| Minimizing |-----------------------
        self.minimize=sciOpt.fmin_bfgs
        minimizeReturn=self.minimize(costFunc, xi, fprime=gradCostFunc,  
                                        args=self.costFuncArgs, 
                                        maxiter=self.maxiter,
                                        retall=self.retall,
                                        full_output=True)
        self.xi_a=minimizeReturn[0]
        self.fOpt=minimizeReturn[1]
        self.gOpt=minimizeReturn[2]
        self.hInvOpt=minimizeReturn[3]
        self.fCalls=minimizeReturn[4]
        self.gCalls=minimizeReturn[5]
        self.warnFlag=minimizeReturn[6]
        if self.retall:
            self.allvecs=minimizeReturn[7]

        #----| Final Gradient test |--------------
        if gradientTest:
            printGradTest(gradTest(costFunc, gradCostFunc, self.xi_a, 
                                    *self.costFuncArgs))

        #----| Analysis |-------------------------
        self.increment=self.B_sqrt(self.xi_a, self.s_var,
                                    self.rCTilde_sqrt)
        self.analysis=self.increment+self.traj_bkg[0]


#===========================================================
if __name__=="__main__":
    import matplotlib.pyplot as plt
    from dVar import pos2Idx, fCorr_isoHomo, B_sqrt_op, B_sqrt_op_Adj, \
                        rCTilde_sqrt_isoHomo, \
                        degrad, opObs_Idx, opObs_Idx_Adj
    from kd_observationOp import whereTrajTime
    import pyKdV as kdv
    
    
    Ntrc=100
    L=300.
    g=kdv.SpectralGrid(Ntrc, L)
        
    kdvParam=kdv.Param(g, beta=1., gamma=-1.)
    tInt=1.
    maxA=4.
    
    model=kdv.Launcher(kdvParam,tInt, maxA)

    x0_truth_base=kdv.rndFiltVec(g, Ntrc=g.Ntrc/5,  amp=1.)
    soliton=kdv.soliton(g.x, 0., amp=5., beta=1., gamma=-1)
    longWave=1.5*kdv.gauss(g.x, 40., 20. )-1.*kdv.gauss(g.x, -20., 14. )
    x0_truth=x0_truth_base+longWave
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
                    rCTilde_sqrt, retall=True)

    da.minimize()
    x0_a=da.analysis
    sub=plt.subplot(nSubLine, 1,1)
    sub.plot(g.x, x0_truth, 'k--')
    sub.plot(g.x, x0_bkg, 'b')
    sub.plot(g.x, x0_a, 'r')
    sub.legend(["$x_t$","$x_b$","$x_a$"], loc='best')
    plt.show()


