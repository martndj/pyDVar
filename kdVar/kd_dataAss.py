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

    def __init__(self, s_grid, x_bkg, s_var, B_sqrt, B_sqrt_Adj, 
                    H, model, H_TL_Adj, tlmArgs, argsH, dObs, dR_inv, 
                    rCTilde_sqrt, maxiter=100, retall=False ):

        if not (isinstance(s_grid, SpectralGrid)):
            raise self.KdVDataAssError("s_grid <SpectralGrid>")
        self.grid=s_grid

        if not (isinstance(x_bkg,np.ndarray)):
            raise self.KdVDataAssError("x_bkg <numpy.ndarray>")
        if not (len(x_bkg)==self.grid.N):
            raise self.KdVDataAssError("len(x_bkg)<>self.grid.N")
        self.x_bkg=x_bkg

        if not (isinstance(s_var,np.ndarray)):
            raise self.KdVDataAssError("s_var <numpy.ndarray>")
        if not (s_var.shape==(self.grid.N,)):
            raise self.KdVDataAssError("s_var.shape<>self.grid.N")
        self.s_var=s_var

        if (not callable(B_sqrt)) or (not callable(B_sqrt_Adj)):
            raise self.KdVDataAssError("B_sqrt[_Adj] <functions>")
        self.B_sqrt=B_sqrt
        self.B_sqrt_Adj=B_sqrt_Adj

        if ((not callable(H)) or (not callable(H_TL_Adj))):
            raise self.KdVDataAssError("H[_TL_Adj] <functions>")
        if not isinstance(model, kdv.Launcher):
            raise self.KdVDataAssError("model <pyKdV.Launcher>")
        self.model=model
        self.H=H
        self.H_TL_Adj=H_TL_Adj
        self.argsH=argsH
        self.tlmArgs=tlmArgs

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
        self.costFuncArgs=(self.x_bkg, self.s_var, 
                        self.B_sqrt, self.B_sqrt_Adj,
                        self.H, self.model, self.H_TL_Adj, 
                        kdv.TLMLauncher, self.tlmArgs, self.argsH, 
                        self.dObs, self.dR_inv,
                        self.rCTilde_sqrt)
       
        xi=np.zeros(self.grid.N)

        #----| Gradient test |--------------------
        if gradientTest:
            printGradTest(gradTest(costFunc, gradCostFunc, xi, 
                                    self.costFuncArgs))

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
                                    self.costFuncArgs))

        #----| Analysis |-------------------------
        self.increment=self.B_sqrt(self.xi_a, self.s_var,
                                    self.rCTilde_sqrt)
        self.analysis=self.increment+self.x_bkg


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
    tInt=3.
    maxA=2.
    maxiter=50
    
    model=kdv.Launcher(kdvParam, maxA)

    rndLFBase=kdv.rndFiltVec(g, Ntrc=g.Ntrc/5,  amp=0.4)
    soliton=kdv.soliton(g.x, 0., amp=1.9, beta=1., gamma=-1)
    longWave=0.8*kdv.gauss(g.x, 40., 20. )-0.5*kdv.gauss(g.x, -20., 14. )

    x0_truth=longWave
    x_truth=model.integrate(x0_truth, tInt)

    x0_bkg=np.zeros(g.N)
    x_bkg=model.integrate(x0_bkg, tInt)
    
    #----| Observations |---------
    dObsPos={}
    nObsTime=3
    for i in xrange(nObsTime):
        dObsPos[tInt/(i+1)]=x_truth[x_truth.whereTime(tInt/(i+1))]
        
    H=kd_opObs
    H_TL_Adj=kd_opObs_TL_Adj
    staticObsOp=None
    sObsOpArgs=()
    argsHcom=(g, dObsPos, staticObsOp, sObsOpArgs)
    
    sigR=.5
    dObs=H(x0_truth, model,  *argsHcom) 
    
    dR_inv={}
    for t in dObsPos.keys():
        dR_inv[t]=sigR**(-1)*np.eye(len(dObsPos[t]))

    #----| Observations to be assimilated
    nTime=len(dObs.keys())
    nSubRow=3
    nSubLine=nTime/nSubRow+1
    if nTime%nSubRow: nSubLine+=1
    plt.figure(figsize=(12.,12.))
    i=0
    for t in np.sort(dObs.keys()):
        i+=1
        sub=plt.subplot(nSubLine, nSubRow, nSubRow+i)
        ti=whereTrajTime(x_truth, t)
        sub.plot(g.x, dObs[t], 'g')
        sub.plot(g.x, x_bkg[ti], 'b')
        sub.set_title("$t=%f$"%t)
        if i==nTime:
            sub.legend(["$y$", "$x_b$"], loc="lower left")

    #----| Preconditionning |-----------
    Lc=10.
    sig=2.
    corr=fCorr_isoHomo(g, Lc)
    rCTilde_sqrt=rCTilde_sqrt_isoHomo(g, corr)
    var=sig*np.ones(g.N)
    xi=np.zeros(g.N)


    #----| Assimilation |---------------
    da=KdVDataAss(g, x0_bkg, var, B_sqrt_op, B_sqrt_op_Adj,
                    H, model, H_TL_Adj, (kdvParam,), argsHcom, dObs,
                    dR_inv, rCTilde_sqrt, retall=True, maxiter=maxiter)

    da.minimize()
    x0_a=da.analysis
    x_a=model.integrate(x0_a, tInt)

    sub=plt.subplot(nSubLine, 1,1)
    sub.plot(g.x, x0_truth, 'k--')
    sub.plot(g.x, x0_bkg, 'b--')
    sub.plot(g.x, x0_a, 'r--')
    sub.plot(g.x, x_truth.final(), 'k')
    sub.plot(g.x, x_bkg.final(), 'b')
    sub.plot(g.x, x_a.final(), 'r')
    sub.legend(["${x_t}_0$","${x_b}_0$","${x_a}_0$",
                "${x_t}_f$","${x_b}_f$","${x_a}_f$"], loc='best')
    plt.show()


