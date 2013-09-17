import numpy as np
import pyKdV as kdv
import scipy.optimize as sciOpt

from dVar import gradTest,  printGradTest

class simpleDataAss(object):   
    """
        no background KdV assimilation
    """
    class DataAssError(Exception):
        pass


    #------------------------------------------------------
    #----| Init |------------------------------------------
    #------------------------------------------------------

    def __init__(self, grid, kdvParam, kdvPropagator,
                    H, HTLM_Adj, argsH, d_Obs, d_RInv, 
                    maxiter=100, retall=False ):

        if not (isinstance(grid, kdv.SpectralGrid)):
            raise self.simpleDataAssError("grid <pyKdV.SpectralGrid>")
        self.grid=grid

        if not isinstance(kdvParam, kdv.Param):
            raise self.simpleDataAssError("kdvParam <pyKdV.Param>")
        if not isinstance(kdvPropagator, kdv.Launcher):
            raise self.simpleDataAssError("kdvPropagator <pyKdV.Launcher>")
        self.kdvParam=kdvParam
        self.propagator=kdvPropagator

        if ((not callable(H)) or (not callable(HTLM_Adj))):
            raise self.simpleDataAssError("H[TLM_Adj] <functions>")
        self.H=H
        self.HTLM_Adj=HTLM_Adj
        self.argsH=(self.propagator,)+argsH
        self.argsHTLM_Adj=argsH


        if not (isinstance(d_Obs, dict)): 
            raise self.simpleDataAssError("d_Obs <dict>")
        for t in d_Obs.iterkeys():
            if not isinstance(d_Obs[t], np.ndarray):
                raise self.simpleDataAssError("d_Obs[t] <numpy.ndarray>")
        self.d_Obs=d_Obs

        if not (isinstance(d_RInv, dict)): 
            raise self.simpleDataAssError("d_RInv <dict>")
        if np.all(np.sort(d_RInv.keys())<>np.sort(d_Obs.keys())):
            raise self.simpleDataAssError("d_RInv.keys()==d_Obs.keys()")
        for t in d_RInv.iterkeys():
            if not (isinstance(d_RInv[t],np.ndarray)):
                raise self.simpleDataAssError("d_RInv[t] <numpy.ndarray>")
            if not (d_RInv[t].shape==self.d_Obs[t].shape,):
                raise self.simpleDataAssError(
                "d_RInv[t].shape==self.d_Obs[t].shape; d_RInv[t] is the diagonal of the full R inverse matrix.")
        self.d_RInv=d_RInv

        self.maxiter=maxiter
        self.retall=retall


    #------------------------------------------------------
    #----| Public methods |--------------------------------
    #------------------------------------------------------


    def minimize(self, gradientTest=True, firstGuess=None):

        if firstGuess==None:
            x=np.zeros(self.grid.N)
        elif isinstance(firstGuess, np.ndarray):
            if len(firstGuess)==self.grid.N:
                x=firstGuess
            else:
                raise self.simpleDataAssError(
                            "len(firstGuess)==self.grid.N")
        else:
            raise self.simpleDataAssError("firstGuess <numpy.ndarray>")
        #----| Gradient test |--------------------
        if gradientTest:
            printGradTest(gradTest(self.__costFunc, self.__gradCostFunc,
                                    x,()))

        #----| Minimizing |-----------------------
        self.minimize=sciOpt.fmin_bfgs
        minimizeReturn=self.minimize(self.__costFunc, x, 
                                        fprime=self.__gradCostFunc,  
                                        maxiter=self.maxiter,
                                        retall=self.retall,
                                        full_output=True)
        self.x_a=minimizeReturn[0]
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
            printGradTest(gradTest(self.__costFunc, self.__gradCostFunc,
                                    self.x_a, ()))

        #----| Analysis |-------------------------
        self.analysis=self.x_a


    #------------------------------------------------------
    #----| Private methods |-------------------------------
    #------------------------------------------------------

    def __costFunc(self, x):
        d_D=self.__departure(x)
        Jo=0.
        for t in d_D.keys():
            Jo+=0.5*np.dot(d_D[t],np.dot(self.d_RInv[t],d_D[t]))
        return Jo
    
    #------------------------------------------------------
        
    def __gradCostFunc(self, x):
        d_D=self.__departure(x)
        d_NormD={}
        for t in d_D.keys():
            d_NormD[t]=np.dot(self.d_RInv[t],d_D[t])
        #----| building reference trajectory |--------
        tInt=np.max(self.d_Obs.keys())
        traj_x=self.propagator.integrate(x, tInt)
        tlm=kdv.TLMLauncher(traj_x, self.kdvParam)

        dx0=self.HTLM_Adj(d_NormD, tlm, *self.argsHTLM_Adj)
        grad=-dx0
        return grad

    #------------------------------------------------------
        
    def __departure(self, x):
    
        d_D={}
        d_Hx=self.H(x,*self.argsH)
        for t in d_Hx.keys():
            d_D[t]=d_Obs[t]-d_Hx[t]
    
        return d_D
    
#===========================================================
if __name__=="__main__":
    import matplotlib.pyplot as plt
    from dVar import pos2Idx, degrad, opObs_Idx, opObs_Idx_Adj
    from kd_observationOp import whereTrajTime, kd_opObs, kd_opObs_TL_Adj
    import pyKdV as kdv
    
    
    Ntrc=100
    L=300.
    g=kdv.SpectralGrid(Ntrc, L)
        
    kdvParam=kdv.Param(g, beta=1., gamma=-1.)
    tInt=30.
    maxA=2.
    maxiter=50
    
    model=kdv.Launcher(kdvParam, maxA)

    rndLFBase=kdv.rndFiltVec(g, Ntrc=g.Ntrc/5,  amp=0.4)
    soliton=kdv.soliton(g.x, 0., amp=1.9, beta=1., gamma=-1)
    longWave=0.8*kdv.gauss(g.x, 40., 20. )-0.5*kdv.gauss(g.x, -20., 14. )

    x0_truth=rndLFBase+longWave+soliton
    x_truth=model.integrate(x0_truth, tInt)

    x0_bkg=rndLFBase
    x_bkg=model.integrate(x0_bkg, tInt)
    
    #----| Observations |---------
    d_ObsPos={}
    nObsTime=3
    for i in xrange(nObsTime):
        d_ObsPos[tInt/(i+1)]=x_truth[x_truth.whereTime(tInt/(i+1))]
        
    H=kd_opObs
    H_TL_Adj=kd_opObs_TL_Adj
    staticObsOp=None
    sObsOpArgs=()
    argsHcom=(g, d_ObsPos, staticObsOp, sObsOpArgs)
    
    sigR=.5
    d_Obs=H(x0_truth, model,  *argsHcom) 
    
    d_RInv={}
    for t in d_ObsPos.keys():
        d_RInv[t]=sigR**(-1)*np.eye(len(d_ObsPos[t]))

    #----| Observations to be assimilated
    nTime=len(d_Obs.keys())
    nSubRow=3
    nSubLine=nTime/nSubRow+1
    if nTime%nSubRow: nSubLine+=1
    plt.figure(figsize=(12.,12.))
    i=0
    for t in np.sort(d_Obs.keys()):
        i+=1
        sub=plt.subplot(nSubLine, nSubRow, nSubRow+i)
        ti=whereTrajTime(x_truth, t)
        sub.plot(g.x, d_Obs[t], 'g')
        sub.plot(g.x, x_bkg[ti], 'b')
        sub.set_title("$t=%f$"%t)
        if i==nTime:
            sub.legend(["$y$", "$x_b$"], loc="lower left")

    da=simpleDataAss(g, kdvParam, model, H, H_TL_Adj, argsHcom, d_Obs,
                    d_RInv, retall=True, maxiter=maxiter)

    da.minimize(firstGuess=x0_bkg)
    x0_a=da.analysis
    x_a=model.integrate(x0_a, tInt)


    sub=plt.subplot(nSubLine, 1,1)
    sub.plot(g.x, x0_truth, 'k--', linewidth=2)
    sub.plot(g.x, x0_bkg, 'b--')
    sub.plot(g.x, x0_a, 'r--')
    sub.plot(g.x, x_truth.final(), 'k', linewidth=2)
    sub.plot(g.x, x_bkg.final(), 'b')
    sub.plot(g.x, x_a.final(), 'r')
    sub.legend(["${x_t}_0$","${x_b}_0$","${x_a}_0$",
                "${x_t}_f$","${x_b}_f$","${x_a}_f$"], loc='best')
    plt.show()


