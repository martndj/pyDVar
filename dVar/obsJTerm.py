from jTerm import JTerm, norm
from observations import StaticObs, TimeWindowObs
from pseudoSpec1D import PeriodicGrid, Launcher, TLMLauncher
import numpy as np

class BkgJTerm(JTerm):
    """
    Background model state JTerm subclass

        BkgJTerm(bkg, grid, metric=None)

            bkg     :   background model state <numpy.ndarray>
            grid    :   <PeriodicGrid>
            metric  :   information metric (B^{-1})
                            <float | numpy.ndarray >
    """
    class BkgJTermError(JTerm):
        pass


    #------------------------------------------------------
    #----| Init |------------------------------------------
    #------------------------------------------------------

    def __init__(self, bkg, g, metric=1., maxGradNorm=None): 

        if not isinstance(g, PeriodicGrid):
            raise self.BkgJTermError("g <pseudoSpec1D.PeriodicGrid>")
        self.grid=g

        if not isinstance(bkg, np.ndarray):
            raise self.BkgJTermError("bkg <numpy.ndarray>")
        if not (bkg.ndim==1 and len(bkg)==g.N):
            raise self.BkgJTermError("bkg.shape==(g.N,)")
        self.bkg=bkg
        self.N=self.grid.N

        if isinstance(metric, (float, int)):
            self.metric=metric*np.eye(self.N)
        elif isinstance(metric, np.ndarray):
            if metric.ndim==1:
                self.metric=np.diag(metric)
            elif metric.ndim==2:
                self.metric=metric
            else:
                raise self.BkgJTermError("metric.ndim=[1|2]")
        else:   
            raise self.BkgJTermError("metric <None | numpy.ndarray>")


        if not (isinstance(maxGradNorm, float) or maxGradNorm==None):
            raise self.BkgJTermError("maxGradNorm <None|float>")
        self.maxGradNorm=maxGradNorm 
        self.args=()

        self.isMinimized=False
        
    #------------------------------------------------------
    #----| Private methods |-------------------------------
    #------------------------------------------------------

    def __xValidate(self, x):
        if not isinstance(x, np.ndarray):
            raise self.BkgJTermError("x <numpy.array>")
        if not x.dtype=='float64':
            raise self.BkgJTermError("x.dtype=='float64'")
        if x.ndim<>1:
            raise self.BkgJTermError("x.ndim==1")
        if len(x)<>self.grid.N:
            raise self.BkgJTermError("len(x)==self.nlModel.grid.N")

    #------------------------------------------------------

    def _costFunc(self, x): 
        self.__xValidate(x)
        inno=(x-self.bkg)
        return 0.5*np.dot(inno, np.dot(self.metric, inno)) 

    #------------------------------------------------------

    def _gradCostFunc(self, x):
        self.__xValidate(x)
        inno=(x-self.bkg)
        return -np.dot(self.metric, inno)

#=====================================================================
#---------------------------------------------------------------------
#=====================================================================

class StaticObsJTerm(JTerm):
    """
    Static observations JTerm subclass

    StaticObsJTerm(obs, g)

        obs             :   <StaticObs>
        g               :   <PeriodicGrid>
    """
    class StaticObsJTermError(Exception):
        pass
        
        
    #------------------------------------------------------
    #----| Init |------------------------------------------
    #------------------------------------------------------

    def __init__(self, obs, g, maxGradNorm=None): 

        if not isinstance(obs, StaticObs):
            raise self.StaticObsJTermError("obs <SaticObs>")
        self.obs=obs
        self.nObs=self.obs.nObs

        if not isinstance(g, PeriodicGrid):
            raise self.StaticObsJTermError("g <pseudoSpec1D.PeriodicGrid>")
        self.modelGrid=g

        self.obsOpTLMAdj=self.obs.obsOpTLMAdj
        self.obsOpTLMAdjArgs=self.obs.obsOpArgs

        if not (isinstance(maxGradNorm, float) or maxGradNorm==None):
            raise self.StaticObsJTermError("maxGradNorm <None|float>")
        self.maxGradNorm=maxGradNorm 
        self.args=()

        self.isMinimized=False
        
    #------------------------------------------------------
    #----| Private methods |-------------------------------
    #------------------------------------------------------

    def __xValidate(self, x):
        if not isinstance(x, np.ndarray):
            raise self.StaticObsJTermError("x <numpy.array>")
        if not x.dtype=='float64':
            raise self.StaticObsJTermError("x.dtype=='float64'")
        if x.ndim<>1:
            raise self.StaticObsJTermError("x.ndim==1")
        if len(x)<>self.modelGrid.N:
            raise self.StaticObsJTermError("len(x)==self.nlModel.grid.N")

    #------------------------------------------------------

    def _costFunc(self, x, normalize=False): 
        self.__xValidate(x)
        inno=self.obs.innovation(x, self.modelGrid)
        if normalize:
            return (0.5/self.nObs)*np.dot(
                                inno, np.dot(self.obs.metric, inno)) 
        else:
            return 0.5*np.dot(inno, np.dot(self.obs.metric, inno)) 

    #------------------------------------------------------

    def _gradCostFunc(self, x, normalize=False):
        self.__xValidate(x)
        inno=self.obs.innovation(x, self.modelGrid)
        if self.obsOpTLMAdj==None:
            grad= -np.dot(self.obs.metric, inno)
        else:
            grad= -self.obsOpTLMAdj(np.dot(self.obs.metric, inno),
                                            self.modelGrid,
                                            self.obs.coord,
                                            *self.obsOpTLMAdjArgs)

        if normalize:
            grad= (1./self.nObs)*grad
        else:
            pass 
        return grad

#=====================================================================
#---------------------------------------------------------------------
#=====================================================================

class TWObsJTerm(JTerm):
    """    
    Time window observations JTerm subclass

    TWObsJTerm(obs, nlModel, tlm)

        obs             :   <StaticObs>
        nlModel         :   propagator model <Launcher>
        tlm             :   tangean linear model <TLMLauncher>
    """
    
    class TWObsJTermError(Exception):
        pass

    #------------------------------------------------------
    #----| Init |------------------------------------------
    #------------------------------------------------------

    def __init__(self, obs, nlModel, tlm, 
                    t0=0., tf=None,
                    maxGradNorm=None): 

        if not isinstance(obs, TimeWindowObs):
            raise self.TWObsJTermError("obs <TimeWindowObs>")


        self.tWin=np.zeros(2)
        self.tWin[0]=t0
        if tf==None : 
            self.tWin[1]=obs.tMax
        else:
            self.tWin[1]=tf



        self.obs=self.__obsCycleExtract(obs)
        self.nTimes=self.obs.nTimes
        self.nObs=self.obs.nObs

        if not (isinstance(nlModel,Launcher)):
            raise self.TWObsJTermError("nlModel <Launcher>")
        if not (isinstance(tlm, TLMLauncher)):
            raise self.TWObsJTermError("tlm <TLMLauncher>")        
        if not (nlModel.param==tlm.param):
            raise self.TWObsJTermError("nlModel.param==tlm.param")
        self.nlModel=nlModel
        self.tlm=tlm
        self.modelGrid=nlModel.grid

        if not (isinstance(maxGradNorm, float) or maxGradNorm==None):
            raise self.TWObsJTermError("maxGradNorm <None|float>")
        self.maxGradNorm=maxGradNorm 
        self.args=()

        self.isMinimized=False
        
    #------------------------------------------------------
    #----| Private methods |-------------------------------
    #------------------------------------------------------

    def __obsCycleExtract(self, obs):
        obsTimes=obs.d_Obs.keys()
        obsTimes.sort()

        d_ObsExt={}
        for t in obsTimes:
            if t>self.tWin[0] and t <=self.tWin[1] : 
                d_ObsExt[t]=obs.d_Obs[t]
    
        return TimeWindowObs(d_ObsExt)
    
    #------------------------------------------------------

    def __xValidate(self, x):
        if not isinstance(x, np.ndarray):
            raise self.TWObsJTermError("x <numpy.array>")
        if not x.dtype=='float64':
            raise self.TWObsJTermError("x.dtype=='float64'")
        if x.ndim<>1:
            raise self.TWObsJTermError("x.ndim==1")
        if len(x)<>self.nlModel.grid.N:
            raise self.TWObsJTermError("len(x)==self.nlModel.grid.N")
            
    #------------------------------------------------------

    def _costFunc(self, x): 
        self.__xValidate(x)
        d_inno=self.obs.innovation(x, self.nlModel, t0=self.tWin[0])
        Jo=0.5*self.obs.prosca(d_inno, d_inno)
        return Jo

    #------------------------------------------------------

    def _gradCostFunc(self, x):
        self.__xValidate(x)
        d_inno=self.obs.innovation(x, self.nlModel, t0=self.tWin[0])
        d_NormInno={}
        for t in d_inno.keys():
            d_NormInno[t]=np.dot(self.obs[t].metric,d_inno[t])
        
        grad=-self.obs.modelEquivalent_Adj(d_NormInno, self.tlm, 
                                            t0=self.tWin[0])
        return grad


#=====================================================================
#---------------------------------------------------------------------
#=====================================================================

if __name__=="__main__":
    import matplotlib.pyplot as plt
    import pyKdV as kdv
    from observations import rndSampling,  obsOp_Coord, obsOp_Coord_Adj

    testOpHAdj=False
    testOpHGrad=False
    testGradJ=True
    
    Ntrc=144
    g=kdv.PeriodicGrid(Ntrc)
    kdvParam=kdv.Param(g)
    model=kdv.kdvLauncher(kdvParam, dt=0.01)
    tlm=kdv.kdvTLMLauncher(kdvParam)

    x0=kdv.rndSpecVec(g, Ntrc=10,  amp=1., seed=0)
    x0+=1.5*kdv.gauss(g.x, 40., 20. )-1.*kdv.gauss(g.x, -20., 14. )
    pert=kdv.rndSpecVec(g, Ntrc=10,  amp=.1, seed=1)
    xt=x0+pert
    
    tInt=10.
    traj=model.integrate(xt, tInt)
    tlm.reference(traj)
    
    nObs=10
    freqObs=2
    d_Obs={}
    for tObs in [i*tInt/freqObs for i in xrange(1,freqObs+1)]:
        coords=rndSampling(g, nObs, seed=tObs)
        d_Obs[tObs]=StaticObs(coords, 
                              traj.whereTime(tObs)[g.pos2Idx(coords)],
                              obsOp_Coord, obsOp_Coord_Adj)
    twObs1=TimeWindowObs(d_Obs)

    if testOpHAdj:
        #----| AD chain adjoint test |----------------
        print("\nTesting AD Chain adjoint") 
        # <y, Hx> - <H*y, x>
        print("  1: x -( H )-> Hx ")
        x=kdv.rndSpecVec(g, amp=0.1, seed=1)
        y=twObs1.modelEquivalent(kdv.rndSpecVec(g, amp=0.1, seed=2), tlm)
    
        Hx=twObs1.modelEquivalentTLM(x, tlm)
        Ay=twObs1.modelEquivalent_Adj(y, tlm)
        
        y_Hx=0.
        for t in y.keys():
            y_Hx+=np.dot(y[t], Hx[t])
        Ay_x=np.dot(Ay, x)
        print("    <y, Hx> - <H*y, x>=%e\n"%(y_Hx-Ay_x))
    
    if testOpHGrad:
        powRange=[-1,-14]
        x=kdv.rndSpecVec(g, amp=1., seed=1)
        NLx=twObs1.modelEquivalent(x, model)
        J1=0.
        for t in NLx.keys():
            J1+=0.5*np.dot(NLx[t], NLx[t])

        gradJ1=twObs1.modelEquivalent_Adj(NLx, tlm)
        n2GradJ1=np.dot(gradJ1,gradJ1)

        test={}
        for power in xrange(powRange[0],powRange[1], -1):
            eps=10.**(power)
            NLx_eps=twObs1.modelEquivalent(x-eps*gradJ1, model)
            
            Jeps=0.
            for t in NLx_eps.keys():
                Jeps+=0.5*np.dot(NLx_eps[t], NLx_eps[t]) 
            res=((J1-Jeps)/(eps*n2GradJ1))
            test[power]=[Jeps, res]
            print(test[power])


    if testGradJ:
        J2=TWObsJTerm(twObs1, model, tlm)
        x=kdv.rndSpecVec(g, amp=1., seed=1)
        J2.gradTest(x)
