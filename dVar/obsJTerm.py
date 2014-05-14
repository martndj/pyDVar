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

    #------------------------------------------------------
    #----| Init |------------------------------------------
    #------------------------------------------------------

    def __init__(self, bkg, g, metric=1., maxGradNorm=None): 

        if not isinstance(g, PeriodicGrid):
            raise TypeError("g <pseudoSpec1D.PeriodicGrid>")
        self.grid=g

        if not isinstance(bkg, np.ndarray):
            raise TypeError("bkg <numpy.ndarray>")
        if not (bkg.ndim==1 and len(bkg)==g.N):
            raise ValueError("bkg.shape==(g.N,)")
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
                raise ValueError("metric.ndim=[1|2]")
        else:   
            raise TypeError("metric <None | numpy.ndarray>")


        if not (isinstance(maxGradNorm, float) or maxGradNorm==None):
            raise TypeError("maxGradNorm <None|float>")
        self.maxGradNorm=maxGradNorm 
        self.args=()

        self.isMinimized=False
        
    #------------------------------------------------------
    #----| Private methods |-------------------------------
    #------------------------------------------------------

    def __xValidate(self, x):
        if not isinstance(x, np.ndarray):
            raise TypeError("x <numpy.array>")
        if not x.dtype=='float64':
            raise TypeError("x.dtype=='float64'")
        if x.ndim<>1:
            raise ValueError("x.ndim==1")
        if len(x)<>self.grid.N:
            raise ValueError("len(x)==self.nlModel.grid.N")

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
        
    #------------------------------------------------------
    #----| Init |------------------------------------------
    #------------------------------------------------------

    def __init__(self, obs, g, maxGradNorm=None): 

        if not isinstance(obs, StaticObs):
            raise TypeError("obs <SaticObs>")
        self.obs=obs
        self.nObs=self.obs.nObs

        if not isinstance(g, PeriodicGrid):
            raise TypeError("g <pseudoSpec1D.PeriodicGrid>")
        self.modelGrid=g

        self.obsOpTLMAdj=self.obs.obsOpTLMAdj
        self.obsOpTLMAdjArgs=self.obs.obsOpArgs

        if not (isinstance(maxGradNorm, float) or maxGradNorm==None):
            raise TypeError("maxGradNorm <None|float>")
        self.maxGradNorm=maxGradNorm 
        self.args=()

        self.isMinimized=False
        
    #------------------------------------------------------
    #----| Private methods |-------------------------------
    #------------------------------------------------------

    def __xValidate(self, x):
        if not isinstance(x, np.ndarray):
            raise TypeError("x <numpy.array>")
        if not x.dtype=='float64':
            raise TypeError("x.dtype=='float64'")
        if x.ndim<>1:
            raise ValueError("x.ndim==1")
        if len(x)<>self.modelGrid.N:
            raise ValueError("len(x)==self.nlModel.grid.N")

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
    
    #------------------------------------------------------
    #----| Init |------------------------------------------
    #------------------------------------------------------

    def __init__(self, obs, nlModel, tlm, 
                    t0=0., tf=None,
                    maxGradNorm=None): 

        if not isinstance(obs, TimeWindowObs):
            raise TypeError("obs <TimeWindowObs>")


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
            raise TypeError("nlModel <Launcher>")
        if not (isinstance(tlm, TLMLauncher)):
            raise TypeError("tlm <TLMLauncher>")        
        if not (nlModel.param==tlm.param):
            raise ValueError("nlModel.param==tlm.param")
        self.nlModel=nlModel
        self.tlm=tlm
        self.modelGrid=nlModel.grid

        if not (isinstance(maxGradNorm, float) or maxGradNorm==None):
            raise TypeError("maxGradNorm <None|float>")
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
            raise TypeError("x <numpy.array>")
        if not x.dtype=='float64':
            raise TypeError("x.dtype=='float64'")
        if x.ndim<>1:
            raise TypeError("x.ndim==1")
        if len(x)<>self.nlModel.grid.N:
            raise TypeError("len(x)==self.nlModel.grid.N")
            
    #------------------------------------------------------

    def _costFunc(self, x): 
        self.__xValidate(x)
        d_inno=self.obs.innovation(x, self.nlModel, t0=self.tWin[0])
        Jo=0.5*self.obs.prosca(d_inno, d_inno)
        return Jo

    #------------------------------------------------------

    def _gradCostFunc(self, x):
        self.__xValidate(x)
        self.tlm.reference(self.nlModel.integrate(
                                x, 
                                self.obs.times[-1]-self.tWin[0],
                                t0=self.tWin[0]))
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

    dummyModel=False

    testGradJStatObs=True
    testGradJTWObs=True
    
    Ntrc=144
    tTotal=5.
    t0=3.
    tInt=tTotal-t0
    dt=0.01
    nObs=10
    freqObs=3


    g=kdv.PeriodicGrid(Ntrc)
    if not dummyModel:
        kdvParam=kdv.Param(g)
        if dt > kdv.dtStable(kdvParam, 10.): raise RuntimeError()
        model=kdv.kdvLauncher(kdvParam, dt=dt)
        tlm=kdv.kdvTLMLauncher(kdvParam)
    else:
        model=kdv.IdLauncher(g, dt=dt)
        tlm=kdv.IdTLM(g)

    x0=kdv.rndSpecVec(g, Ntrc=10,  amp=1., seed=0)
    x0+=1.5*kdv.gauss(g.x, 40., 20. )-1.*kdv.gauss(g.x, -20., 14. )
    pert=kdv.rndSpecVec(g, Ntrc=10,  amp=.1, seed=1)
    xt=x0+pert
    
    traj=model.integrate(xt, tTotal)
    #tlm.reference(traj)
    
    d_Obs={}
    for tObs in [i*tTotal/freqObs for i in xrange(1,freqObs+1)]:
        coords=rndSampling(g, nObs, seed=tObs)
        d_Obs[tObs]=StaticObs(coords, 
                              traj.whereTime(tObs)[g.pos2Idx(coords)],
                              obsOp_Coord, obsOp_Coord_Adj)
    twObs1=TimeWindowObs(d_Obs)

    if testGradJStatObs:
        print("\nStaticObsJTerm gradient test") 
        obs=twObs1[tTotal]
        J2=StaticObsJTerm(obs, g)
        x=kdv.rndSpecVec(g, amp=1., seed=1)
        J2.gradTest(x)


    if testGradJTWObs:
        print("\nTWObsJTerm gradient test") 
        J3=TWObsJTerm(twObs1, model, tlm, t0=t0)
        x=kdv.rndSpecVec(g, amp=1., seed=1)
        J3.gradTest(x0)
