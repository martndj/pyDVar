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
        self.args=(self.maxGradNorm, )

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
    #----| Public methods |--------------------------------
    #------------------------------------------------------

    def J(self, x, maxNorm=None): 
        self.__xValidate(x)
        inno=(x-self.bkg)
        return 0.5*np.dot(inno, np.dot(self.metric, inno)) 

    #------------------------------------------------------

    def gradJ(self, x, maxNorm=None):
        self.__xValidate(x)
        inno=(x-self.bkg)
        grad=-np.dot(self.metric, inno)
        if maxNorm==None:
            return grad
        elif isinstance(maxNorm, float):
            normGrad=norm(grad)
            if normGrad>maxNorm:
                grad=(grad/normGrad)*(maxNorm)
            return grad
        else:
            raise self.BkgJTermError("maxNorm <float>")

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
        self.args=(self.maxGradNorm, )

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
    #----| Public methods |--------------------------------
    #------------------------------------------------------

    def J(self, x, maxNorm=None, normalize=False): 
        self.__xValidate(x)
        inno=self.obs.innovation(x, self.modelGrid)
        if normalize:
            return (0.5/self.nObs)*np.dot(
                                inno, np.dot(self.obs.metric, inno)) 
        else:
            return 0.5*np.dot(inno, np.dot(self.obs.metric, inno)) 

    #------------------------------------------------------

    def gradJ(self, x, maxNorm=None, normalize=False):
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
        if maxNorm==None:
            return grad
        elif isinstance(maxNorm, float):
            normGrad=norm(grad)
            if normGrad>maxNorm:
                grad=(grad/normGrad)*(maxNorm)
            return grad
        else:
            raise self.StaticObsJTermError("maxNorm <float>")


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
                    t0=0., tFinal=None,
                    maxGradNorm=None): 

        if not isinstance(obs, TimeWindowObs):
            raise self.TWObsJTermError("obs <TimeWindowObs>")


        self.tWin=np.zeros(2)
        self.tWin[0]=t0
        if tFinal==None : 
            self.tWin[1]=obs.tMax
        else:
            self.tWin[1]=tFinal



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
        self.args=(self.maxGradNorm, )

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
    #----| Public methods |--------------------------------
    #------------------------------------------------------

    def J(self, x, maxNorm=None, normalize=False): 
        self.__xValidate(x)
        d_inno=self.obs.innovation(x, self.nlModel, t0=self.tWin[0])
        Jo=0.
        for t in self.obs.times:
            Jo+=0.5*np.dot(d_inno[t],np.dot(self.obs[t].metric,d_inno[t]))
        if normalize:
            return (1./self.nObs)*Jo
        else:
            return Jo

    #------------------------------------------------------

    def gradJ(self, x, maxNorm=None, normalize=False):
        self.__xValidate(x)
        d_inno=self.obs.innovation(x, self.nlModel, t0=self.tWin[0])
        d_NormInno={}
        for t in d_inno.keys():
            d_NormInno[t]=np.dot(self.obs[t].metric,d_inno[t])
        #----| building reference trajectory |--------
        tInt=np.max(self.obs.times)
        traj_x=self.nlModel.integrate(x, tInt)
        self.tlm.initialize(traj_x)
        #----| Adjoint retropropagation |-------------
        i=0
        MAdjObs=np.zeros(self.nlModel.grid.N)
        for t in self.obs.times[::-1]:
            i+=1
            if i<self.nTimes:
                t_pre=self.obs.times[-1-i]
            else:
                t_pre=self.tWin[0]

            if self.obs[t].obsOpTLMAdj==None:
                w=d_NormInno[t]
            else:   
                w=self.obs[t].obsOpTLMAdj(d_NormInno[t], 
                                            self.nlModel.grid,
                                            self.obs[t].coord,
                                            *self.obs[t].obsOpArgs)

            MAdjObs=self.tlm.adjoint(w+MAdjObs, tInt=t-t_pre, t0=t_pre)
            w=MAdjObs
        
        grad=-MAdjObs
        if normalize:
            grad= (1./self.nObs)*grad
        else:
            pass
        
        if maxNorm==None:
            return grad
        elif isinstance(maxNorm, float):
            normGrad=norm(grad)
            if normGrad>maxNorm:
                grad=(grad/normGrad)*(maxNorm)
            return grad
        else:
            raise self.TWObsJTermError("maxNorm <float>")


#=====================================================================
#---------------------------------------------------------------------
#=====================================================================


if __name__=='__main__':
