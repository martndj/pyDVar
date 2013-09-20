from jTerm import JTerm
from observations import StaticObs, TimeWindowObs
from pseudoSpec1D import PeriodicGrid, Launcher, TLMLauncher
import numpy as np

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

    def __init__(self, obs, g): 

        if not isinstance(obs, StaticObs):
            raise StaticObsJTermError("obs <SaticObs>")
        self.obs=obs

        if not isinstance(g, PeriodicGrid):
            raise StaticObsJTermError("g <pseudoSpec1D.PeriodicGrid>")
        self.modelGrid=g

        self.obsOpTLMAdj=self.obs.obsOpTLMAdj
        self.obsOpTLMAdjArgs=self.obs.obsOpArgs

        self.args=()
        
    #------------------------------------------------------
    #----| Private methods |-------------------------------
    #------------------------------------------------------

    def __xValidate(self, x):
        if not isinstance(x, np.ndarray):
            raise self.TWObsJTermError("x <numpy.array>")
        if not x.dtype=='float64':
            raise self.TWObsJTermError("x.dtype=='float64'")
        if x.ndim<>1:
            raise self.TWObsJTermError("x.ndim==1")
        if len(x)<>self.modelGrid.N:
            raise self.TWObsJTermError("len(x)==self.nlModel.grid.N")

    #------------------------------------------------------
    #----| Public methods |--------------------------------
    #------------------------------------------------------

    def J(self, x): 
        self.__xValidate(x)
        inno=self.obs.innovation(x, self.modelGrid)
        return 0.5*np.dot(inno, np.dot(self.obs.metric, inno)) 

    #------------------------------------------------------

    def gradJ(self, x):
        self.__xValidate(x)
        inno=self.obs.innovation(x, self.modelGrid)
        if self.obsOpTLMAdj==None:
            return -np.dot(self.obs.metric, inno)
        else:
            return -self.obsOpTLMAdj(np.dot(self.obs.metric, inno),
                                            self.modelGrid,
                                            self.obs.coord,
                                            *self.obsOpTLMAdjArgs)

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

    def __init__(self, obs, nlModel, tlm): 

        if not isinstance(obs, TimeWindowObs):
            raise self.TWObsJTermError("obs <TimeWindowObs>")
        self.obs=obs
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


        self.args=()
        
    #------------------------------------------------------
    #----| Private methods |-------------------------------
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

    def J(self, x): 
        self.__xValidate(x)
        d_inno=self.obs.innovation(x, self.nlModel)
        Jo=0.
        for t in self.obs.times:
            Jo+=0.5*np.dot(d_inno[t],np.dot(self.obs[t].metric,d_inno[t]))
        return Jo

    #------------------------------------------------------

    def gradJ(self, x):
        self.__xValidate(x)
        d_inno=self.obs.innovation(x, self.nlModel)
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
            if i<self.nObs:
                t_pre=self.obs.times[-1-i]
            else:
                t_pre=0.

            if self.obs[t].obsOpTLMAdj==None:
                w=d_NormInno[t]
            else:   
                w=self.obs[t].obsOpTLMAdj(d_NormInno[t], 
                                            self.nlModel.grid,
                                            self.obs[t].coord,
                                            *self.obs[t].obsOpArgs)

            MAdjObs=self.tlm.adjoint(w+MAdjObs, tInt=t-t_pre, t0=t_pre)
            w=MAdjObs
        
        return -MAdjObs

#=====================================================================
#---------------------------------------------------------------------
#=====================================================================


if __name__=='__main__':

    import matplotlib.pyplot as plt
    from observations import degrad, pos2Idx, obsOp_Coord, obsOp_Coord_Adj
    import pyKdV as kdv
    from jTerm import TrivialJTerm
    
    Ntrc=100
    L=300.
    g=PeriodicGrid(Ntrc, L)
    

    rndLFBase=kdv.rndFiltVec(g, Ntrc=g.Ntrc/5,  amp=0.4)
    soliton=kdv.soliton(g.x, 0., amp=1.3, beta=1., gamma=-1)

    x0_truth=soliton
    x0_degrad=degrad(x0_truth, 0., 0.3)
    x0_bkg=np.zeros(g.N)

    print("=======================================================")
    print("----| Static obs |-------------------------------------")
    print("=======================================================")
    obs1=StaticObs(g, x0_truth)
    obs2Pos=np.array([-50.,0., 50.])
    obs2=StaticObs(obs2Pos, x0_truth[pos2Idx(g, obs2Pos)],
                    obsOp_Coord, obsOp_Coord_Adj)
    JObs1=StaticObsJTerm(obs1, g) 
    JObs2=StaticObsJTerm(obs2, g) 
    
    #----| First test: only degrated first guess
    JObs1.minimize(x0_degrad)
    JObs2.minimize(x0_degrad)

    #----| Second test: with null background term
    #   the background term will constraint toward 0
    Jbkg=TrivialJTerm()
    JSum=JObs1+Jbkg*3.
    JSum.minimize(x0_degrad)
    
    plt.figure()
    plt.plot(g.x, x0_truth, 'k', linewidth=2)
    plt.plot(g.x, x0_degrad, 'b')
    plt.plot(g.x, JObs1.analysis, 'g')
    plt.plot(g.x, JObs2.analysis, 'm')
    plt.plot(JObs2.obs.interpolate(g), JObs2.obs.values, 'mo')
    plt.plot(g.x, JSum.analysis, 'r')
    
    print("\n\n=======================================================")
    print("----| Dynamic obs |------------------------------------")
    print("=======================================================")
    def gaussProfile(x):
        return 0.03*kdv.gauss(x, 40., 20. )\
                -0.02*kdv.gauss(x, -20., 14. )

    kdvParam=kdv.Param(g, beta=1., gamma=-1., rho=gaussProfile)
    tInt=20.
    maxA=2.
    maxiter=50
    
    model=kdv.kdvLauncher(kdvParam, maxA)
    tlm=kdv.kdvTLMLauncher(kdvParam)
    x_truth=model.integrate(x0_truth, tInt)
    x_degrad=model.integrate(x0_degrad, tInt)
    x_bkg=model.integrate(x0_bkg, tInt)

    nObsTime=3
    d_Obs1={}
    for i in xrange(nObsTime):
        d_Obs1[tInt*(i+1)/nObsTime]=StaticObs(g,
            x_truth.whereTime(tInt*(i+1)/nObsTime))
    timeObs1=TimeWindowObs(d_Obs1)

    JTWObs=TWObsJTerm(timeObs1, model, tlm) 
    JTWObs.minimize(x0_bkg)
    x_a=model.integrate(JTWObs.analysis, tInt)

    plt.figure()
    i=0
    for t in timeObs1.times:
        i+=1
        sub=plt.subplot(nObsTime, 1, i)
        sub.plot(g.x, x_truth.whereTime(t), 'k', linewidth=2.5)
        sub.plot(timeObs1[t].interpolate(g), timeObs1[t].values, 'g')
        sub.plot(g.x, x_a.whereTime(t), 'r')
        sub.set_title("t=%.2f"%t)
    plt.show()
