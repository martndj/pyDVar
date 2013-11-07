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
        self.nObs=self.obs.nObs

        if not isinstance(g, PeriodicGrid):
            raise StaticObsJTermError("g <pseudoSpec1D.PeriodicGrid>")
        self.modelGrid=g

        self.obsOpTLMAdj=self.obs.obsOpTLMAdj
        self.obsOpTLMAdjArgs=self.obs.obsOpArgs

        self.args=()
        self.isMinimized=False
        
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

    def J(self, x, normalize=False): 
        self.__xValidate(x)
        inno=self.obs.innovation(x, self.modelGrid)
        if normalize:
            return (0.5/self.nObs)*np.dot(
                                inno, np.dot(self.obs.metric, inno)) 
        else:
            return 0.5*np.dot(inno, np.dot(self.obs.metric, inno)) 

    #------------------------------------------------------

    def gradJ(self, x, normalize=False):
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
            return (1./self.nObs)*grad
        else:
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

    def __init__(self, obs, nlModel, tlm): 

        if not isinstance(obs, TimeWindowObs):
            raise self.TWObsJTermError("obs <TimeWindowObs>")
        self.obs=obs
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


        self.args=()
        self.isMinimized=False
        
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

    def J(self, x, normalize=False): 
        self.__xValidate(x)
        d_inno=self.obs.innovation(x, self.nlModel)
        Jo=0.
        for t in self.obs.times:
            Jo+=0.5*np.dot(d_inno[t],np.dot(self.obs[t].metric,d_inno[t]))
        if normalize:
            return (1./self.nObs)*Jo
        else:
            return Jo

    #------------------------------------------------------

    def gradJ(self, x, normalize=False):
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
            if i<self.nTimes:
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
        
        if normalize:
            return -(1./self.nObs)*MAdjObs
        else:
            return -MAdjObs



#=====================================================================
#---------------------------------------------------------------------
#=====================================================================


if __name__=='__main__':

    import matplotlib.pyplot as plt
    from observations import degrad, obsOp_Coord, obsOp_Coord_Adj
    import pyKdV as kdv
    
    minimize=True
    staticObs=False
    TWObs=True
    convergencePlot=True
    Ntrc=120
    L=300.
    g=PeriodicGrid(Ntrc, L)
    
    #----| Nature run (truth) |-------------
    rndLFBase=kdv.rndSpecVec(g, Ntrc=8)
    x0Soliton=0.
    soliton=kdv.soliton(g.x, x0Soliton, amp=3., beta=1., gamma=-1)
        
    x0_truth=soliton
    x0_degrad=degrad(x0_truth, 0., 0.3)
    x0_bkg=np.zeros(g.N)

    if staticObs:
        print("=======================================================")
        print("----| Static obs |-------------------------------------")
        print("=======================================================")
        obs1=StaticObs(g, x0_truth)
        obs2Pos=np.array([-50.,0., 50.])
        obs2=StaticObs(obs2Pos, x0_truth[g.pos2Idx(obs2Pos)],
                        obsOp_Coord, obsOp_Coord_Adj)
        JObs1=StaticObsJTerm(obs1, g) 
        JObs2=StaticObsJTerm(obs2, g) 
        
        #----| First test: only degrated first guess
        JObs1.minimize(x0_degrad)
        JObs2.minimize(x0_degrad)
    
        #----| Second test: with null background term
        #   the background term will constraint toward 0
        
        plt.figure()
        plt.plot(g.x, x0_truth, 'k', linewidth=2)
        plt.plot(g.x, x0_degrad, 'b')
        plt.plot(g.x, JObs1.analysis, 'g')
        plt.plot(g.x, JObs2.analysis, 'm')
        plt.plot(JObs2.obs.interpolate(g), JObs2.obs.values, 'mo')
    
    if TWObs:
        print("\n\n=======================================================")
        print("----| Dynamic obs |------------------------------------")
        print("=======================================================")
                
        def gauss(x):
            x0=0.
            sig=5.
            return -0.1*np.exp(-((x-x0)**2)/(2*sig**2))
                
        kdvParam=kdv.Param(g, beta=1., gamma=-1)#, rho=gauss)
        dt=kdv.dtStable(g, kdvParam, maxA=4., dtMod=0.5)   
        model=kdv.kdvLauncher(kdvParam, dt=dt )
        tlm=kdv.kdvTLMLauncher(kdvParam)
       
        tInt=10. 
        
        x_truth=model.integrate(x0_truth, tInt)
        tlm.initialize(x_truth)
        
        #----| Observations |-------------------
        sigma=0.1
        x_obs=x_truth
        
        nTObs=4
        t_tObs=[]
        for i in xrange(nTObs):
            t_tObs.append((i+1)*tInt/(nTObs))
        t_tObs.sort()
        
        nPosObs=30
        dxPosObs=2.
        d_Obs={}
        d_posObs={}
        
        for t in t_tObs:
            xSol=x0Soliton+kdv.cSoliton(amp=3., beta=1., gamma=-1)*t
            d_posObs[t]=np.zeros(nPosObs)
            d_posObs[t][0]=xSol
            for i in xrange(1,nPosObs):
                if i%2:
                    d_posObs[t][i]=xSol+(i/2+1)*dxPosObs
                else:
                    d_posObs[t][i]=xSol-((i-1)/2+1)*dxPosObs
        
        
        for t in t_tObs:
            R_inv=np.ones(nPosObs)*sigma**(-1)
            obsValues=obsOp_Coord(x_obs.whereTime(t), g, d_posObs[t])
        
            d_Obs[t]=StaticObs(d_posObs[t], obsValues, 
                                obsOp=obsOp_Coord, 
                                obsOpTLMAdj=obsOp_Coord_Adj, 
                                metric=R_inv)
        
        timeObs=TimeWindowObs(d_Obs)
        
        #----| First Guess |--------------------
        x0_bkg=np.zeros(g.N)
        
        Jo=TWObsJTerm(timeObs, model, tlm)
        J=Jo
        
        if minimize:
            #----| Minimizing |---------------------
            print("\nMinimizing J...")
            #J.minimize(np.zeros(g.N))
            J.minimize(x0_bkg, maxiter=50)
            print("Analysis innovation:\n ")
            print(timeObs.innovation(J.analysis, model))
            
            
            
            #----| Integrating trajectories |-------
            x0_a=J.analysis
            x_a=model.integrate(x0_a, tInt)
            print("\nTruth-Analysis amplitude:\n ")
            print(" initial time: %f"%g.norm(x0_truth-x0_a))
            print(" final time:   %f"%g.norm(x_truth.final-x_a.final))
            
            
            timeObs.plot(g, trajectory=x_truth, trajectoryStyle='k:')
            timeObs.plot(g, trajectory=x_a, trajectoryStyle='r')
            
            
            plt.figure(figsize=(12.,6.))
            sub=plt.subplot(1,1,1)
            sub.plot(g.x, x0_truth, 'g')
            sub.plot(g.x, x_truth.final, 'b')
            sub.plot(g.x, x0_a, 'r')
            sub.plot(g.x, x_a.final, 'm')
            sub.plot(g.x, x0_bkg, 'k')
            sub.legend(["${x_t}_0$", "${x_t}_f$",
                        "${x_a}_0$","${x_a}_f$",
                        "${x_b}_0$",
                        ], loc='best')

            if convergencePlot:
                print("\nAnalyzing convergence speed...")
                plt.figure()
                plt.semilogy(J.convergence())
                plt.xlabel("$n$ (# iterations)")
                plt.ylabel("$J$")
    plt.show()
