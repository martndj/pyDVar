import numpy as np
from pseudoSpec1D import PeriodicGrid, Launcher
import random as rnd

#-----------------------------------------------------------
#----| Utilitaries |----------------------------------------
#-----------------------------------------------------------

def degrad(signal,mu,sigma,seed=0.7349156729):
    ''' 
    Gaussian noise signal degradation

    degrad(u,mu,sigma,seed=...)

    u       :  input signal
    mu      :  noise mean (gaussian mean)
    sigma   :  noise variance
    '''
    rnd.seed(seed)
    sig_degrad=signal.copy()
    for i in xrange(signal.size):
        sig_degrad[i]=signal[i]+rnd.gauss(mu, sigma)
    return sig_degrad

#-----------------------------------------------------------

def pos2Idx(g, pos):
    """
    Convert space position to grid index
    """
    if not isinstance(pos, np.ndarray):
        raise ObservationOpError("pos <numpy.ndarray>")
    if pos.ndim<>1:
        raise ObservationOpError("pos.ndim=1")
    N=len(pos)
    idx=np.zeros(N, dtype=int)
    for i in xrange(N):
        idx[i]=np.min(np.where(g.x>=pos[i]))
    return idx

#-----------------------------------------------------------
#----| Observation operators |------------------------------
#-----------------------------------------------------------

def obsOp_Coord(x, g, obsCoord):
    """
    Trivial static observation operator
    """
    idxObs=pos2Idx(g, obsCoord)
    nObs=len(idxObs)
    H=np.zeros(shape=(nObs,g.N))
    for i in xrange(nObs):
        H[i, idxObs[i]]=1.
    return np.dot(H,x)

def obsOp_Coord_Adj(obs, g, obsCoord):
    """
    Trivial static observation operator adjoint
    """
    idxObs=pos2Idx(g, obsCoord)
    nObs=len(idxObs)
    H=np.zeros(shape=(nObs,g.N))
    for i in xrange(nObs):
        H[i, idxObs[i]]=1.
    return np.dot(H.T,obs)

#=====================================================================
#---------------------------------------------------------------------
#=====================================================================

class StaticObs(object):
    """
    StaticObs class

    StaticObs(coord, values, obsOp, obsOpTLMAdj, obsOpArgs=())
        coord       :   observation positions
                            <pseudoSpec1D.PeriodicGrid | numpy.ndarray>
                            (PeriodicGrid for continuous observations)
        values      :   observation values <numpy.ndarray>
        obsOp       :   static observation operator <function | None>
        obsOpTLMAdj :   static observation TLM adjoint <function | None>
                            (both None when observation space = 
                             model space)
        obsOpArgs   :   obsOp additional arguments
                            obsOp(x_state, x_grid, x_obsSpaceCoord, 
                                    *obsOpArgs)
    """
    class StaticObsError(Exception):
        pass


    #------------------------------------------------------
    #----| Init |------------------------------------------
    #------------------------------------------------------

    def __init__(self, coord, values, obsOp=None, obsOpTLMAdj=None,
                    obsOpArgs=(), metric=None):

        if isinstance(coord, PeriodicGrid):
            self.coord=coord.x
            self.nObs=coord.N
        elif isinstance(coord, np.ndarray):
            if coord.ndim <> 1:
                raise self.StaticObsError("coord.ndim==1")
            self.coord=coord
            self.nObs=len(coord)
        else:
            raise self.StaticObsError(
                "coord <pseudoSpec1D.PeriodicGrid | numpy.ndarray>")

        if not isinstance(values, np.ndarray):
            raise self.StaticObsError("coord <numpy.ndarray>")
        if values.ndim<>1 or len(values)<>self.nObs:
            raise self.StaticObsError("len(values)==self.nObs")
        self.values=values

        if not ((callable(obsOp) and callable(obsOpTLMAdj)) or 
                (obsOp==None and obsOpTLMAdj==None)):
            raise self.StaticObsError(
                                "obsOp, obsOpTLMAdj <function | None>")
        if not isinstance(obsOpArgs, tuple):
            raise self.StaticObsError("obsOpArgs <tuple>")
        self.obsOp=obsOp
        self.obsOpTLMAdj=obsOpTLMAdj
        self.obsOpArgs=obsOpArgs

        if metric==None:
            self.metric=np.eye(self.nObs)
        elif isinstance(metric, np.ndarray):
            if metric.ndim==1:
                self.metric=np.diag(metric)
            elif metric.ndom==2:
                self.metric=metric
            else:
                raise self.StaticObsError("metric.ndim=[1|2]")
        else:   
            raise self.StaticObsError("metric <None | numpy.ndarray>")
    #------------------------------------------------------
    #----| Private methods |-------------------------------
    #------------------------------------------------------

    def __pos2Idx(self, g):
        
        idx=np.zeros(self.nObs, dtype=int)
        for i in xrange(self.nObs):
            idx[i]=np.min(np.where(g.x>=self.coord[i]))
        return idx

    #------------------------------------------------------
    #----| Public methods |--------------------------------
    #------------------------------------------------------

    def modelEquivalent(self, x, g):
        
        if self.obsOp<>None:
            return self.obsOp(x, g, self.coord, *self.obsOpArgs)
        else:
            return x

    #------------------------------------------------------
    
    def innovation(self, x, g):
        return self.values-self.modelEquivalent(x, g)

    #------------------------------------------------------

    def interpolate(self, g):
        return g.x[self.__pos2Idx(g)]



#=====================================================================
#---------------------------------------------------------------------
#=====================================================================


class TimeWindowObs(object):
    """
    TimeWindowObs : discrete times observations class

        d_Obs       :   {time : <staticObs>} <dict>
        propagator  :   propagator launcher <Launcher>

    """

    class TimeWindowObsError(Exception):
        pass


    #------------------------------------------------------
    #----| Init |------------------------------------------
    #------------------------------------------------------

    def __init__(self, d_Obs):
        
        if not isinstance(d_Obs, dict):
            raise self.TimeWindowObsError("d_Obs <dict {time:<StaticObs>}>")
        for t in d_Obs.keys():
            if not (isinstance(t, (float,int)) 
                    and isinstance(d_Obs[t], StaticObs)):
                raise self.TimeWindowObsError(
                        "d_Obs <dict {time <float>: <StaticObs>}>")
            if d_Obs[t].obsOp<>d_Obs[d_Obs.keys()[0]].obsOp:
                raise self.TimeWindowObsError("all obsOp must be the same")
        self.times=np.sort(d_Obs.keys())
        self.tMax=self.times.max()
        self.d_Obs=d_Obs
        self.nObs=len(d_Obs)
        self.obsOp=d_Obs[d_Obs.keys()[0]].obsOp
        self.obsOpArgs=d_Obs[d_Obs.keys()[0]].obsOpArgs

                
    #------------------------------------------------------
    #----| Private methods |-------------------------------
    #------------------------------------------------------

    def __propagatorValidate(self, propagator):
        if not isinstance(propagator, Launcher):
            raise self.TimeWindowObsError("propagator <Launcher>")

    #------------------------------------------------------

    def __integrate(self, x, propagator):
        self.__propagatorValidate(propagator)
        d_xt={}
        t0=0.
        x0=x
        for t in self.times:
            if t==0.:
                d_xt[t]=x0
            else:
                d_xt[t]=(propagator.integrate(x0,t-t0)).final    
            x0=d_xt[t]
            t0=t
        
        return d_xt
    #------------------------------------------------------
    #----| Public methods |--------------------------------
    #------------------------------------------------------

    def modelEquivalent(self, x, propagator):
        self.__propagatorValidate(propagator)
        g=propagator.grid
        d_Hx={}
        d_xt=self.__integrate(x, propagator)
        for t in self.times:
            d_Hx[t]=self.d_Obs[t].modelEquivalent(d_xt[t], g)
        return d_Hx

    #------------------------------------------------------
    
    def innovation(self, x, propagator):
        self.__propagatorValidate(propagator)
        d_inno={}
        d_Hx=self.modelEquivalent(x, propagator)
        for t in self.times:
            d_inno[t]=self.d_Obs[t].values-d_Hx[t]
        return d_inno
        
    #------------------------------------------------------
    #----| Classical overloads |---------------------------
    #------------------------------------------------------

    def __getitem__(self, t):
        return self.d_Obs[t]

#=====================================================================
#---------------------------------------------------------------------
#=====================================================================

if __name__=="__main__":
    import matplotlib.pyplot as plt
    import pyKdV as kdv
    from pseudoSpec1D import PeriodicGrid
    
    #----| Static obs |---------------------------    
    Ntrc=100
    L=300.
    g=PeriodicGrid(Ntrc, L)
        

    x0_truth_base=kdv.rndFiltVec(g, Ntrc=g.Ntrc/5,  amp=1.)
    gaussWave=1.5*kdv.gauss(g.x, 40., 20. )-1.*kdv.gauss(g.x, -20., 14. )
    soliton=kdv.soliton(g.x, 0., amp=5., beta=1., gamma=-1)

    x0_truth=x0_truth_base+gaussWave
    x0_degrad=degrad(x0_truth, 0., 0.3)

    obs1=StaticObs(g, x0_degrad, None, ())

    obs2Coord=np.array([-50., 0., 70.])
    obs2=StaticObs(obs2Coord, x0_degrad[pos2Idx(g, obs2Coord)],
                    obsOp_Coord, ())


    plt.subplot(211)
    plt.title("Static observations")
    plt.plot(g.x, x0_degrad, 'r', linewidth=3)
    plt.plot(g.x, x0_truth, 'k', linewidth=3)
    plt.plot(obs1.interpolate(g), obs1.values, 'g')
    plt.plot(obs1.interpolate(g), obs1.modelEquivalent(x0_truth, g), 'b')
    plt.subplot(212)
    plt.plot(g.x, x0_degrad, 'r')
    plt.plot(g.x, x0_truth, 'k', linewidth=3)
    plt.plot(obs2.interpolate(g), obs2.values, 'go')
    plt.plot(obs2.interpolate(g), obs2.modelEquivalent(x0_truth, g), 'bo')

    #----| time window obs |----------------------
    kdvParam=kdv.Param(g, beta=1., gamma=-1.)
    tInt=10.
    maxA=5.
    
    model=kdv.kdvLauncher(kdvParam, maxA)
    x_truth=model.integrate(x0_truth, tInt)
    x_degrad=model.integrate(x0_degrad, tInt)

    nObsTime=3
    d_Obs1={}
    for i in xrange(nObsTime):
        d_Obs1[tInt*(i+1)/nObsTime]=StaticObs(g,
            x_degrad.whereTime(tInt*(i+1)/nObsTime), None)
    timeObs1=TimeWindowObs(d_Obs1)

    d_Obs2={}
    for i in xrange(nObsTime):
        t=tInt*(i+1)/nObsTime
        captorPosition=-80.+20.*t
        obsCoord=captorPosition+np.array([-10.,-5.,0.,5.,10.])
        obsValues=x_degrad.whereTime(t)[pos2Idx(g, obsCoord)]
        d_Obs2[t]=StaticObs(obsCoord,obsValues, obsOp_Coord)
    timeObs2=TimeWindowObs(d_Obs2)


    plt.figure()
    i=0
    for t in timeObs1.times:
        i+=1
        sub=plt.subplot(nObsTime, 1, i)
        sub.plot(g.x, x_truth.whereTime(t), 'k', linewidth=2.5)
        sub.plot(g.x, x_degrad.whereTime(t), 'r', linewidth=2.5)
        sub.plot(timeObs1[t].interpolate(g), timeObs1[t].values, 'g')
        sub.set_title("t=%.2f"%t)

    plt.figure()
    i=0
    for t in timeObs2.times:
        i+=1
        sub=plt.subplot(nObsTime, 1, i)
        sub.plot(g.x, x_truth.whereTime(t), 'k', linewidth=2.5)
        sub.plot(g.x, x_degrad.whereTime(t), 'r')
        sub.plot(timeObs2[t].interpolate(g), timeObs2[t].values, 'go')
        sub.set_title("t=%.2f"%t)
    plt.show()
