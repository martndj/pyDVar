import numpy as np
import pseudoSpec1D as ps1d
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
    Trivial static observation operator
    Adjoint
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

    StaticObs(coord, values, obsOp, obsOpArgs=())
        coord       :   observation positions
                            <pseudoSpec1D.SpectralGrid | numpy.ndarray>
                            (SpectralGrid for continuous observations)
        values      :   observation values <numpy.ndarray>
        obsOp       :   static observation operator <function | None>
                            (None when observation space = model space)
        obsOpArgs   :   obsOp additional arguments
                            obsOp(x_state, x_grid, x_obsSpaceCoord, 
                                    *obsOpArgs)
    """
    class StaticObsError(Exception):
        pass


    #------------------------------------------------------
    #----| Init |------------------------------------------
    #------------------------------------------------------

    def __init__(self, coord, values, obsOp, obsOpArgs=()):

        if isinstance(coord, ps1d.SpectralGrid):
            self.coord=coord.x
            self.nObs=coord.N
        elif isinstance(coord, np.ndarray):
            if coord.ndim <> 1:
                raise self.StaticObsError("coord.ndim==1")
            self.coord=coord
            self.nObs=len(coord)
        else:
            raise self.StaticObsError(
                "coord <pseudoSpec1D.SpectralGrid | numpy.ndarray>")

        if not isinstance(values, np.ndarray):
            raise self.StaticObsError("coord <numpy.ndarray>")
        if values.ndim<>1 or len(values)<>self.nObs:
            raise self.StaticObsError("len(values)==self.nObs")
        self.values=values

        if not (callable(obsOp) or obsOp==None):
            raise self.StaticObsError("obsOp <function | None>")
        if not isinstance(obsOpArgs, tuple):
            raise self.StaticObsError("obsOpArgs <tuple>")
        self.obsOp=obsOp
        self.obsOpArgs=obsOpArgs

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
        return self.yValue-self.modelEquivalent(x, g)

    #------------------------------------------------------

    def modelSpace(self, g):
        return g.x[self.__pos2Idx(g)]



#=====================================================================
#---------------------------------------------------------------------
#=====================================================================


class timeWindowObs(object):

    class timeWindowObsError(Exception):
        pass


    #------------------------------------------------------
    #----| Init |------------------------------------------
    #------------------------------------------------------

    def __init__(self, d_Obs, propagator):
        
        if not isinstance(d_Obs, dict):
            raise timeWindowObsError("d_Obs <dict {time:<StaticObs>}>")
        for t in d_Obs.keys():
            if not (isinstance(t, (float,int)) 
                    and isinstance(d_Obs[t], StaticObs)):
                raise timeWindowObsError(
                        "d_Obs <dict {time <float>: <StaticObs>}>")
        self.times=np.sort(d_Obs.keys())
        self.tMax=self.times.max()
        self.d_Obs=d_Obs

        if not isinstance(propagator, object):
            raise timeWindowObsError("propagator <Launcher object>")
        self.propagator=propagator
                
    #------------------------------------------------------
    #----| Private methods |-------------------------------
    #------------------------------------------------------

    def __integrate(self, x):
        d_xt={}
        t0=0.
        x0=x
        for t in self.times:
            d_xt[t]=(self.propagator.integrate(x0,t-t0)).final()    
            x0=d_xt[t]
            t0=t
        
        return d_xt
    #------------------------------------------------------
    #----| Public methods |--------------------------------
    #------------------------------------------------------

    def modelEquivalent(self, x, g):
        
        d_Hx={}
        d_xt=self.__integrate(x)
        for t in self.times:
            d_Hx[t]=self.d_Obs[t].modelEquivalent(d_xt[t], g)
        return d_Hx

    #------------------------------------------------------
    
    def innovation(self, x, g):

        d_inno={}
        d_Hx=self.modelEquivalent(x, g)
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
    from pseudoSpec1D import SpectralGrid
    
    #----| Static obs |---------------------------    
    Ntrc=100
    L=300.
    g=SpectralGrid(Ntrc, L)
        

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
    plt.plot(obs1.modelSpace(g), obs1.values, 'g')
    plt.plot(obs1.modelSpace(g), obs1.modelEquivalent(x0_truth, g), 'b')
    plt.subplot(212)
    plt.plot(g.x, x0_degrad, 'r')
    plt.plot(g.x, x0_truth, 'k', linewidth=3)
    plt.plot(obs2.modelSpace(g), obs2.values, 'go')
    plt.plot(obs2.modelSpace(g), obs2.modelEquivalent(x0_truth, g), 'bo')

    #----| time window obs |----------------------
    kdvParam=kdv.Param(g, beta=1., gamma=-1.)
    tInt=10.
    maxA=5.
    
    model=kdv.Launcher(kdvParam, maxA)
    x_truth=model.integrate(x0_truth, tInt)
    x_degrad=model.integrate(x0_degrad, tInt)

    nObsTime=3
    d_Obs1={}
    for i in xrange(nObsTime):
        d_Obs1[tInt*(i+1)/nObsTime]=StaticObs(g,
            x_degrad.whereTime(tInt*(i+1)/nObsTime), None)
    timeObs1=timeWindowObs(d_Obs1, model)

    d_Obs2={}
    for i in xrange(nObsTime):
        t=tInt*(i+1)/nObsTime
        captorPosition=-80.+20.*t
        obsCoord=captorPosition+np.array([-10.,-5.,0.,5.,10.])
        obsValues=x_degrad.whereTime(t)[pos2Idx(g, obsCoord)]
        d_Obs2[t]=StaticObs(obsCoord,obsValues, obsOp_Coord)
    timeObs2=timeWindowObs(d_Obs2, model)


    plt.figure()
    i=0
    for t in timeObs1.times:
        i+=1
        sub=plt.subplot(nObsTime, 1, i)
        sub.plot(g.x, x_truth.whereTime(t), 'k', linewidth=2.5)
        sub.plot(g.x, x_degrad.whereTime(t), 'r', linewidth=2.5)
        sub.plot(timeObs1[t].modelSpace(g), timeObs1[t].values, 'g')
        sub.set_title("t=%.2f"%t)

    plt.figure()
    i=0
    for t in timeObs2.times:
        i+=1
        sub=plt.subplot(nObsTime, 1, i)
        sub.plot(g.x, x_truth.whereTime(t), 'k', linewidth=2.5)
        sub.plot(g.x, x_degrad.whereTime(t), 'r')
        sub.plot(timeObs2[t].modelSpace(g), timeObs2[t].values, 'go')
        sub.set_title("t=%.2f"%t)
    plt.show()
