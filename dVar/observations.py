import numpy as np
from pseudoSpec1D import Grid, Launcher, Trajectory
import random as rnd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
import pickle

#-----------------------------------------------------------
#----| Utilitaries |----------------------------------------
#-----------------------------------------------------------

def degrad(signal,mu,sigma,seed=0.7349156729):
    ''' 
    Gaussian noise signal degradation

    degrad(u,mu,sigma,seed=...)

    signal  :  input signal
    mu      :  noise mean (gaussian mean)
    sigma   :  noise variance
    '''
    rnd.seed(seed)
    sig_degrad=signal.copy()
    for i in xrange(signal.size):
        sig_degrad[i]=signal[i]+rnd.gauss(mu, sigma)
    return sig_degrad

def degradTraj(traj, mu, sigma, seed=None):
    '''
    <!> replaced by Trajectory.degrad()
    '''
    return traj.degrad(mu, sigma, seed=seed)


#-----------------------------------------------------------
#----| Sampling |-------------------------------------------
#-----------------------------------------------------------

def homoSampling(grid, nObs):
    if not isinstance(grid, Grid):
        raise TypeError("grid <pseudoSpec>")

    coord=[]
    ObsDx=(grid.max()-grid.min())/nObs
    for j in xrange(nObs):
        coord.append(grid.min()+j*ObsDx)
    return coord

def rndSampling(grid, nObs, precision=2, seed=None):
    if not isinstance(grid, Grid):
        raise TypeError("grid <pseudoSpec>")

    rnd.seed(seed)
    coord=[]
    i=0
    while (i < nObs):
        
        pick=round(rnd.random()*grid.L, precision)
        if grid.centered:
            pick-=grid.L/2
        if ((not pick in coord) and (pick <=grid.max()) 
                and (pick >=grid.min()) ):
            coord.append(pick)
            i+=1
    coord.sort()
    return coord 

def removeDuplicates(coord):
    coord=list(set(coord))
    coord.sort()
    return coord 



#-----------------------------------------------------------
#----| Observation operators |------------------------------
#-----------------------------------------------------------

def obsOp_Coord(x, g, obsCoord):
    """
    Trivial static observation operator
    """
    idxObs=g.pos2Idx(obsCoord)
    nObs=len(idxObs)
    H=np.zeros(shape=(nObs,g.N))
    for i in xrange(nObs):
        H[i, idxObs[i]]=1.
    return np.dot(H,x)

def obsOp_Coord_Adj(obs, g, obsCoord):
    """
    Trivial static observation operator adjoint
    """
    idxObs=g.pos2Idx(obsCoord)
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
                            <pseudoSpec1D.Grid | numpy.ndarray>
                            (Grid for continuous observations)
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

        if isinstance(coord, Grid):
            self.grid=coord
            self.coord=coord.x
            self.nObs=coord.N
        elif isinstance(coord, np.ndarray):
            if coord.ndim <> 1:
                raise self.StaticObsError("coord.ndim==1")
            self.coord=coord
            self.nObs=len(coord)
        elif isinstance(coord, list): 
            self.coord=np.array(coord)
            self.nObs=len(coord)
        else:
            raise self.StaticObsError(
                "coord <pseudoSpec1D.Grid | numpy.ndarray>")

        if isinstance(values, list):
            values=np.array(values)
            if len(values)<>self.nObs:
                raise ValueError()
        elif isinstance(values, np.ndarray):
            if values.ndim<>1 or len(values)<>self.nObs:
                raise ValueError()
        else:
            raise TypeError()
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
        elif isinstance(metric, (float, int)):
            self.metric=metric*np.eye(self.nObs)
        elif isinstance(metric, np.ndarray):
            if metric.ndim==1:
                self.metric=np.diag(metric)
            elif metric.ndim==2:
                # this is why coord must not be sorted!
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
        if not isinstance(g, Grid):
            raise self.StaticObsError("g <Grid>")
        if not isinstance(x, np.ndarray):
            raise self.StaticObsError("x <numpy.ndarray>")
        if not (x.ndim==1 or len(x)==g.N):
            raise self.StaticObsError("x.shape=(g.N)")
        if self.obsOp<>None:
            return self.obsOp(x, g, self.coord, *self.obsOpArgs)
        else:
            return x

    #------------------------------------------------------
    
    def innovation(self, x, g):
        if not isinstance(g, Grid):
            raise self.StaticObsError("g <Grid>")
        if not isinstance(x, np.ndarray):
            raise self.StaticObsError("x <numpy.ndarray>")
        if not (x.ndim==1 or len(x)==g.N):
            raise self.StaticObsError("x.shape=(g.N)")
        return self.values-self.modelEquivalent(x, g)

    #------------------------------------------------------

    def interpolate(self, g):
        if not isinstance(g, Grid):
            raise self.StaticObsError("g <Grid>")
        return g.x[self.__pos2Idx(g)]


    #------------------------------------------------------
    
    def prosca(self, y1, y2):
        if len(y1)<>self.nObs or len(y2)<>self.nObs:
            raise self.StaticObsError()
        return np.dot(y1, np.dot(self.metric, y2))

    #------------------------------------------------------

    def norm(self, y):
        return np.sqrt(self.prosca(y,y))

    #------------------------------------------------------

    def correlation(self, y):
        return self.prosca(self.values, y)/(
                self.norm(self.values)*self.norm(y))
    
    def corrModelEq(self, x, grid):
        y=self.modelEquivalent(x, grid)
        return self.correlation(y)
    
    def corrModelEqBkg(self, v, x_bkg, grid):
        inno=self.values-self.modelEquivalent(x_bkg, grid)
        Hv=self.modelEquivalent(v,grid)
        return self.prosca(Hv,inno)/(self.norm(inno)*self.norm(Hv))


    #------------------------------------------------------
    #----| I/O method |------------------------------------
    #------------------------------------------------------

    def dump(self, fun):
        pickle.dump(self.coord, fun)
        pickle.dump(self.metric, fun)
        pickle.dump(self.obsOp, fun)
        pickle.dump(self.obsOpArgs, fun)
        pickle.dump(self.obsOpTLMAdj, fun)
        pickle.dump(self.values, fun)

    #-------------------------------------------------------
    #----| Plotting methods |-------------------------------
    #-------------------------------------------------------
    
    def plot(self, values, g,  axe=None, 
                linestyle='', marker='o', **kwargs):

        if not isinstance(g, Grid):
            raise self.StaticObsError("g <Grid>")
        axe=self.__checkAxe(axe)
        axe.plot(self.interpolate(g), values, marker=marker,
                    linestyle=linestyle, **kwargs)
        return axe

    #-------------------------------------------------------

    def plotModelEquivalent(self, field, g, axe=None, 
                            linestyle='', marker='o', **kwargs):
        if not isinstance(g, Grid):
            raise self.StaticObsError("g <Grid>")
        axe=self.__checkAxe(axe)
        axe=self.plot(self.modelEquivalent(field, g), g, axe=axe, 
                            linestyle='', marker='o', **kwargs)
        return axe

    #-------------------------------------------------------

    def plotObs(self, g, continuousField=None, axe=None, 
                marker='o',  correlation=False, xlim=None,   
                continuousFieldStyle='k-', 
                continuousFieldLabel=None,
                **kwargs):

        if not isinstance(g, Grid):
            raise self.StaticObsError("g <Grid>")
        axe=self.__checkAxe(axe)
        axe=self.plot(self.values, g, axe=axe,
                        marker=marker, linestyle='', **kwargs)
        if isinstance(continuousField, np.ndarray):
            if (continuousField.ndim==1 and 
                    len(continuousField)==g.N):
                axe.plot(g.x, continuousField, continuousFieldStyle, 
                        label=continuousFieldLabel)
                if correlation==True:
                    axe.text(0.05,0.95,r'$\rho_c=%.2f$'%self.correlation(
                        self.modelEquivalent(continuousField, g)),
                        transform=axe.transAxes)
            else:
                raise self.StaticObsError(
                        "incompatible continuous field dimensions")
        if xlim<>None:
            axe.set_xlim(xlim)
        return axe

    #-------------------------------------------------------

    def plotInno(self, g, x, continuousField=None, axe=None, 
                marker='o', xlim=None,   
                continuousFieldStyle='k-', 
                continuousFieldLabel=None,
                **kwargs):

        if not isinstance(g, Grid):
            raise self.StaticObsError("g <Grid>")
        axe=self.__checkAxe(axe)
        axe=self.plot(self.innovation(x, g), g, axe=axe,
                        marker=marker, linestyle='', **kwargs)
        if isinstance(continuousField, np.ndarray):
            if (continuousField.ndim==1 and 
                    len(continuousField)==g.N):
                axe.plot(g.x, continuousField, continuousFieldStyle, 
                        label=continuousFieldLabel)
            else:
                raise self.StaticObsError(
                        "incompatible continuous field dimensions")
        if xlim<>None:
            axe.set_xlim(xlim)
        return axe



    #-------------------------------------------------------

    def __checkAxe(self, axe):
        if axe==None:
            axe=plt.subplot(111)
        elif not (isinstance(axe,(Axes, GridSpec))):
            raise self.StaticObsError(
                "axe < matplotlib.axes.Axes | matplotlib.gridspec.GridSpec >")
        return axe
    #------------------------------------------------------
    #----| Classical overloads |----------------------------
    #-------------------------------------------------------

    def __str__(self):
        output="____| StaticObs |___________________________"
        output+="\n   nObs=%d"%self.nObs
        output+="\n   coord:\n     %s\n"%self.coord.__str__()
        output+="\n   observation operator:\n     %s"%self.obsOp
        output+="\n   observation tangeant operator adjoint:\n     %s"%self.obsOpTLMAdj
        output+="\n____________________________________________"
        return output

    #------------------------------------------------------

    def _extendMetric(self, statObs):
        '''
        Build the observation operator metric
        assuming observation sets independant
        '''
        metric=np.zeros(shape=(self.nObs+statObs.nObs,
                              self.nObs+statObs.nObs))
        metric[:self.nObs, :self.nObs]=self.metric
        metric[-statObs.nObs:, -statObs.nObs:]=statObs.metric
        return metric

    def __add__(self, statObs, obsOpEq=True):
        if not isinstance(statObs, StaticObs): raise TypeError()
        
        if obsOpEq:
            if (self.obsOp<>statObs.obsOp or
                self.obsOpTLMAdj<>statObs.obsOpTLMAdj or
                self.obsOpArgs<>statObs.obsOpArgs):
                raise ValueError()

        metric=self._extendMetric(statObs)

        newCoord=self.coord.tolist()
        newCoord.extend(statObs.coord.tolist())

        newValues=self.values.tolist()
        newValues.extend(statObs.values.tolist())

        return StaticObs(newCoord, newValues, 
                         obsOp=self.obsOp, obsOpTLMAdj=self.obsOpTLMAdj,
                         obsOpArgs=self.obsOpArgs, metric=metric)

        
        
        
        
#=====================================================================

def loadStaticObs(fun):

    coord=pickle.load(fun)
    metric=pickle.load(fun)
    obsOp=pickle.load(fun)
    obsOpArgs=pickle.load(fun)
    obsOpTLMAdj=pickle.load(fun)
    values=pickle.load(fun)

    sObs=StaticObs(coord, values, obsOp=obsOp, obsOpTLMAdj=obsOpTLMAdj,
                    obsOpArgs=obsOpArgs, metric=metric)
    return sObs

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

        self.times=d_Obs.keys()
        self.times.sort()

        self.nTimes=len(self.times)
        self.tMax=np.max(self.times)
        self.tMin=np.max(self.times)
        self.d_Obs=d_Obs
        self.nObs=0
        for t in self.times:
            self.nObs+=self.d_Obs[t].nObs
        self.obsOp=d_Obs[self.times[0]].obsOp
        self.obsOpArgs=d_Obs[self.times[0]].obsOpArgs

                
    #------------------------------------------------------
    #----| Private methods |-------------------------------
    #------------------------------------------------------

    def __propagatorValidate(self, propagator):
        if not isinstance(propagator, Launcher):
            raise self.TimeWindowObsError("propagator <Launcher>")

    #------------------------------------------------------

    def __integrate(self, x, propagator, t0=0.):
        self.__propagatorValidate(propagator)
        d_xt={}
        x0=x
        for t in self.times:
            if t==t0:
                d_xt[t]=x0
            else:
                d_xt[t]=(propagator.integrate(x0,t-t0)).final    
            x0=d_xt[t]
            t0=t
        
        return d_xt
    #------------------------------------------------------
    #----| Public methods |--------------------------------
    #------------------------------------------------------

    def modelEquivalent(self, x, propagator, t0=0.):
        self.__propagatorValidate(propagator)
        g=propagator.grid
        d_Hx={}
        d_xt=self.__integrate(x, propagator, t0=t0)
        for t in self.times:
            d_Hx[t]=self.d_Obs[t].modelEquivalent(d_xt[t], g)
        return d_Hx

    #------------------------------------------------------
    
    def innovation(self, x, propagator, t0=0.):
        self.__propagatorValidate(propagator)
        d_inno={}
        d_Hx=self.modelEquivalent(x, propagator, t0=t0)
        for t in self.times:
            d_inno[t]=self.d_Obs[t].values-d_Hx[t]
        return d_inno
        
    #------------------------------------------------------

    def dump(self, fun):
        pickle.dump(self.nTimes, fun)
        pickle.dump(self.times, fun)
        for i in xrange(self.nTimes):      
            self.d_Obs[self.times[i]].dump(fun)        
    
    #------------------------------------------------------
    def cut(self, tMin=None, tMax=None):
        if tMin==None: tMin=self.tMin
        if tMax==None: tMax=self.tMax

        cut_d_Obs={}
        for t in self.d_Obs.keys():
            if t >= tMin and t<= tMax:
                cut_d_Obs[t]=self[t]
                
        return TimeWindowObs(cut_d_Obs)
    #-------------------------------------------------------
    #----| Plotting methods |-------------------------------
    #-------------------------------------------------------
    
    def plotObs(self, g, nbGraphLine=3, trajectory=None, correlation=False, 
                    trajectoryStyle='k', xlim=None, 
                    trajectoryLabel=None,
                    **kwargs):


        if not (isinstance(trajectory, Trajectory) or trajectory==None): 
            raise self.TimeWindowObsError("trajectory <None | Trajectory>")
        if self.nTimes < nbGraphLine:
            nSubRow=self.nTimes
        else:
            nSubRow=nbGraphLine
        nSubLine=self.nTimes/nSubRow
        if self.nTimes%nSubRow: nSubLine+=1
        i=0
        axes=[]
        for t in self.times:
            axes.append(plt.subplot(nSubLine, nSubRow, i+1))
            if trajectory==None:
                self[t].plotObs(g, axe=axes[i], xlim=xlim, **kwargs)
            else:
                self[t].plotObs(g, axe=axes[i], xlim=xlim,
                                continuousField=trajectory.whereTime(t),
                                continuousFieldStyle=trajectoryStyle, 
                                continuousFieldLabel=trajectoryLabel,
                                correlation=correlation, **kwargs)
            axes[i].set_title("$t=%f$"%t)
            i+=1

        return axes
    #------------------------------------------------------
    #----| Classical overloads |---------------------------
    #------------------------------------------------------

    def __getitem__(self, t):
        return self.d_Obs[t]

    #-------------------------------------------------------

    def __str__(self):
        output="====| TimeWindowObs |==========================="
        output+="\n nTimes=%d"%self.nTimes
        output+="\n %s"%self.times.__str__()
        output+="\n\n observation operator:\n  %s"%self.obsOp
        for t in self.times:
            output+="\n\n ["+str(t)+"]\n"
            output+=self.d_Obs[t].__str__()
        output+="\n================================================"
        return output

    #------------------------------------------------------

    def __add__(self, twObs):
        if not isinstance(twObs, TimeWindowObs):
            raise TypeError()

        d_Obs1=self.d_Obs.copy()
        d_Obs2=twObs.d_Obs.copy()
        new_d_Obs={}
        # check for cooccurences
        for t in d_Obs1.keys():
            if t in d_Obs2.keys():
                new_d_Obs[t]=d_Obs1.pop(t)+d_Obs2.pop(t)
        # merge the rest
        new_d_Obs.update(d_Obs1)
        new_d_Obs.update(d_Obs2)
                
        return TimeWindowObs(new_d_Obs)
#=====================================================================

def loadTWObs(fun):

    nTimes=pickle.load(fun)
    times=pickle.load(fun)
    d_Obs={}
    for i in xrange(nTimes):
        d_Obs[times[i]]=loadStaticObs(fun)

    TWObs=TimeWindowObs(d_Obs) 
    return TWObs



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
        

    x0_truth_base=kdv.rndSpecVec(g, Ntrc=10,  amp=1.)
    gaussWave=1.5*kdv.gauss(g.x, 40., 20. )-1.*kdv.gauss(g.x, -20., 14. )
    soliton=kdv.soliton(g.x, 0., amp=5., beta=1., gamma=-1)

    x0_truth=x0_truth_base+gaussWave
    x0_degrad=degrad(x0_truth, 0., 0.3)

    obs1=StaticObs(g, x0_degrad, None)

    obs2Coord=np.array([-50., 0., 70.])
    obs2=StaticObs(obs2Coord, x0_degrad[g.pos2Idx(obs2Coord)],
                    obsOp_Coord, obsOp_Coord_Adj)


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
    
    model=kdv.kdvLauncher(kdvParam, maxA=5.)
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
        obsValues=x_degrad.whereTime(t)[g.pos2Idx(obsCoord)]
        d_Obs2[t]=StaticObs(obsCoord,obsValues,
                            obsOp_Coord, obsOp_Coord_Adj)
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
