import numpy as np
from pyKdV import SpectralGrid, Trajectory
from dVar import pos2Idx


class obsTimeOpError(Exception):
    pass

#-----------------------------------------------------------

def whereTrajTime(u, time):
    return np.where(u.time>=time)[0].min()

#-----------------------------------------------------------

def kd_departure(x, H, argsH, dObs):
    """
        Departures
        for KdV TLM


        x           :   state variable <numpy.ndarray>
        H           :   non-linear observation operator
        argsH       :   observation operators common arguments <list>
        dObs        :   observations <dict>
                            {time <float>   :   values <np.array>, ...}

        <!> for incremental, define dObs as y'=y-H(x_b)
    """
    
    if not (isinstance(dObs, dict)): 
        raise obsTimeOpError("dObs <dict>")
    for t in dObs.iterkeys():
        if not isinstance(dObs[t], np.ndarray):
            raise obsTimeOpError("dObs[t] <numpy.ndarray>")

    dDeparture={}
    dHx=H(x,*argsH)
    
    for t in dHx.keys():
        dDeparture[t]=dObs[t]-dHx[t]

    return dDeparture

#-----------------------------------------------------------

def kd_opObs(x, dynamicModel, g, dObsPos, staticObsOp, sObsArgs):
    """
        Non-linear observation operator
        for KdV NL propagation

        x           :   state <numpy.ndarray>
        dynamicModel:   model launcher with integrate() method <Launcher>
        g           :   <SpectralGrid>
        dObsPos     :   observation coordinates <dict>
                            {time <float>   :   positions <np.array>, ...}
        staticObsOp :   static observation operator
        sObsArgs    :   static observation operator arguments <list>
    """
    if not (isinstance(g, SpectralGrid)):
        raise obsTimeOpError("g <pyKdV.SpectralGrid>")
    if not (isinstance(dObsPos, dict)): 
        raise obsTimeOpError("dObsPos <dict>")
    for t in dObsPos.iterkeys():
        if not isinstance(dObsPos[t], np.ndarray):
            raise obsTimeOpError("dObsPos[t] <numpy.ndarray>")
    
    if not (staticObsOp==None or callable(staticObsOp)):
        raise obsTimeOpError("staticObsOp <None | function>")
    if not isinstance(dynamicModel, object):
        raise obsTimeOpError("dynamicModel <Launcher object>")

    #----| Model equivalent |-----------
    HMx={}
    for t in np.sort(dObsPos.keys()):
        traj=dynamicModel.integrate(x,t)
        if staticObsOp==None:
            HMx[t]=traj.final()
        else:
            HMx[t]=staticObsOp(traj.final(), g, pos2Idx(g, dObsPos[t]),
                                *sObsArgs)
        # <TODO> continue integration from t ?
 
    return HMx


#-----------------------------------------------------------

def kd_opObs_TL(dx, dynamicTLM, g, dObsPos, staticObsOp, sObsArgs):
    """
        Tangent linear observation operator
        for KdV TLM

        dx          :   state increment <numpy.ndarray>
        dynamicTLM  :   TLM launcher with integrate() method
        g           :   <SpectralGrid>
        dObsPos     :   observation coordinates <dict>
                            {time <float>   :   positions <np.array>, ...}
        staticObsOp :   static observation operator
        sObsArgs    :   static observation operator arguments <list>
    """
    if not (isinstance(g, SpectralGrid)):
        raise obsTimeOpError("g <pyKdV.SpectralGrid>")
    if not (isinstance(dObsPos, dict)): 
        raise obsTimeOpError("dObsPos <dict>")
    for t in dObsPos.iterkeys():
        if not isinstance(dObsPos[t], np.ndarray):
            raise obsTimeOpError("dObsPos[t] <numpy.ndarray>")
    if not (staticObsOp==None or callable(staticObsOp)):
        raise obsTimeOpError("staticObsOp <None | function>")
    if not isinstance(dynamicTLM, object):
        raise obsTimeOpError("dynamicTLM <Launcher object>")

    HMdx={}
    #----| code to be transposed |--
    t_pre=0.
    dx_pre=dx
    for t in np.sort(dObsPos.keys()):
        dx_t=dynamicTLM.integrate(dx_pre, tInt=t-t_pre, t0=t_pre)
        t_pre=t
        dx_pre=dx_t
     #-------------------------------
        if staticObsOp==None:
            HMdx[t]=dx_t
        else:
            HMdx[t]=staticObsOp(dx_t, g, pos2Idx(g, dObsPos[t]), *sObsArgs)
    return HMdx
 
#-----------------------------------------------------------


def kd_opObs_TL_Adj(dObs, dynamicTLM, g, dObsPos, sObsOp_Adj, sObsArgs):
    """
        Adjoint of tangent linear observation operator
        for KdV TLM

        dObs        :   observations <dict>
                            {time <float>   :   values <np.array>, ...}
        dynamicTLM  :   TLM launcher with adjoint() method
        g           :   <SpectralGrid>
        dObsPos     :   observation coordinates <dict>
                            {time <float>   :   positions <np.array>, ...}
        sObsOp_Adj  :   static observation operator adjoint
        sObsArgs    :   static observation operator arguments <list>

    """
    if not (isinstance(g, SpectralGrid)):
        raise obsTimeOpError("g <pyKdV.SpectralGrid>")
    if not (isinstance(dObsPos, dict)): 
        raise obsTimeOpError("dObsPos <dict>")
    for t in dObsPos.iterkeys():
        if not isinstance(dObsPos[t], np.ndarray):
            raise obsTimeOpError("dObsPos[t] <numpy.ndarray>")
    if not (sObsOp_Adj==None or callable(sObsOp_Adj)):
        raise obsTimeOpError("sObsOp_Adj <None | function>")
    if not isinstance(dynamicTLM, object):
        raise obsTimeOpError("dynamicTLM <Launcher object>")

    tOrder=np.argsort(dObsPos.keys())
    nTime=len(tOrder)
    
    i=0
    M_TH_AdjObs=np.zeros(g.N)

    #----| transposed code |--------
    for t in np.sort(dObsPos.keys())[::-1]:
        i+=1
        if i<nTime:
            t_pre=dObsPos.keys()[tOrder[-1-i]]
        else:
            t_pre=0.
        if sObsOp_Adj==None:
            w=dObs[t]
        else:
            w=sObsOp_Adj(dObs[t], g, pos2Idx(g, dObsPos[t]),  *sObsArgs)


        M_TH_AdjObs=dynamicTLM.adjoint(w+M_TH_AdjObs,
                                        tInt=t-t_pre, t0=t_pre)

        w=M_TH_AdjObs
    #-------------------------------
    
    return M_TH_AdjObs
 
 

#===========================================================
if __name__=="__main__":
    import matplotlib.pyplot as plt
    import pyKdV as kdv
    from dVar import degrad
    
    
    Ntrc=100
    L=300.
    g=SpectralGrid(Ntrc, L)
        
    kdvParam=kdv.Param(g, beta=1., gamma=-1.)
    tInt=10.
    maxA=5.
    
    model=kdv.Launcher(kdvParam, maxA)


    x0_truth_base=kdv.rndFiltVec(g, Ntrc=g.Ntrc/5,  amp=1.)
    gaussWave=1.5*kdv.gauss(g.x, 40., 20. )-1.*kdv.gauss(g.x, -20., 14. )
    soliton=kdv.soliton(g.x, 0., amp=5., beta=1., gamma=-1)

    x0_truth=x0_truth_base+gaussWave
    x_truth=model.integrate(x0_truth, tInt)

    x0_bkg=x0_truth_base
    x_bkg=model.integrate(x0_bkg, tInt)

    #----| Observations |---------
    dObsPos={}
    nObsTime=4
    for i in xrange(nObsTime):
        dObsPos[tInt/(i+1)]=x_truth[x_truth.whereTime(tInt/(i+1))]
    
    H=kd_opObs
    H_TL=kd_opObs_TL
    H_TL_Adj=kd_opObs_TL_Adj
    staticObsOp=None
    sObsOpArgs=()
    argsHcom=(g, dObsPos, staticObsOp, sObsOpArgs)
    
    sigR=.5
    dObs=H(x0_truth, model, *argsHcom) 
                         
    
    dR_inv={}
    for t in dObsPos.keys():
        dR_inv[t]=sigR**(-1)*np.eye(len(dObsPos[t]))
    
    #----| Validating H_TL_Adj |----
    x_rnd=kdv.rndFiltVec(g, amp=0.5)
    tlm=kdv.TLMLauncher(x_truth, kdvParam)
    dY=dObs
    Hx=H_TL(x_rnd, tlm, *argsHcom)
    H_Adjy=H_TL_Adj(dY, tlm, *argsHcom)
    prod1=0.
    for t in Hx.keys():
        prod1+=np.dot(dY[t], Hx[t])
    prod2=np.dot(H_Adjy, x_rnd)
    print(prod1, prod2, np.abs(prod1-prod2))
        
    
    #----| Departures |-----------
    argsDep=(model,)+ argsHcom
    dDepartures=kd_departure(x0_bkg, H, argsDep, dObs)

    
    #----| Post-processing |------
    nTime=len(dObs.keys())
    plt.figure(figsize=(10.,3.*nTime))
    i=0
    for t in np.sort(dObs.keys()):
        i+=1
        sub=plt.subplot(2*nTime, 1, 2*i-1)
        ti=whereTrajTime(x_truth, t)
        sub.plot(g.x, dObs[t], 'g')
        sub.plot(g.x, x_bkg[ti], 'b')
        sub.set_title("$t=%f$"%t)
        if i==nTime:
            sub.legend(["$x_{t}$", "$y$", "$x_b$"], loc="lower left")

        subDep=plt.subplot(2*nTime, 1, 2*i)
        subDep.plot(g.x, dDepartures[t], 'r')
        subDep.axhline(y=0, color='k')

    plt.show()    
