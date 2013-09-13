import numpy as np
import pyKdV as kdv
from dVar import pos2Idx, B_sqrt_op, opObs_Idx, opObs_Idx_Adj

import matplotlib.pyplot as plt ### to be removed

class obsTimeOpError(Exception):
    pass

#-----------------------------------------------------------

def whereTrajTime(u, time):
    return np.where(u.time>=time)[0].min()

#-----------------------------------------------------------

def kd_departure(xi, traj_bkg, var, B_sqrt_op, H, H_TL, argsH, dObs,
                    rCTilde_sqrt):
    """
        Departures
        for KdV TLM

        Isotropic and homogeneous correlations

        xi          :   preconditioned state variable <numpy.ndarray>
        traj_bkg    :   background trajectory <pyKdV.Trajectory>
        var         :   model variances <numpy.ndarray>
        B_sqrt_op   :   B^{1/2} operator
        H           :   non-linear observation operator
        H_TL        :   tangent linear observation operator
        argsH       :   observation operators common arguments <list>
        dObs        :   observations <dict>
                            {time <float>   :   values <np.array>, ...}
        rCTilde_sqrt:   CTilde^{1/2} diagonal <numpy.ndarray>
    """
    
    if not (isinstance(dObs, dict)): 
        raise obsTimeOpError("dObs <dict>")
    for t in dObs.iterkeys():
        if not isinstance(dObs[t], np.ndarray):
            raise obsTimeOpError("dObs[t] <numpy.ndarray>")

    x=B_sqrt_op(xi, var, rCTilde_sqrt)+traj_bkg[0]
    dHtraj_bkg=H(traj_bkg[0], *argsH)
    dH_AdjLx=H_TL(x-traj_bkg[0], traj_bkg, *argsH)

    dDeparture={}
    for t in dHtraj_bkg.keys():
        dDeparture[t]=-dObs[t]+dHtraj_bkg[t]+dH_AdjLx[t]

    return dDeparture

#-----------------------------------------------------------

def kd_opObs(x, g, dObsPos, kdvParam, maxA):
    """
        Non-linear observation operator
        for KdV NL propagation

        x           :   state <numpy.ndarray>
        g           :   <SpectralGrid>
        dObsPos     :   observation coordinates <dict>
                            {time <float>   :   positions <np.array>, ...}
        kdvParam    :   <pyKdV.Param>
        maxA        :   maximum expected amplitude <float>
    """
    if not (isinstance(g, kdv.SpectralGrid)):
        raise obsTimeOpError("g <pyKdV.SpectralGrid>")
    if not (isinstance(dObsPos, dict)): 
        raise obsTimeOpError("dObsPos <dict>")
    for t in dObsPos.iterkeys():
        if not isinstance(dObsPos[t], np.ndarray):
            raise obsTimeOpError("dObsPos[t] <numpy.ndarray>")
    if not (isinstance(kdvParam, kdv.Param)):
        raise obsTimeOpError("kdvParam <pyKdV.Param>")

    #----| Model equivalent |-----------
    HMx={}
    for t in np.sort(dObsPos.keys()):
        # parallelize this?
        traj=kdv.Launcher(kdvParam, t, maxA).integrate(x)
        HMx[t]=opObs_Idx(traj.final(), g, pos2Idx(g, dObsPos[t]))

    return HMx


#-----------------------------------------------------------

def kd_opObs_TL(dx, traj_bkg, g, dObsPos, kdvParam, maxA):
    """
        Tangent linear observation operator
        for KdV TLM

        dx          :   state increment <numpy.ndarray>
        traj_bkg    :   background trajectory <pyKdV.Trajectory>
        g           :   <SpectralGrid>
        dObsPos     :   observation coordinates <dict>
                            {time <float>   :   positions <np.array>, ...}
        kdvParam    :   <pyKdV.Param>
    """
    if not (isinstance(traj_bkg, kdv.Trajectory)):
        raise obsTimeOpError("traj_bkg <pyKdV.Trajectory>")
    if not (isinstance(g, kdv.SpectralGrid)):
        raise obsTimeOpError("g <pyKdV.SpectralGrid>")
    if not (isinstance(dObsPos, dict)): 
        raise obsTimeOpError("dObsPos <dict>")
    for t in dObsPos.iterkeys():
        if not isinstance(dObsPos[t], np.ndarray):
            raise obsTimeOpError("dObsPos[t] <numpy.ndarray>")
    if not (isinstance(kdvParam, kdv.Param)):
        raise obsTimeOpError("kdvParam <pyKdV.Param>")

    HMdx={}
    #----| code to be transposed |--
    t_pre=0.
    dx_pre=dx
    for t in np.sort(dObsPos.keys()):
        dx_t=kdv.TLMLauncher(kdvParam, traj_bkg,
                             tInt=t-t_pre, t0=t_pre).integrate(dx_pre)
        t_pre=t
        dx_pre=dx_t
     #-------------------------------
        HMdx[t]=opObs_Idx(dx_t, g, pos2Idx(g, dObsPos[t]))
    return HMdx
 
#-----------------------------------------------------------


def kd_opObs_TL_Adj(dObs, traj_bkg, g, dObsPos, kdvParam, maxA):
    """
        Adjoint of tangent linear observation operator
        for KdV TLM

        dObs        :   observations <dict>
                            {time <float>   :   values <np.array>, ...}
        traj_bkg    :   background trajectory <pyKdV.Trajectory>
        g           :   <SpectralGrid>
        dObsPos     :   observation coordinates <dict>
                            {time <float>   :   positions <np.array>, ...}
        kdvParam    :   <pyKdV.Param>

        <!> does not validate adjoint test
    """
    if not (isinstance(traj_bkg, kdv.Trajectory)):
        raise obsTimeOpError("traj_bkg <pyKdV.Trajectory>")
    if not (isinstance(g, kdv.SpectralGrid)):
        raise obsTimeOpError("g <pyKdV.SpectralGrid>")
    if not (isinstance(dObsPos, dict)): 
        raise obsTimeOpError("dObsPos <dict>")
    for t in dObsPos.iterkeys():
        if not isinstance(dObsPos[t], np.ndarray):
            raise obsTimeOpError("dObsPos[t] <numpy.ndarray>")
    if not (isinstance(kdvParam, kdv.Param)):
        raise obsTimeOpError("kdvParam <pyKdV.Param>")

    tOrder=np.argsort(dObsPos.keys())
    nTime=len(tOrder)
    
    i=0
    M_TH_AdjObs=np.zeros(traj_bkg.grid.N)

    #----| transposed code |--------
    for t in np.sort(dObsPos.keys())[::-1]:
        i+=1
        if i<nTime:
            t_pre=dObsPos.keys()[tOrder[-1-i]]
        else:
            t_pre=0.
        w=opObs_Idx_Adj(dObs[t], g, pos2Idx(g, dObsPos[t]))


        M_TH_AdjObs=kdv.TLMLauncher(kdvParam, traj_bkg, 
                        tInt=t-t_pre, t0=t_pre).adjoint(w+M_TH_AdjObs)


        w=M_TH_AdjObs
    #-------------------------------
    
    return M_TH_AdjObs
 
 

#===========================================================
if __name__=="__main__":
    import matplotlib.pyplot as plt
    from dVar import pos2Idx, fCorr_isoHomo, degrad, B_sqrt_op, \
                        rCTilde_sqrt_isoHomo, opObs_Idx, opObs_Idx_Adj
    import pyKdV as kdv
    
    
    Ntrc=100
    L=300.
    g=kdv.SpectralGrid(Ntrc, L)
        
    kdvParam=kdv.Param(g, beta=1., gamma=-1.)
    tInt=10.
    maxA=5.
    
    model=kdv.Launcher(kdvParam,tInt, maxA)

    x0_truth_base=kdv.rndFiltVec(g, Ntrc=g.Ntrc/5,  amp=1.)
    wave=kdv.soliton(g.x, 0., amp=5., beta=1., gamma=-1)\
                +1.5*kdv.gauss(g.x, 40., 20. )-1.*kdv.gauss(g.x, -20., 14. )
    x0_truth=x0_truth_base+wave
    x_truth=model.integrate(x0_truth)

    x0_bkg=x0_truth_base
    x_bkg=model.integrate(x0_bkg)
    
    #----| Observations |---------
    dObsPos={}
    dObsPos[tInt/4.]=np.array([-30.,  70.])
    dObsPos[tInt/3.]=np.array([-120., -34., -20., 2.,  80., 144.])
    dObsPos[tInt/2.]=np.array([-90., -85, 4., 10.])
    dObsPos[tInt]=np.array([-50., 0., 50.])
    
    H=kd_opObs
    H_TL=kd_opObs_TL
    H_TL_Adj=kd_opObs_TL_Adj
    argsHcom=(g, dObsPos, kdvParam, maxA)
    
    sigR=.5
    x0_degrad=degrad(x0_truth, 0., sigR)                   
    dObs_degrad=H(x0_degrad, *argsHcom) 
    dObs_truth=H(x0_truth,  *argsHcom) 
                         
    
    dR_inv={}
    for t in dObsPos.keys():
        dR_inv[t]=sigR**(-1)*np.eye(len(dObsPos[t]))
    
    #----| Validating H_TL_Adj |----
    x_rnd=kdv.rndFiltVec(g, amp=0.5)
    dY=dObs_degrad
    Hx=H_TL(x_rnd, x_bkg, *argsHcom)
    H_Adjy=H_TL_Adj(dY, x_bkg, *argsHcom)
    prod1=0.
    for t in Hx.keys():
        prod1+=np.dot(dY[t], Hx[t])
    prod2=np.dot(H_Adjy, x_rnd)
    print(prod1, prod2, np.abs(prod1-prod2))
        
    
    #----| Preconditionning |-----
    Lc=10.
    sig=0.4
    corr=fCorr_isoHomo(g, Lc)
    rCTilde_sqrt=rCTilde_sqrt_isoHomo(g, corr)
    var=sig*np.ones(g.N)
    xi=np.zeros(g.N)
    
    #----| Departures |-----------
    dDepartures=kd_departure(xi, x_bkg, var, B_sqrt_op, H, H_TL, argsHcom,
                                dObs_degrad, rCTilde_sqrt)
    for t in np.sort(dObs_degrad.keys()):
        print("t=%f"%t)
        print(dDepartures[t])
    
    
    #----| Post-processing |------
    nTime=len(dObs_degrad.keys())
    plt.figure(figsize=(10.,3.*nTime))
    i=0
    for t in np.sort(dObs_degrad.keys()):
        i+=1
        sub=plt.subplot(nTime, 1, i)
        ti=whereTrajTime(x_truth, t)
        sub.plot(g.x, x_truth[ti], 'g')
        sub.plot(g.x[pos2Idx(g, dObsPos[t])], dObs_truth[t], 'go')
        sub.plot(g.x[pos2Idx(g, dObsPos[t])], dObs_degrad[t], 'ro')
        sub.plot(g.x, x_bkg[ti], 'b')
        sub.plot(g.x[pos2Idx(g, dObsPos[t])], 
                    x_bkg[ti][pos2Idx(g, dObsPos[t])], 'bo')
        sub.set_title("$t=%f$"%t)
        if i==nTime:
            sub.legend(["$x_{t}$", "$H(x_{t})$", "$y$", "$x_b$", 
                        "$H(x_b)$"], loc="lower left")
    plt.show()    
