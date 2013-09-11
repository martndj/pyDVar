import numpy as np
import pyKdV as kdv
from dVar import pos2Idx, B_sqrt_op

class obsTimeOpError(Exception):
    pass

#-----------------------------------------------------------

def whereTrajTime(u, time):
    return np.where(u.time>=time)[0].min()

#-----------------------------------------------------------

def kd_departure(xi, x_b, var, B_sqrt_op, H, H_TL, argsH, dObs,
                    rCTilde_sqrt):
    """

        H       :   non-linear observation operator
        H_TL    :   tangent linear observation operator
    """
    
    if not (isinstance(dObs, dict)): 
        raise obsTimeOpError("dObs <dict>")
    for t in dObs.iterkeys():
        if not isinstance(dObs[t], np.ndarray):
            raise obsTimeOpError("dObs[t] <numpy.ndarray>")

    x=B_sqrt_op(xi, var, rCTilde_sqrt)+x_b
    dHx_b=H(x_b, *argsH)
    dH_TLx=H_TL(x-x_b, x_b, *argsH)

    dDeparture={}
    for t in dHx_b.keys():
        dDeparture[t]=dObs[t]-dHx_b[t]-dH_TLx[t]

    return dDeparture

#-----------------------------------------------------------

def kd_opObs(x, g,  dObs, H_op, kdvParam, maxA):
    """
        Non-linear observation operator

        x       :   state <numpy.ndarray>
        g       :   <SpectralGrid>
        dObs    :   {time <float>   :   idxObs <np.array>, ...} <dict>
        H_op    :   static observation operator
    """
    if not (isinstance(dObs, dict)): 
        raise obsTimeOpError("dObs <dict>")
    for t in dObs.iterkeys():
        if not isinstance(dObs[t], np.ndarray):
            raise obsTimeOpError("dObs[t] <numpy.ndarray>")

    #----| Model equivalent |-----------
    HMx={}
    for t in dObs.iterkeys():
        # parallelize this?
        tLauncher=kdv.Launcher(kdvParam, x)
        traj=tLauncher.integrate(t, maxA)
        HMx[t]=H_op(traj.final(), g, pos2Idx(g, dObs[t]))

    return HMx


#-----------------------------------------------------------

def kd_opObs_TL(dx, x_bkg, g,  dObs, H_op, kdvParam, maxA):
    """
        tangent linear observation operator

        dx      :   state increment <numpy.ndarray>
        x_bkg   :   background state <numpy.ndarray>
        g       :   <SpectralGrid>
        dObs    :   {time <float>   :   idxObs <np.array>, ...} <dict>
        H_op    :   static observation operator
    """
    if not (isinstance(dObs, dict)): 
        raise obsTimeOpError("dObs <dict>")
    for t in dObs.iterkeys():
        if not isinstance(dObs[t], np.ndarray):
            raise obsTimeOpError("dObs[t] <numpy.ndarray>")

    #----| Model equivalent |-----------
    tInt=np.max(dObs.keys())
    launcher_bkg=kdv.Launcher(kdvParam, x_bkg)
    traj_bkg=launcher_bkg.integrate(tInt, maxA)
    HMdx={}
    t_pre=0.
    dx_pre=dx
    for t in dObs.iterkeys():
        tLauncher=kdv.TLMLauncher(kdvParam, traj_bkg, dx_pre)
        dx_t=tLauncher.integrate(tInt=t-t_pre, t0=t_pre)
        HMdx[t]=H_op(dx_t, g, pos2Idx(g, dObs[t]))
        t_pre=t
        dx_pre=dx_t
    return HMdx
 
#-----------------------------------------------------------

def kd_opObs_TL_nonSequential(dx, x_bkg, g,  dObs, H_op, kdvParam, maxA):
    """
        tangent linear observation operator

        dx      :   state increment <numpy.ndarray>
        x_bkg   :   background state <numpy.ndarray>
        g       :   <SpectralGrid>
        dObs    :   {time <float>   :   idxObs <np.array>, ...} <dict>
        H_op    :   static observation operator
    """
    if not (isinstance(dObs, dict)): 
        raise obsTimeOpError("dObs <dict>")
    for t in dObs.iterkeys():
        if not isinstance(dObs[t], np.ndarray):
            raise obsTimeOpError("dObs[t] <numpy.ndarray>")

    #----| Model equivalent |-----------
    tInt=np.max(dObs.keys())
    launcher_bkg=kdv.Launcher(kdvParam, x_bkg)
    traj_bkg=launcher_bkg.integrate(tInt, maxA)
    HMdx={}
    for t in dObs.iterkeys():
        # parallelize this?
        tLauncher=kdv.TLMLauncher(kdvParam, traj_bkg, dx)
        dx_t=tLauncher.integrate(t)
        HMdx[t]=H_op(dx_t, g, pos2Idx(g, dObs[t]))

    return HMdx
 

#===========================================================
if __name__=="__main__":
    import matplotlib.pyplot as plt
    from dVar import opObs_Idx, degrad

    grid=kdv.SpectralGrid(150,300.)
    tInt=3.
    maxA=2.

    param=kdv.Param(grid, beta=1., gamma=-1.)

    x0_truth=kdv.rndFiltVec(grid, Ntrc=grid.Ntrc/5,  amp=1.)
    launcher=kdv.Launcher(param, x0_truth)
    x_truth=launcher.integrate(tInt, maxA)
    x0_degrad=degrad(x0_truth, 0., 0.3)


    x_bkg=np.zeros(grid.N)

    dObs={}
    dObs[0.1]=np.array([-30.,  70.])
    dObs[0.5]=np.array([-120., -34., -20., 2.,  80., 144.])
    dObs[1.2]=np.array([-90., -85, 4., 10.])
    dObs[2.5]=np.array([-50., 0., 50.])
    obs_degrad=kd_opObs(x0_degrad, grid,  dObs, opObs_Idx,
                            param, maxA)
    obs_truth=kd_opObs(x0_truth, grid,  dObs, opObs_Idx,
                            param, maxA)


    


    nTime=len(obs_degrad.keys())
    i=0
    for t in np.sort(obs_degrad.keys()):
        i+=1
        plt.subplot(nTime, 1, i)
        ti=whereTrajTime(x_truth, t)
        plt.plot(grid.x, x_truth[ti], 'k')
        plt.plot(grid.x[pos2Idx(grid, dObs[t])], obs_degrad[t], 'ro')
        plt.plot(grid.x[pos2Idx(grid, dObs[t])], obs_truth[t], 'go')


