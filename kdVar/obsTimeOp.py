import numpy as np
import pyKdV as kdv
from dVar import pos2Idx

class obsTimeOpError(Exception):
    pass

def whereTrajTime(u, time):
    return np.where(u.time>=time)[0].min()


def opObs_kdv(x, g,  dObs, H_op, kdvArgs, verbose=False):
    """
        Non-linear observation operator

        x       :   state <numpy.ndarray>
        g       :   <SpectralGrid>
        dObs    :   {time <float>   :   idxObs <np.array>, ...} <dict>
        H_op    :   static observation operator
        kdvArgs :   (param, maxA, 
    """
    if not (isinstance(dObs, dict)): 
        raise obsTimeOpError("dObs <dict>")
    for t in dObs.iterkeys():
        if not isinstance(dObs[t], np.ndarray):
            raise obsTimeOpError("dObs[t] <numpy.ndarray>")

        

    #----| KdV arguments |--------------
    kdvParam=kdvArgs[0]
    maxA=kdvArgs[1]

    #----| Model equivalent |-----------
    HMx={}
    for t in dObs.iterkeys():
        tLauncher=kdv.Launcher(kdvParam, x)
        traj=tLauncher.integrate(t, maxA)
        HMx[t]=H_op(traj.final(), g, pos2Idx(g, dObs[t]))

    return HMx



#===========================================================
if __name__=="__main__":
    import matplotlib.pyplot as plt
    from dVar import opObs_Idx_op, degrad

    grid=kdv.SpectralGrid(150,300.)
    tInt=3.
    maxA=2.

    param=kdv.Param(grid, beta=1., gamma=-1.)
    kdvArgs=(param, maxA)

    x0_truth=kdv.rndFiltVec(grid, Ntrc=grid.Ntrc/5,  amp=1.)
    launcher=kdv.Launcher(param, x0_truth)
    x_truth=launcher.integrate(tInt, maxA)
    x0_degrad=degrad(x0_truth, 0., 0.3)

    dObs={}
    dObs[0.1]=np.array([-30.,  70.])
    dObs[0.5]=np.array([-120., -34., -20., 2.,  80., 144.])
    dObs[1.2]=np.array([-90., -85, 4., 10.])
    dObs[2.5]=np.array([-50., 0., 50.])
    obs_degrad=opObs_kdv(x0_degrad, grid,  dObs, opObs_Idx_op,
                            kdvArgs)
    obs_truth=opObs_kdv(x0_truth, grid,  dObs, opObs_Idx_op,
                            kdvArgs)

    nTime=len(obs_degrad.keys())
    i=0
    for t in np.sort(obs_degrad.keys()):
        i+=1
        plt.subplot(nTime, 1, i)
        ti=whereTrajTime(x_truth, t)
        plt.plot(grid.x, x_truth[ti], 'k')
        plt.plot(grid.x[pos2Idx(grid, dObs[t])], obs_degrad[t], 'ro')
        plt.plot(grid.x[pos2Idx(grid, dObs[t])], obs_truth[t], 'go')


