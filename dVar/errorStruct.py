import numpy as np
from modelCovariances import make_BisoHomo_args,  B_sqrt_isoHomo_op
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec, SubplotSpec

def checkAxe(axe):
    if axe==None:
        axe=plt.subplot(111)
    if isinstance(axe, int):
        if len(str(axe))<>3:
            raise ValueError(
                "Single argument to subplot must be a 3-digit integer")
        axe=plt.subplot(axe)
    elif isinstance(axe,SubplotSpec):
        axe=plt.subplot(axe)
    elif isinstance(axe,Axes):
        pass
    return axe

def errStr_isoHomo(grid, bkgLC, bkgSig=1., seed=None):
    '''
    Produce a random isotropic and homogeneous error structure 
        (coherent with the statics assimilation statistics using
            B_sqrt_isoHomo_op)
    '''
    xi=grid.zeros()
    B_args=make_BisoHomo_args(grid, bkgLC, bkgSig)

    np.random.seed(seed)
    for i in xrange(grid.N):
        xi[i]=np.random.normal()
    return B_sqrt_isoHomo_op(xi, *B_args)

def sample_err_isoHomo(grid, bkgLC, bkgSig=1., nRlz=1000, std=False):
    errPS=[]
    for i in xrange(nRlz):
        rlz=errStr_isoHomo(grid, bkgLC, bkgSig=bkgSig)
        nDemi=int(grid.N-1)/2
        data=np.zeros(nDemi)
        data=np.abs(np.fft.fft(rlz)[0:nDemi])
        errPS.append(data)
    
    errPSMean=np.mean(errPS, axis=0)
    if not std:
        return errPSMean
    else:
        errPSStd=np.std(errPS, axis=0)
        return errPSMean, errPSStd

def plot_err_isoHomo(grid, bkgLC, bkgSig=1., nRlz=1000, 
                    fill_between=True, axe=None, alpha=0.2):
    axe=checkAxe(axe)
    if fill_between:
        errPSMean, errPSStd=sample_err_isoHomo(grid, bkgLC, 
                                bkgSig=bkgSig, nRlz=nRlz,
                                std=True)
    else:
        errPSMean=sample_err_isoHomo(grid, bkgLC, 
                                bkgSig=bkgSig, nRlz=nRlz)
    N=len(errPSMean)
    k=np.linspace(0,N-1, N)
    axe.plot(k, errPSMean)
    if fill_between:
        axe.fill_between(k, errPSMean+errPSStd, errPSMean-errPSStd, 
                            alpha=alpha)
    return axe
