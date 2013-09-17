import numpy as np
from kd_observationOp import kd_departure, kd_opObs, kd_opObs_TL,\
                                kd_opObs_TL_Adj
from dVar import gradTest,  printGradTest

def costFunc(xi, x_b, var, B_sqrt_op, B_sqrt_op_Adj,
                    H, model, H_TL_Adj, tlmLauncher, tlmLArgs,
                    argsH, dObs, dR_inv, rCTilde_sqrt,
                    background=True):

    if background:
        J_xi=0.5*np.dot(xi, xi)
        x=B_sqrt_op(xi, var, rCTilde_sqrt)+x_b
    else:
        J_xi=0.
        x=xi


    dD=kd_departure(x, H, (model,)+argsH, dObs)
    J_o=0.
    for t in dD.keys():
        J_o+=0.5*np.dot(dD[t],np.dot(dR_inv[t],dD[t]))
    return J_xi+J_o

def gradCostFunc(xi, x_b, var, B_sqrt_op, B_sqrt_op_Adj,
                    H, model, H_TL_Adj, tlmLauncher, tlmLArgs,
                    argsH, dObs, dR_inv, rCTilde_sqrt,
                    background=True):

    if background:
        x=B_sqrt_op(xi, var, rCTilde_sqrt)+x_b
    else:
        x=xi

    dDep=kd_departure(x, H, (model,)+ argsH, dObs)

    dNormDep={}
    for t in dDep.keys():
        dNormDep[t]=np.dot(dR_inv[t],dDep[t])

    #----| building reference trajectory |--------
    tInt=np.max(dDep.keys())
    traj_x=model.integrate(x, tInt)
    tlm=tlmLauncher(traj_x, *tlmLArgs)

    dx0=H_TL_Adj(dNormDep, tlm, *argsH)
    if background:
        gradJ_o=-B_sqrt_op_Adj(dx0, var, rCTilde_sqrt)
        gradJ=xi+gradJ_o
    else:
        gradJ=-dx0
    
    return gradJ


#===========================================================
if __name__=="__main__":
    import matplotlib.pyplot as plt
    from dVar import pos2Idx, fCorr_isoHomo, degrad, \
                        B_sqrt_op, B_sqrt_op_Adj, \
                        rCTilde_sqrt_isoHomo, opObs_Idx, opObs_Idx_Adj
    import pyKdV as kdv
    
    
    Ntrc=100
    L=300.
    g=kdv.SpectralGrid(Ntrc, L)
        
    kdvParam=kdv.Param(g, beta=1., gamma=-1.)
    tInt=3.
    maxA=5.
    
    model=kdv.Launcher(kdvParam, maxA)

    x0_truth_base=kdv.rndFiltVec(g, Ntrc=g.Ntrc/5,  amp=1.)
    soliton=kdv.soliton(g.x, 0., amp=5., beta=1., gamma=-1)
    gaussWave=1.5*kdv.gauss(g.x, 40., 20. )-1.*kdv.gauss(g.x, -20., 14. )

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
    H_TL_Adj=kd_opObs_TL_Adj
    staticObsOp=None
    sObsOpArgs=()
    argsHcom=(g, dObsPos, staticObsOp, sObsOpArgs)

    sigR=.5
    dObs=H(x0_truth, model, *argsHcom) 
                         
    
    dR_inv={}
    for t in dObsPos.keys():
        dR_inv[t]=sigR**(-1)*np.eye(len(dObsPos[t]))
    
    
    #----| Preconditionning |-----
    Lc=10.
    sig=0.4
    corr=fCorr_isoHomo(g, Lc)
    rCTilde_sqrt=rCTilde_sqrt_isoHomo(g, corr)
    var=sig*np.ones(g.N)
    xi=np.zeros(g.N)


    #----| Gradient test |--------
    argsCostFunc=(x0_bkg, var, B_sqrt_op, B_sqrt_op_Adj,
                    H, model, H_TL_Adj, kdv.TLMLauncher, (kdvParam,),
                    argsHcom, dObs, dR_inv, rCTilde_sqrt)

    printGradTest(gradTest(costFunc, gradCostFunc, xi, argsCostFunc))
    

