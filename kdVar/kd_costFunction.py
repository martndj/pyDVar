import numpy as np
from kd_observationOp import kd_departure, kd_opObs, kd_opObs_TL,\
                                kd_opObs_TL_Adj
from dVar import gradTest,  printGradTest

def costFunc(xi, traj_b, var, B_sqrt_op, B_sqrt_op_Adj,
                    H, H_TL, H_TL_Adj, argsH, dObs, dR_inv,
                    rCTilde_sqrt):


    J_xi=0.5*np.dot(xi, xi)


    x=B_sqrt_op(xi, var, rCTilde_sqrt)+traj_b[0]
    dD=kd_departure(x, traj_b, H, H_TL, argsH, dObs)
    J_o=0.
    for t in dD.keys():
        J_o+=0.5*np.dot(dD[t],np.dot(dR_inv[t],dD[t]))
    return J_xi+J_o

def gradCostFunc(xi, traj_b, var, B_sqrt_op, B_sqrt_op_Adj,
                    H, H_TL, H_TL_Adj, argsH, dObs, dR_inv,
                    rCTilde_sqrt):

    x=B_sqrt_op(xi, var, rCTilde_sqrt)+traj_b[0]
    dDep=kd_departure(x, traj_b, H, H_TL, argsH, dObs)

    dNormDep={}
    for t in dDep.keys():
        dNormDep[t]=np.dot(dR_inv[t],dDep[t])

    # <TODO>: H_TL_Adj must be linearized around x(t) and not x_b(t)
    gradJ_o=-B_sqrt_op_Adj(H_TL_Adj(dNormDep, traj_b, *argsH), 
                        var, rCTilde_sqrt)

    
    return xi+gradJ_o


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
    wave=kdv.soliton(g.x, 0., amp=5., beta=1., gamma=-1)\
                +1.5*kdv.gauss(g.x, 40., 20. )-1.*kdv.gauss(g.x, -20., 14. )
    x0_truth=x0_truth_base+wave
    x_truth=model.integrate(x0_truth, tInt)

    x0_bkg=x0_truth_base
    x_bkg=model.integrate(x0_bkg, tInt)
    
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
    
    
    #----| Preconditionning |-----
    Lc=10.
    sig=0.4
    corr=fCorr_isoHomo(g, Lc)
    rCTilde_sqrt=rCTilde_sqrt_isoHomo(g, corr)
    var=sig*np.ones(g.N)
    xi=np.zeros(g.N)


    #----| Gradient test |--------
    argsCostFunc=(x_bkg, var, B_sqrt_op, B_sqrt_op_Adj,
                    H, H_TL, H_TL_Adj, argsHcom, dObs_degrad, dR_inv,
                    rCTilde_sqrt)

    printGradTest(gradTest(costFunc, gradCostFunc, xi, argsCostFunc))
    

