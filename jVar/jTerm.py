import numpy as np
from pseudoSpec1D import SpectralGrid, Trajectory, Launcher, TLMLauncher

class JTerm(object):
    """
        Nuclear cost function term for assimilating an information type

    """

    class JTermError(Exception):
        pass

    #------------------------------------------------------
    #----| Init |------------------------------------------
    #------------------------------------------------------

    def __init__(self, tWObs, nlModel, tlm, 
                    obsOpTLAdj, obsOpTLAdjArgs=(),
                    maxiter=100, testAdj=True, testGrad=True):
        
        if not isinstance(tWObs, TimeWindowObs):
            raise JTermError("tWObs <TimeWindowObs>")
        self.obs=tWObs
        self.times=tWObs.times

        if not (isinstance(nlModel,Launcher) or nlModel==None):
            raise self.JTermError("nlModel <Launcher | None>")
        if not (isinstance(tlm, TLMLauncher) or tlm==None):
            raise self.JTermError("tlm <TLMLauncher | None>")        
        if (tlm==None or nlModel==None)and self.times<>np.zeros(1):
            raise slef.JTermError("tlm|nlModel==None <=> self.times=[0.]")
        self.nlModel=nlModel
        self.tlm=tlm
        
        if not (callable(obsOpTLAdj) or obsOpTLAdj==None):
            raise self.JTermError("obsOpTLAdj <function | None>")
        if not isinstance(obsOpTLAdjArgs, tuple):
            raise self.JTermError("obsOpTLAdjArgs <tuple>")
        self.obsOp=tWObs.obsOp
        self.obsOpArgs=tWObs.obsOpArgs
        self.obsOpTLAdj=obsOpTLAdj
        self.obsOpTLAdjArgs=obsOpTLAdjArgs

    #------------------------------------------------------
    #----| Public methods |--------------------------------
    #------------------------------------------------------

    def J(self, x, g):
        d_inno=self.obs.innovation(x, sel.nlModel)
        J=0.
        for t in self.times:
            innoNorm=np.dot(self.obs[t].metric, d_inno[t])
            J+=0.5*np.dot(d_inno[t],innoNorm)

    #------------------------------------------------------

    def gradJ(self, x, g):
        d_inno=self.obs.innovation(x, self.nlModel)
        d_innoNorm={}
        for t in self.times:
            d_innoNorm[t]=np.dot(self.obs[t].metric, d_inno[t])

        tInt=np.max(self.times)
        traj_x=self.nlModel.integrate(x, tInt)
        self.tlm.initialize(traj_x)

        dx0=self.obsOpTLAdj(d_innoNorm, tlm, *self.obsOpTLAdjArgs)
        return 


        
            

            

        
        
