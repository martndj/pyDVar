from jTerm import JTerm
from observations import StaticObs, degrad
from pseudoSpec1D import SpectralGrid
import numpy as np

class StaticObsJTerm(JTerm):

    class StaticObsJTermError(Exception):
        pass
        
        
    #------------------------------------------------------
    #----| Init |------------------------------------------
    #------------------------------------------------------

    def __init__(self, obs, g, obsOpTLMAdj, obsOpTLMAdjArgs=(), 
                    maxiter=100, retall=True, testAdj=False,
                    testGrad=True, testGradMinPow=-1, testGradMaxPow=-14):

        if not isinstance(obs, StaticObs):
            raise StaticObsJTermError("obs <SaticObs>")
        self.obs=obs

        if not isinstance(g, SpectralGrid):
            raise StaticObsJTermError("g <pseudoSpec1D.SpectralGrid>")
        self.modelGrid=g

        if not ((callable(obsOpTLMAdj) or obsOpTLMAdj==None)
                and isinstance(obsOpTLMAdjArgs, tuple)):
            raise StaticObsJTermError("obsOpTLMAdj <function | None>, obsOpTLMAdjArgs <tuple>")
        self.obsOpTLMAdj=obsOpTLMAdj
        self.obsOpTLMAdjArgs=obsOpTLMAdjArgs

        self.args=()
        
        self.maxiter=maxiter
        self.retall=retall
        self.testAdj=testAdj
        self.testGrad=testGrad
        self.testGradMinPow=-1
        self.testGradMaxPow=-14
        #super(PrecondJTerm, self).__configure(maxiter,
        #                                    retall, testAdj, testGrad, 
        #                                    testGradMinPow, testGradMaxPow)

    #------------------------------------------------------
    #----| Public methods |--------------------------------
    #------------------------------------------------------

    def J(self, x): 
        inno=self.obs.innovation(x, self.modelGrid)
        return 0.5*np.dot(inno, np.dot(self.obs.metric, inno)) 

    #------------------------------------------------------

    def gradJ(self, x):
        inno=self.obs.innovation(x, self.modelGrid)
        if self.obsOpTLMAdj==None:
            return -np.dot(self.obs.metric, inno)
        else:
            return -self.obsOpTLMAdj(np.dot(self.obs.metric, inno), 
                                *obs.obsOpTLMAdjArgs)

#=====================================================================
#---------------------------------------------------------------------
#=====================================================================


if __name__=='__main__':

    import matplotlib.pyplot as plt
    import pyKdV as kdv
    from jTerm import PrecondJTerm
    
    Ntrc=100
    L=300.
    g=SpectralGrid(Ntrc, L)
        

    x0_truth_base=kdv.rndFiltVec(g, Ntrc=g.Ntrc/5,  amp=1.)
    gaussWave=1.5*kdv.gauss(g.x, 40., 20. )-1.*kdv.gauss(g.x, -20., 14. )
    soliton=kdv.soliton(g.x, 0., amp=5., beta=1., gamma=-1)

    x0_truth=x0_truth_base+gaussWave
    x0_degrad=degrad(x0_truth, 0., 0.3)

    obs1=StaticObs(g, x0_truth, None)
    
    JObs=StaticObsJTerm(obs1, g, None) 
    
    #----| First test: only degrated first guess
    JObs.minimize(x0_degrad)

    #----| Second test: with null background term
    Jbkg=PrecondJTerm()
    JSum=JObs+Jbkg
    JSum.minimize(x0_degrad)
    
