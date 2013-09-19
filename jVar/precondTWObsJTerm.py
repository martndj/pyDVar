#from jTerm import JTerm
from observations import StaticObs, TimeWindowObs
from obsJTerm import TWObsJTerm
import numpy as np

class PrecondTWObsJTerm(TWObsJTerm):
    """

    """
    
    class PrecondTWObsJTermError(Exception):
        pass

    #------------------------------------------------------
    #----| Init |------------------------------------------
    #------------------------------------------------------

    def __init__(self, obs, nlModel, tlm,
                    x_bkg, B_sqrt, B_sqrtAdj, B_sqrtArgs=(),
                    maxiter=100, retall=True, testAdj=False,
                    testGrad=True, testGradMinPow=-1, testGradMaxPow=-14):

        super(PrecondTWObsJTerm, self).__init__(obs, nlModel, tlm,  
                        maxiter=maxiter, retall=retall, testAdj=testAdj,
                        testGrad=testGrad, testGradMinPow=testGradMinPow,
                        testGradMaxPow=testGradMaxPow)
        if not (callable(B_sqrt) and callable(B_sqrtAdj)):
            raise PrecondTWObsJTermError("B_sqrt[Adj] <function>")
        if not (isinstance(B_sqrtArgs, tuple)):
            raise PrecondTWObsJTermError("B_sqrtArgs <tuple>")
        self.B_sqrt=B_sqrt
        self.B_sqrtAdj=B_sqrtAdj
        self.B_sqrtArgs=B_sqrtArgs
    
        self.__xValidate(x_bkg)
        self.x_bkg=x_bkg

    #------------------------------------------------------
    #----| Private methods |-------------------------------
    #------------------------------------------------------

    def __xValidate(self, xi):
        if not isinstance(xi, np.ndarray):
            raise self.PrecondTWObsJTermError("xi <numpy.array>")
        if not xi.dtype=='float64':
            raise self.PrecondTWObsJTermError("xi.dtype=='float64'")
        if xi.ndim<>1:
            raise self.PrecondTWObsJTermError("xi.ndim==1")
        if len(xi)<>self.nlModel.grid.N:
            raise self.PrecondTWObsJTermError(
                "len(xi)==self.nlModel.grid.N")

    #------------------------------------------------------
    #----| Public methods |--------------------------------
    #------------------------------------------------------

    def J(self, xi): 
        self.__xValidate(xi)
        x=self.B_sqrt(xi, *self.B_sqrtArgs)+self.x_bkg
        return super(PrecondTWObsJTerm, self).J(x)
    #------------------------------------------------------

    def gradJ(self, xi):
        self.__xValidate(xi)
        x=self.B_sqrt(xi, *self.B_sqrtArgs)+self.x_bkg

        dx0=super(PrecondTWObsJTerm, self).gradJ(x)
        return self.B_sqrtAdj(dx0,*self.B_sqrtArgs)

#=====================================================================
#---------------------------------------------------------------------
#=====================================================================

if __name__=='__main__':

    import matplotlib.pyplot as plt
    from observations import degrad, pos2Idx, obsOp_Coord
    from modelCovariances import B_sqrt_op, B_sqrt_op_Adj, fCorr_isoHomo,\
                                    rCTilde_sqrt_isoHomo
    import pyKdV as kdv
    from jTerm import TrivialJTerm
    from pseudoSpec1D import SpectralGrid 
    
    Ntrc=100
    L=300.
    g=SpectralGrid(Ntrc, L)
    

    rndLFBase=kdv.rndFiltVec(g, Ntrc=g.Ntrc/5,  amp=0.4)
    soliton=kdv.soliton(g.x, 0., amp=1.3, beta=1., gamma=-1)

    x0_truth=rndLFBase+soliton
    x0_degrad=degrad(x0_truth, 0., 0.3)
    x0_bkg=rndLFBase

    def gaussProfile(x):
        return 0.03*kdv.gauss(x, 40., 20. )\
                -0.02*kdv.gauss(x, -20., 14. )

    kdvParam=kdv.Param(g, beta=1., gamma=-1.)#, rho=gaussProfile)
    tInt=10.
    maxA=2.
    maxiter=50
    
    model=kdv.kdvLauncher(kdvParam, maxA)
    tlm=kdv.kdvTLMLauncher(kdvParam)
    x_truth=model.integrate(x0_truth, tInt)
    x_degrad=model.integrate(x0_degrad, tInt)
    x_bkg=model.integrate(x0_bkg, tInt)

    nObsTime=3
    d_Obs1={}
    for i in xrange(nObsTime):
        d_Obs1[tInt*(i+1)/nObsTime]=StaticObs(g,
            x_truth.whereTime(tInt*(i+1)/nObsTime))
    timeObs1=TimeWindowObs(d_Obs1)

    #----| Preconditionning |-----------
    Lc=10.
    sig=2.
    corr=fCorr_isoHomo(g, Lc)
    rCTilde_sqrt=rCTilde_sqrt_isoHomo(g, corr)
    var=sig*np.ones(g.N)
    B_sqrtArgs=(var, rCTilde_sqrt)
    xi0=np.zeros(g.N)

    JPTWObs=PrecondTWObsJTerm(timeObs1, model, tlm, 
                        x0_bkg, B_sqrt_op, B_sqrt_op_Adj, B_sqrtArgs) 

    # J=0.5<xi.T,xi>+0.5<(y-H(x)).T,R_inv(y-H(x))>
    #   x=B_sqrt(xi)+x_bkg
    Jxi=TrivialJTerm()
    J=Jxi+JPTWObs*0.1
    J.minimize(xi0)
    x_a=model.integrate(J.analysis, tInt)

    plt.figure()
    i=0
    for t in timeObs1.times:
        i+=1
        sub=plt.subplot(nObsTime+1, 1, i+1)
        sub.plot(g.x, x_truth.whereTime(t), 'k', linewidth=2.5)
        sub.plot(timeObs1[t].interpolate(g), timeObs1[t].values, 'g')
        sub.plot(g.x, x_a.whereTime(t), 'r')
        sub.set_title("t=%.2f"%t)
    sub=plt.subplot(nObsTime+1, 1, 1)
    sub.plot(g.x, J.analysis, 'r')
    sub.plot(g.x, x0_truth, 'k', linewidth=2.5)
    plt.show()
