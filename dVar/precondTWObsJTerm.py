from observations import StaticObs, TimeWindowObs
from obsJTerm import TWObsJTerm
import numpy as np

class PrecondTWObsJTerm(TWObsJTerm):
    """
    Preconditionned time window observations JTerm subclass

    PrecondTWObsJTerm(obs, nlModel, tlm
                        x_bkg, B_sqrt, B_sqrtAdj, B_sqrtArgs=())

        obs             :   <StaticObs>
        nlModel         :   propagator model <Launcher>
        tlm             :   tangean linear model <TLMLauncher>
        x_bkg           :   background state <numpy.ndarray>
        B_sqrt          :   preconditionning operator <function>
        B_sqrtAdj       :   adjoint of preconditionning op. <function>
        B_sqrtArgs      :   arguments <tuple>
                                
    The purpose of this class is to facilitate the convergence of a cost
    function of the form:
        
        J(x)= 0.5*(x-x_bkg)'B^{-1}(x-x_bkg) + 0.5*(y-H(x))'R{-1}(y-H(x))

    by operating a variable change:
        
        xi=B^{-1/2}(x-x_bkg)

    so that the cost function is now in term of xi:
        
        J(xi)= xi'xi + 0.5*(y-H(x))'R{-1}(y-H(x))
        x=B^{1/2}xi+x_b

    Using PrecondTWObsJTerm() we can represent the background term with
    TrivialJTerm() and sum them.
    """
    
    class PrecondTWObsJTermError(Exception):
        pass

    #------------------------------------------------------
    #----| Init |------------------------------------------
    #------------------------------------------------------

    def __init__(self, obs, nlModel, tlm, minimizer=None,
                    x_bkg, B_sqrt, B_sqrtAdj, B_sqrtArgs=()):

        super(PrecondTWObsJTerm, self).__init__(obs, nlModel, tlm)  

        if not (callable(B_sqrt) and callable(B_sqrtAdj)):
            raise PrecondTWObsJTermError("B_sqrt[Adj] <function>")
        if not (isinstance(B_sqrtArgs, tuple)):
            raise PrecondTWObsJTermError("B_sqrtArgs <tuple>")
        self.B_sqrt=B_sqrt
        self.B_sqrtAdj=B_sqrtAdj
        self.B_sqrtArgs=B_sqrtArgs
    
        self.__xValidate(x_bkg)
        self.x_bkg=x_bkg

        self.setMinimizer(minimizer)
        self.isMinimized=False
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
    #------------------------------------------------------

    def minimize(self, xi, 
                    maxiter=50, retall=True, testGrad=True, 
                    testGradMinPow=-1, testGradMaxPow=-14):
        super(PrecondTWObsJTerm, self).minimize(xi, maxiter, retall,
                                                testGrad, 
                                                testGradMinPow, 
                                                testGradMaxPow)
        self.x_a=self.B_sqrt(self.analysis, *self.B_sqrtArgs)+self.x_bkg

    #------------------------------------------------------
    #----| Classical overloads |----------------------------
    #-------------------------------------------------------

    def __str__(self):
        output="////| PrecondTWObsJTerm |//////////////////////////////////"
        output+="\n B_sqrt:\n  %s"%self.B_sqrt
        output+="\n B_sqrtAdj:\n  %s"%self.B_sqrtAdj

        if self.isMinimized:
            if self.warnFlag:
                output+="\n <!> Warning %d <!>"%self.warnFlag
            output+="\n function value=%f"%self.fOpt
            output+="\n gradient norm=%f"%self.gOptNorm
            output+="\n function calls=%d"%self.fCalls
            output+="\n gradient calls=%d"%self.gCalls
        else:
            output+="\n Not minimized"
        output+="\n///////////////////////////////////////////////////////////\n"
        return output
#=====================================================================
#---------------------------------------------------------------------
#=====================================================================

if __name__=='__main__':

    import matplotlib.pyplot as plt
    from observations import degrad, obsOp_Coord, obsOp_Coord_Adj
    from modelCovariances import B_sqrt_op, B_sqrt_op_Adj,\
                                    fCorr_isoHomo,\
                                    rCTilde_sqrt_isoHomo
    import pyKdV as kdv
    from jTerm import TrivialJTerm
    from pseudoSpec1D import PeriodicGrid 
    
    Ntrc=100
    L=300.
    g=PeriodicGrid(Ntrc, L)
    
    kdvParam=kdv.Param(g, beta=1., gamma=-1.)
    tInt=15.

    model=kdv.kdvLauncher(kdvParam, maxA=4.)
    tlm=kdv.kdvTLMLauncher(kdvParam)
    
    base=kdv.rndSpecVec(g, Ntrc=10,  amp=0.4)
    soliton=kdv.soliton(g.x, 0., amp=1.5, beta=1., gamma=-1)
    longWave=0.8*kdv.gauss(g.x, 40., 20. )-0.5*kdv.gauss(g.x, -20., 14. )

    x0_truth=soliton
    x_truth=model.integrate(x0_truth, tInt)

    x0_bkg=np.zeros(g.N)
    x_bkg=model.integrate(x0_bkg, tInt)

    
    nObsTime=9
    nPosObs=50
    d_Obs={}
    for i in xrange(nObsTime):
        t=tInt*(i+1)/nObsTime
        obsPos=np.linspace(-g.L/2., g.L/2.-g.dx, nPosObs)
        obsValues=obsOp_Coord(x_truth.whereTime(t), g, obsPos)
        d_Obs[t]=StaticObs(obsPos, obsValues, obsOp_Coord, obsOp_Coord_Adj)
    timeObs=TimeWindowObs(d_Obs)

    #----| Preconditionning |-----------
    Lc=10.
    sig=2.
    corr=fCorr_isoHomo(g, Lc)
    rCTilde_sqrt=rCTilde_sqrt_isoHomo(g, corr)
    var=sig*np.ones(g.N)
    B_sqrtArgs=(var, rCTilde_sqrt)
    xi0=np.zeros(g.N)

    JPTWObs=PrecondTWObsJTerm(timeObs, model, tlm, 
                        x0_bkg, B_sqrt_op, B_sqrt_op_Adj, B_sqrtArgs) 

    # J=0.5<xi.T,xi>+0.5<(y-H(x)).T,R_inv(y-H(x))>
    #   x=B_sqrt(xi)+x_bkg
    Jxi=TrivialJTerm()
    J=Jxi+JPTWObs*0.1
    # the scaling compensate the huge number of observations making
    # JPTWObs >> Jxi 

    J.minimize(x0_bkg)
    x0_a=B_sqrt_op(J.analysis, *B_sqrtArgs)+x0_bkg
    x_a=model.integrate(x0_a,  tInt)

    nSubRow=3
    nSubLine=timeObs.nObs/nSubRow+1
    if timeObs.nObs%nSubRow: nSubLine+=1
    plt.figure(figsize=(12.,12.))
    i=0
    for t in timeObs.times:
        i+=1
        sub=plt.subplot(nSubLine, nSubRow, nSubRow+i)
        sub.plot(g.x, x_truth.whereTime(t), 'g')
        sub.plot(timeObs[t].interpolate(g), timeObs[t].values, 'go')
        sub.plot(g.x, x_bkg.whereTime(t), 'b')
        sub.plot(timeObs[t].interpolate(g), 
                    x_bkg.whereTime(t)[g.pos2Idx(timeObs[t].coord)], 'bo')
        sub.set_title("$t=%f$"%t)
        if i==timeObs.nObs:
            sub.legend(["$x_{t}$",  "$y$", "$x_b$", 
                        "$H(x_b)$"], loc="lower left")
    sub=plt.subplot(nSubLine, 1,1)
    sub.plot(g.x, x0_truth, 'k--')
    sub.plot(g.x, x0_bkg, 'b--')
    sub.plot(g.x, x0_a, 'r--')
    sub.plot(g.x, x_truth.final, 'k')
    sub.plot(g.x, x_bkg.final, 'b')
    sub.plot(g.x, x_a.final, 'r')
    sub.legend(["${x_t}_0$","${x_b}_0$","${x_a}_0$",
                "${x_t}_f$","${x_b}_f$","${x_a}_f$"], loc='best')
    plt.show()
