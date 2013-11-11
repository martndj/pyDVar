from observations import StaticObs, TimeWindowObs
from jTerm import JTerm, JMinimum
from obsJTerm import TWObsJTerm, StaticObsJTerm
import numpy as np

class PrecondJTerm(JTerm):
    '''

        <!> This is a master class not meant to be instantiated, only
            subclasses should.
    '''
    class PrecondJTermError(Exception):
        pass
    
    #------------------------------------------------------
    #----| Private methods |-------------------------------
    #------------------------------------------------------


    def _xValidate(self, xi):
        if not isinstance(xi, np.ndarray):
            raise self.PrecondJTermError("xi <numpy.array>")
        if not xi.dtype=='float64':
            raise self.PrecondJTermError("xi.dtype=='float64'")
        if xi.ndim<>1:
            raise self.PrecondJTermError("xi.ndim==1")
        if len(xi)<>self.modelGrid.N:
            raise self.PrecondJTermError(
                "len(xi)==self.grid.N")

    #------------------------------------------------------
    #----| Public methods |--------------------------------
    #------------------------------------------------------
    def BSqrt(self, xi):
        return self.B_sqrt(xi, *self.B_sqrtArgs)+self.x_bkg

    #------------------------------------------------------

    def J(self, xi, normalize=False): 
        self._xValidate(xi)
        x=self.BSqrt(xi)
        return super(PrecondJTerm, self).J(x)+0.5*np.dot(xi,xi)

    #------------------------------------------------------

    def gradJ(self, xi, normalize=False):
        self._xValidate(xi)
        x=self.BSqrt(xi)

        dx0=super(PrecondJTerm, self).gradJ(x)
        return self.B_sqrtAdj(dx0,*self.B_sqrtArgs)+xi

    #------------------------------------------------------
    def minimize(self, x_fGuess, maxiter=50, retall=True,
                    testGrad=True, convergence=True, 
                    testGradMinPow=-1, testGradMaxPow=-14):
        super(PrecondJTerm, self).minimize(
                    x_fGuess, maxiter=50, retall=True,
                    testGrad=True, convergence=True, 
                    testGradMinPow=-1, testGradMaxPow=-14)
        self.analysis=self.BSqrt(self.minimum.xOpt)
#    def createMinimum(self, minimizeReturn, maxiter, convergence=True):
#
#        if self.retall:
#            allvecs=[]
#            allvecs_precond=minimizeReturn[7]
#            for vec in allvecs_precond:
#                allvecs.append(self.BSqrt(vec))
#            if convergence:
#                convJVal=self.jAllvecs(allvecs_precond)
#        else:
#            allvecs=None
#            convJVal=None
#
#        self.minimum=JMinimum(
#            self.BSqrt(minimizeReturn[0]), minimizeReturn[1],
#            minimizeReturn[2], minimizeReturn[3], 
#            minimizeReturn[4], minimizeReturn[5], minimizeReturn[6], 
#            maxiter, allvecs=allvecs, convergence=convJVal)


        
class PrecondStaticObsJTerm(PrecondJTerm, StaticObsJTerm):
    '''
    Preconditionned static observation JTerm subclass
    (classical 3D-Var context)
    '''

    #------------------------------------------------------
    #----| Init |------------------------------------------
    #------------------------------------------------------

    def __init__(self, obs, g,
                    x_bkg, B_sqrt, B_sqrtAdj, B_sqrtArgs=()): 
        
        super(PrecondStaticObsJTerm, self).__init__(obs, g)  

        if not (callable(B_sqrt) and callable(B_sqrtAdj)):
            raise self.PrecondStaticObsJTermError("B_sqrt[Adj] <function>")
        if not (isinstance(B_sqrtArgs, tuple)):
            raise self.PrecondStaticObsJTermError("B_sqrtArgs <tuple>")
        self.B_sqrt=B_sqrt
        self.B_sqrtAdj=B_sqrtAdj
        self.B_sqrtArgs=B_sqrtArgs
    
        self._xValidate(x_bkg)
        self.x_bkg=x_bkg

        self.isMinimized=False
        


#=====================================================================
#---------------------------------------------------------------------
#=====================================================================

class PrecondTWObsJTerm(PrecondJTerm, TWObsJTerm):
    """
    Preconditionned time window observations JTerm subclass
    (classical 4D-Var context)

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

    """
    
    class PrecondTWObsJTermError(Exception):
        pass

    #------------------------------------------------------
    #----| Init |------------------------------------------
    #------------------------------------------------------

    def __init__(self, obs, nlModel, tlm, 
                    x_bkg, B_sqrt, B_sqrtAdj, B_sqrtArgs=()):

        super(PrecondTWObsJTerm, self).__init__(obs, nlModel, tlm)  

        if not (callable(B_sqrt) and callable(B_sqrtAdj)):
            raise self.PrecondTWObsJTermError("B_sqrt[Adj] <function>")
        if not (isinstance(B_sqrtArgs, tuple)):
            raise self.PrecondTWObsJTermError("B_sqrtArgs <tuple>")
        self.B_sqrt=B_sqrt
        self.B_sqrtAdj=B_sqrtAdj
        self.B_sqrtArgs=B_sqrtArgs
    
        self._xValidate(x_bkg)
        self.x_bkg=x_bkg

        self.isMinimized=False


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
    from pseudoSpec1D import PeriodicGrid 
    
    Ntrc=100
    L=300.
    g=PeriodicGrid(Ntrc, L)

    #----| Preconditionning |-----------
    Lc=10.
    sig=1.
    corr=fCorr_isoHomo(g, Lc)
    rCTilde_sqrt=rCTilde_sqrt_isoHomo(g, corr)
    var=sig*np.ones(g.N)
    B_sqrtArgs=(var, rCTilde_sqrt)
    xi0=np.zeros(g.N)

    x0_bkg=np.zeros(g.N)

    testStaticObs=True
    testTWObs=False

    if  testStaticObs:
        coord=np.zeros(1)
        value=np.zeros(1)
        value[0]=1.
        obs=StaticObs(coord, value, obsOp=obsOp_Coord,
                            obsOpTLMAdj=obsOp_Coord_Adj)
        J=PrecondStaticObsJTerm(obs, g, x0_bkg, B_sqrt_op,
                                        B_sqrt_op_Adj, B_sqrtArgs)
        J.minimize(x0_bkg)

        plt.plot(g.x, J.analysis)
        J.obs.plot(g)
        plt.show()

    if testTWObs:
        
        kdvParam=kdv.Param(g, beta=1., gamma=-1.)
        tInt=15.
    
        model=kdv.kdvLauncher(kdvParam, maxA=4.)
        tlm=kdv.kdvTLMLauncher(kdvParam)
        
        base=kdv.rndSpecVec(g, Ntrc=10,  amp=0.4)
        soliton=kdv.soliton(g.x, 0., amp=1.5, beta=1., gamma=-1)
        longWave=0.8*(kdv.gauss(g.x, 40., 20. )
                        -0.5*kdv.gauss(g.x, -20., 14. ))
    
        x0_truth=soliton
        x_truth=model.integrate(x0_truth, tInt)
    
        x_bkg=model.integrate(x0_bkg, tInt)
    
        
        nObsTime=9
        nPosObs=50
        d_Obs={}
        for i in xrange(nObsTime):
            t=tInt*(i+1)/nObsTime
            obsPos=np.linspace(-g.L/2., g.L/2.-g.dx, nPosObs)
            obsValues=obsOp_Coord(x_truth.whereTime(t), g, obsPos)
            d_Obs[t]=StaticObs(obsPos, obsValues, obsOp_Coord,
                                obsOp_Coord_Adj)
        timeObs=TimeWindowObs(d_Obs)
    
    
        J=PrecondTWObsJTerm(timeObs, model, tlm, 
                            x0_bkg, B_sqrt_op, B_sqrt_op_Adj, B_sqrtArgs) 
    
    
        J.minimize(x0_bkg)
        x0_a=J.analysis
        x_a=model.integrate(x0_a,  tInt)
    
        nSubRow=3
        nSubLine=timeObs.nTimes/nSubRow+1
        if timeObs.nTimes%nSubRow: nSubLine+=1
        plt.figure(figsize=(12.,12.))
        i=0
        for t in timeObs.times:
            i+=1
            sub=plt.subplot(nSubLine, nSubRow, nSubRow+i)
            sub.plot(g.x, x_truth.whereTime(t), 'g')
            sub.plot(timeObs[t].interpolate(g), timeObs[t].values, 'go')
            sub.plot(g.x, x_bkg.whereTime(t), 'b')
            sub.plot(timeObs[t].interpolate(g), 
                        x_bkg.whereTime(t)[g.pos2Idx(timeObs[t].coord)], 
                        'bo')
            sub.set_title("$t=%f$"%t)
            if i==timeObs.nTimes:
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
