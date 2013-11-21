from observations import StaticObs, TimeWindowObs
from jTerm import JTerm, JMinimum, norm
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

    def _costFunc(self, xi, normalize=False): 
        self._xValidate(xi)
        x=self.BSqrt(xi)
        return super(PrecondJTerm, self)._costFunc(x)+0.5*np.dot(xi,xi)

    #------------------------------------------------------

    def _gradCostFunc(self, xi, normalize=False):
        self._xValidate(xi)
        x=self.BSqrt(xi)

        dx0=super(PrecondJTerm, self)._gradCostFunc(x)
        grad= self.B_sqrtAdj(dx0,*self.B_sqrtArgs)+xi
        return grad
    
    #------------------------------------------------------
    #----| Public methods |--------------------------------
    #------------------------------------------------------
    

    def gradJ(self, xi):
        if self.maxGradNorm==None:
            return self._gradCostFunc(xi, *self.args)
        elif isinstance(self.maxGradNorm, float):
            grad=self._gradCostFunc(xi, *self.args)
            # norm compared in physical space
            # BSqrt being linear
            normGrad=norm(self.BSqrt(grad))
            if normGrad>self.maxGradNorm:
                grad=(grad/normGrad)*(self.maxGradNorm)
            return grad
    
    #------------------------------------------------------
    
    def BSqrt(self, xi):
        return self.B_sqrt(xi, *self.B_sqrtArgs)+self.x_bkg

    #------------------------------------------------------
    
    def minimize(self, maxiter=50, retall=True,
                    testGrad=True, convergence=True, 
                    testGradMinPow=-1, testGradMaxPow=-14):
        super(PrecondJTerm, self).minimize(
                    self.x_bkg, maxiter=maxiter, retall=retall,
                    testGrad=testGrad, convergence=convergence, 
                    testGradMinPow=testGradMinPow,
                    testGradMaxPow=testGradMaxPow)
        self.analysis=self.BSqrt(self.minimum.xOpt)

        
#=====================================================================
#---------------------------------------------------------------------
#=====================================================================

class PrecondStaticObsJTerm(PrecondJTerm, StaticObsJTerm):
    '''
    Preconditionned static observation JTerm subclass
    (classical 3D-Var context)
    '''

    #------------------------------------------------------
    #----| Init |------------------------------------------
    #------------------------------------------------------

    def __init__(self, obs, g,
                    x_bkg, B_sqrt, B_sqrtAdj, B_sqrtArgs=(),
                    maxGradNorm=None): 
        
        super(PrecondStaticObsJTerm, self).__init__(obs, g, 
                                                maxGradNorm=maxGradNorm)  

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
                    x_bkg, B_sqrt, B_sqrtAdj, B_sqrtArgs=(),
                    t0=0., tFinal=None, maxGradNorm=None):

        super(PrecondTWObsJTerm, self).__init__(obs, nlModel, tlm, 
                                            t0=t0, tFinal=tFinal,
                                            maxGradNorm=maxGradNorm)  

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

#if __name__=='__main__':
