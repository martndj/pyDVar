import numpy as np
import scipy.optimize as sciOpt
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
import pickle
#from fmin_bfgs import fmin_bfgs

def norm(x):
    return np.sqrt(np.dot(x,x))


class JMinimum(object):
    """
    Minimisation result of a JTerm
    """
    #------------------------------------------------------
    #----| Init |------------------------------------------
    #------------------------------------------------------
    def __init__(self, xOpt, fOpt, gOpt, BOpt,
                    fCalls, gCalls, 
                    warnFlag, maxiter, 
                    allvecs=None, convergence=None):
        if ((not convergence==None) and (allvecs==None)):
            raise Exception()
        self.xOpt=xOpt
        self.fOpt=fOpt
        self.gOpt=gOpt
        self.BOpt=BOpt
        self.fCalls=fCalls
        self.gCalls=gCalls
        self.warnFlag=warnFlag
        self.maxiter=maxiter
        self.allvecs=allvecs
        self.convergence=convergence

        self.gOptNorm=np.sqrt(np.dot(self.gOpt,self.gOpt))

    #------------------------------------------------------

    def dump(self, fun):
        pickle.dump(self.xOpt, fun)
        pickle.dump(self.fOpt, fun)
        pickle.dump(self.gOpt, fun)
        pickle.dump(self.BOpt, fun)
        pickle.dump(self.fCalls, fun)
        pickle.dump(self.gCalls, fun)
        pickle.dump(self.warnFlag, fun)
        pickle.dump(self.maxiter, fun)
        pickle.dump(self.allvecs, fun)
        pickle.dump(self.convergence, fun)
        
#---------------------------------------------------------------------

def loadJMinimum(fun):
    
    xOpt=pickle.load(fun)
    fOpt=pickle.load(fun)
    gOpt=pickle.load(fun)
    BOpt=pickle.load(fun)
    fCalls=pickle.load(fun)
    gCalls=pickle.load(fun)
    warnFlag=pickle.load(fun)
    maxiter=pickle.load(fun)
    allvecs=pickle.load(fun)
    convergence=pickle.load(fun)
    jMin=JMinimum(xOpt, fOpt, gOpt, BOpt,
                    fCalls, gCalls, 
                    warnFlag, maxiter, 
                    allvecs=allvecs, convergence=convergence)
    return jMin
    
#=====================================================================
#---------------------------------------------------------------------
#=====================================================================

class JTerm(object):
    """
    JTerm(costFunc, gradCostFunc, args=()) 

        costFunc, gradCostFunc(x, *args)

        <!> This is a master class not meant to be instantiated, only
            subclasses should.

        JTerms (and sub classes) can be summed :  JSum=((J1+J2)+J3)+...
        JTerms (and sub classes) can be scaled :  JMult=J1*.5
    """

    class JTermError(Exception):
        pass

    #------------------------------------------------------
    #----| Init |------------------------------------------
    #------------------------------------------------------

    def __init__(self, costFunc, gradCostFunc, 
                    args=(), maxGradNorm=None):
        
        if not (callable(costFunc) and callable(gradCostFunc)):
            raise self.JTermError("costFunc, gardCostFunc <function>")

        self._costFunc=costFunc
        self._gradCostFunc=gradCostFunc

        if not (isinstance(maxGradNorm, float) or maxGradNorm==None):
            raise self.JTermError("maxGradNorm <None|float>")
        self.maxGradNorm=maxGradNorm 
        self.args=args

        self.isMinimized=False
        self.retall=False

    #------------------------------------------------------
    #----| Public methods |--------------------------------
    #------------------------------------------------------

    def J(self, x):
        return self._costFunc(x,*self.args) 

    #------------------------------------------------------

    def gradJ(self, x):
        if self.maxGradNorm==None:
            return self._gradCostFunc(x, *self.args)
        elif isinstance(self.maxGradNorm, float):
            grad=self._gradCostFunc(x, *self.args)
            normGrad=norm(grad)
            if np.isnan(normGrad):
                grad=np.zeros(self.modelGrid.N)
            elif normGrad>self.maxGradNorm:
                grad=(grad/normGrad)*(self.maxGradNorm)
            return grad

    def normGradJ(self, x):
        return norm(self.gradJ(x))

    #------------------------------------------------------


    def minimize(self, x_fGuess, maxiter=50, retall=True,
                    testGrad=True, finalTestGrad=False, convergence=True, 
                    testGradMinPow=-1, testGradMaxPow=-14):


        self.retall=retall
        self.minimizer=sciOpt.fmin_bfgs
        #self.minimizer=fmin_bfgs

        if x_fGuess.dtype<>'float64':
            raise self.JTermError("x_fGuess.dtype=='float64'")
        #----| Gradient test |--------------------
        if testGrad:
            self.testGradInit=self.gradTest(x_fGuess,
                                powRange=[testGradMinPow, testGradMaxPow])

        #----| Minimizing |-----------------------
        minimizeReturn=self.minimizer(self.J, x_fGuess, args=self.args,
                                        fprime=self.gradJ,  
                                        maxiter=maxiter, retall=self.retall,
                                        full_output=True)

        self.createMinimum(minimizeReturn, maxiter, convergence=convergence)
        self.createAnalysis()

        #----| Final Gradient test |--------------
        if finalTestGrad:
            if self.minimum.warnFlag==2:
                print("Gradient and/or function calls not changing:")
                print(" not performing final gradient test.")
                self.testGradFinal=None
            else:
                self.testGradFinal=self.gradTest(self.minimum.xOpt,
                                powRange=[testGradMinPow, testGradMaxPow])


    #-----------------------------------------------------

    def createMinimum(self, minimizeReturn, maxiter, convergence=True):
        if self.retall:
            allvecs=minimizeReturn[7]
            if convergence:
                convJVal=self._jAllVecs(allvecs)
        else:
            allvecs=None
            convJVal=None

        self.minimum=JMinimum(
            minimizeReturn[0], minimizeReturn[1], minimizeReturn[2],
            minimizeReturn[3], minimizeReturn[4], minimizeReturn[5],
            minimizeReturn[6], maxiter,
            allvecs=allvecs, convergence=convJVal)
        self.isMinimized=True

    #-----------------------------------------------------

    def createAnalysis(self):
        if np.any(np.isnan(self.minimum.gOpt)):
            if not self.retall:
                raise jTermError(
                    "No previous state to fall back: try minimize with retall=True")
            nIters=len(self.minimum.allvecs)
            self.analysis=self.minimum.allvecs[nIters-2]
        else:
            self.analysis=self.minimum.xOpt
                
    #------------------------------------------------------
    #----| Gradient test |---------------------------------

    def _gTest(self, x, J0, gradJ0, n2GradJ0, powRange):
        test={}
        for power in xrange(powRange[0],powRange[1], -1):
            eps=10.**(power)
            Jeps=self._costFunc(x-eps*gradJ0)
            
            res=((J0-Jeps)/(eps*n2GradJ0))
            test[power]=[Jeps, res]
        return test

    def gradTestString(self, J0, n2GradJ0, test):
        s="----| Gradient test |------------------\n"
        s+="  J0      =%+25.15e\n"%J0
        s+=" |grad|^2 =%+25.15e\n"%n2GradJ0
        for i in  (np.sort(test.keys())[::-1]):
            s+="%4d %+25.15e  %+25.15e\n"%(i, test[i][0], test[i][1])
        return s

    def gradTest(self, x, powRange=[-1,-14], 
                    findFirst9=False,
                    output=True):
        J0=self._costFunc(x)
        gradJ0=self._gradCostFunc(x)
        n2GradJ0=np.dot(gradJ0, gradJ0)

        powRangeTrial=powRange[:]
        test=self._gTest(x, J0, gradJ0, n2GradJ0, powRangeTrial)

        if output:  print(self.gradTestString(J0, n2GradJ0, test))

        if findFirst9:
            powerMin=powRangeTrial[1]
            for power in sorted(test.keys()):
                digits=str(test[power][1])
                if digits=='nan':
                    break
                fDigit,mantissa=digits.split('.')
                if (int(fDigit)==0 and int(mantissa[0])==9):
                    powerMin=power
                
            if -powRangeTrial[1]+powerMin<7:
                print("  <!> not enough powers left to conclude test")
                powRangeTrial[0]=powerMin
                powRangeTrial[1]=powerMin-7
                print("      redoing on %s"%powRangeTrial)
                test=self._gTest(x, J0, gradJ0, n2GradJ0, powRangeTrial)
                if output:  print(self.gradTestString(J0, n2GradJ0, test))
            else:
                testTmp={}
                for power in sorted(test.keys()):
                    if power <= powerMin:
                        testTmp[power]=test[power]
                    else:
                        break
                test=testTmp



        return (J0, n2GradJ0, test)

    #------------------------------------------------------

    def plotCostFunc(self, x, epsMin=-1., epsMax=1., nDx=10, axe=None):
        axe=self._checkAxe(axe) 
        
        dx=(epsMax-epsMin)/nDx
        J=np.zeros(nDx)
        xPlusDx=np.linspace(epsMin, epsMax, nDx)
        grad=self.gradJ(x)
        for i in xrange(nDx):
            alpha=epsMin+i*dx
            J[i]=self.J(x+alpha*grad)
        axe.plot(xPlusDx,J, '^-')
        return xPlusDx, J
    #------------------------------------------------------
    #----| Private methods |-------------------------------
    #------------------------------------------------------
    

    def _jAllVecs(self, allvecs):
        convJVal=[]
        for i in xrange(len(allvecs)):
            convJVal.append(self.J(allvecs[i]))
        return convJVal

    #------------------------------------------------------
    #----| Classical overloads |----------------------------
    #-------------------------------------------------------

    def __str__(self):
        output="////| jTerm |//////////////////////////////////////////////"
        if self.isMinimized:
            if self.minimum.warnFlag:
                output+="\n <!> Warning %d <!>"%self.minimum.warnFlag
            output+="\n function value=%f"%self.minimum.fOpt
            output+="\n gradient norm=%f"%self.minimum.gOptNorm
            output+="\n function calls=%d"%self.minimum.fCalls
            output+="\n gradient calls=%d"%self.minimum.gCalls
        else:
            output+="\n Not minimized"
        output+="\n///////////////////////////////////////////////////////////\n"
        return output

    #-------------------------------------------------------

    def __add__(self, J2):
        if not isinstance(J2, JTerm):
            raise self.JTermError("J1,J2 <JTerm>")

        def CFSum(x):
            return self.J(x)+J2.J(x)
        def gradCFSum(x):
            return self.gradJ(x)+J2.gradJ(x)

        if (self.maxGradNorm==None and J2.maxGradNorm==None):
            maxGradNorm=None
        elif self.maxGradNorm==None:
            maxGradNorm=J2.maxGradNorm
        elif J2.maxGradNorm==None:
            maxGradNorm=self.maxGradNorm


        JSum=JTerm(CFSum, gradCFSum, maxGradNorm=maxGradNorm)
        return JSum

    #------------------------------------------------------

    def __mul__(self, scalar):
        if not isinstance(scalar,float):
            raise self.JTermError("scalar <float>")

        def CFMult(x):
            return self.J(x)*scalar
        def gradCFMult(x):
            return self.gradJ(x)*scalar

        JMult=JTerm(CFMult, gradCFMult)
        return JMult
            
    
    #-------------------------------------------------------
    #----| Private plotting methods |-----------------------
    #-------------------------------------------------------

    def _checkAxe(self, axe):
        if axe==None:
            axe=plt.subplot(111)
        elif not (isinstance(axe,(Axes, GridSpec))):
            raise self.JTermError(
            "axe < matplotlib.axes.Axes | matplotlib.gridspec.GridSpec >")
        return axe

#=====================================================================
#---------------------------------------------------------------------
#=====================================================================

class TrivialJTerm(JTerm):
    
    class TrivialJTermError(Exception):
        pass


    #------------------------------------------------------
    #----| Init |------------------------------------------
    #------------------------------------------------------

    def __init__(self, maxGradNorm=None):
        if not (isinstance(maxGradNorm, float) or maxGradNorm==None):
            raise self.TrivialJTermError("maxGradNorm <None|float>")
        self.maxGradNorm=maxGradNorm 
        self.args=()
        self.isMinimized=False

    #------------------------------------------------------
    #----| Private methods |-------------------------------
    #------------------------------------------------------

    def __xValidate(self, x):
        if not isinstance(x, np.ndarray):
            raise self.TrivialJTermError("x <numpy.array>")
        if x.ndim<>1:
            raise TWObsJTermError("x.ndim==1")
    #------------------------------------------------------
    #----| Public methods |--------------------------------
    #------------------------------------------------------

    def _costFunc(self, x):
        self.__xValidate(x)
        return 0.5*np.dot(x,x) 

    #------------------------------------------------------

    def _gradCostFunc(self, x):
        self.__xValidate(x)
        return x

#=====================================================================
#---------------------------------------------------------------------
#=====================================================================
