import numpy as np
import scipy.optimize as sciOpt


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

    def __init__(self, costFunc, gradCostFunc, args=()):
        
        if not (callable(costFunc) and callable(gradCostFunc)):
            raise self.JTermError("costFunc, gardCostFunc <function>")

        self.__costFunc=costFunc
        self.__gradCostFunc=gradCostFunc

        if not isinstance(args,tuple):
            raise self.JTermError("args <tuple>")
        self.args=args

    #------------------------------------------------------
    #----| Public methods |--------------------------------
    #------------------------------------------------------

    def J(self, x):
        return self.__costFunc(x,*self.args) 

    #------------------------------------------------------

    def gradJ(self, x):
        return self.__gradCostFunc(x, *self.args)

    #------------------------------------------------------

    def minimize(self, x_fGuess, 
                    maxiter=50, retall=True,
                    testGrad=True, testGradMinPow=-1, testGradMaxPow=-14):

        if x_fGuess.dtype<>'float64':
            raise JTermError("x_fGuess.dtype=='float64'")
        #----| Gradient test |--------------------
        if testGrad:
            self.gradTest(x_fGuess,
                            powRange=[testGradMinPow, testGradMaxPow])

        #----| Minimizing |-----------------------
        self.minimize=sciOpt.fmin_bfgs
        minimizeReturn=self.minimize(self.J, x_fGuess, args=self.args,
                                        fprime=self.gradJ,  
                                        maxiter=maxiter,
                                        retall=retall,
                                        full_output=True)
        self.analysis=minimizeReturn[0]
        self.fOpt=minimizeReturn[1]
        self.gOpt=minimizeReturn[2]
        self.gOptNorm=np.sqrt(np.dot(self.gOpt,self.gOpt))
        self.hInvOpt=minimizeReturn[3]
        self.fCalls=minimizeReturn[4]
        self.gCalls=minimizeReturn[5]
        self.warnFlag=minimizeReturn[6]
        if retall:
            self.allvecs=minimizeReturn[7]

        #----| Final Gradient test |--------------
        if testGrad:
            if self.warnFlag==2:
                print("Gradient and/or function calls not changing:")
                print(" not performing final gradient test.")
            else:
                self.gradTest(self.analysis,
                            powRange=[testGradMinPow, testGradMaxPow])



    #------------------------------------------------------

    def __add__(self, J2):
        if not isinstance(J2, JTerm):
            raise self.JTermError("J1,J2 <JTerm>")

        def CFSum(x):
            return self.J(x)+J2.J(x)
        def gradCFSum(x):
            return self.gradJ(x)+J2.gradJ(x)

        JSum=JTerm(CFSum, gradCFSum)
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
    #------------------------------------------------------

    def gradTest(self, x, output=True, powRange=[-1,-14]):
        J0=self.J(x)
        gradJ0=self.gradJ(x)
        test={}
        for power in xrange(powRange[0],powRange[1], -1):
            eps=10.**(power)
            Jeps=self.J(x-eps*gradJ0)
            
            n2GradJ0=np.dot(gradJ0, gradJ0)
            res=((J0-Jeps)/(eps*n2GradJ0))
            test[power]=[Jeps, res]

        if output:
            print("----| Gradient test |------------------")
            print("  J0      =%+25.15f"%J0)
            print(" |grad|^2 =%+25.15f"%n2GradJ0)
            for i in  (np.sort(test.keys())[::-1]):
                print("%4d %+25.15f  %+25.15f"%(i, test[i][0], test[i][1]))

        return (J0, n2GradJ0, test)


#=====================================================================
#---------------------------------------------------------------------
#=====================================================================

class TrivialJTerm(JTerm):
    
    class TrivialJTermError(Exception):
        pass


    #------------------------------------------------------
    #----| Init |------------------------------------------
    #------------------------------------------------------

    def __init__(self):

        self.args=()

    #------------------------------------------------------
    #----| Private methods |-------------------------------
    #------------------------------------------------------

    def __xValidate(self, x):
        if not isinstance(x, np.ndarray):
            raise TWObsJTermError("x <numpy.array>")
        if x.ndim<>1:
            raise TWObsJTermError("x.ndim==1")
    #------------------------------------------------------
    #----| Public methods |--------------------------------
    #------------------------------------------------------

    def J(self, x):
        self.__xValidate(x)
        return 0.5*np.dot(x,x) 

    #------------------------------------------------------

    def gradJ(self, x):
        self.__xValidate(x)
        return x

#=====================================================================
#---------------------------------------------------------------------
#=====================================================================


if __name__=='__main__':

    J1=TrivialJTerm()
    x=np.ones(10)
    
    print("====| Simple cost function |=======")
    print("First Guess:")
    print(x)
    J1.minimize(x)
    print("Analysis:")
    print(J1.analysis)


    J2=TrivialJTerm()
    print("\n\n====| Two terms cost function |====")
    print("First Guess:")
    print(x)
    JSum=J1+(J2*.5)
    JSum.minimize(x)
    print("Analysis:")
    print(JSum.analysis)

