import numpy as np
import scipy.optimize as sciOpt


class JTerm(object):
    """
    JTerm(costFunc, gradCostFunc, args=(), 
            maxiter=100, retall=True, testAdj=False,
            testGrad=True, testGradMinPow=-1, testGradMaxPow=-14)

        costFunc, gradCostFunc(x, *args)

        JTerms can be summed :  JSum=((J1+J2)+J3)+...
    """

    class JTermError(Exception):
        pass

    #------------------------------------------------------
    #----| Init |------------------------------------------
    #------------------------------------------------------

    def __init__(self, costFunc, gradCostFunc, args=(), 
                    maxiter=100, retall=True, testAdj=False,
                    testGrad=True, testGradMinPow=-1, testGradMaxPow=-14):
        
        if not (callable(costFunc) and callable(gradCostFunc)):
            raise self.JTermError("costFunc, gardCostFunc <function>")

        self.__costFunc=costFunc
        self.__gradCostFunc=gradCostFunc

        if not isinstance(args,tuple):
            raise self.JTermError("args <tuple>")
        self.args=args

        self.__configure(maxiter, retall, testAdj, testGrad, 
                            testGradMinPow, testGradMaxPow)

    #------------------------------------------------------
    #----| Private methods |-------------------------------
    #------------------------------------------------------

    def __configure(self, maxiter, retall, testAdj, testGrad, 
                            testGradMinPow, testGradMaxPow):
        self.maxiter=maxiter
        self.retall=retall
        self.testAdj=testAdj
        self.testGrad=testGrad
        self.testGradMinPow=-1
        self.testGradMaxPow=-14
    #------------------------------------------------------
    #----| Public methods |--------------------------------
    #------------------------------------------------------

    def J(self, x):
        return self.__costFunc(x,*self.args) 

    #------------------------------------------------------

    def gradJ(self, x):
        return self.__gradCostFunc(x, *self.args)

    #------------------------------------------------------

    def minimize(self, x_fGuess):
        
        #----| Gradient test |--------------------
        if self.testGrad:
            self.gradTest(x_fGuess)

        #----| Minimizing |-----------------------
        self.minimize=sciOpt.fmin_bfgs
        minimizeReturn=self.minimize(self.J, x_fGuess, args=self.args,
                                        fprime=self.gradJ,  
                                        maxiter=self.maxiter,
                                        retall=self.retall,
                                        full_output=True)
        self.x_a=minimizeReturn[0]
        self.fOpt=minimizeReturn[1]
        self.gOpt=minimizeReturn[2]
        self.hInvOpt=minimizeReturn[3]
        self.fCalls=minimizeReturn[4]
        self.gCalls=minimizeReturn[5]
        self.warnFlag=minimizeReturn[6]
        if self.retall:
            self.allvecs=minimizeReturn[7]

        #----| Final Gradient test |--------------
        if self.testGrad:
            self.gradTest(self.x_a)

        #----| Analysis |-------------------------
        self.analysis=self.x_a


    #------------------------------------------------------

    def __add__(self, J2):
        if not isinstance(J2, JTerm):
            raise self.JTermError("J1,J2 <JTerm>")

        def CFSum(x):
            return self.J(x)+J2.J(x)

        def gradCFSum(x):
            return self.gradJ(x)+J2.gradJ(x)

        JSum=JTerm(CFSum, gradCFSum,
                    maxiter=max(self.maxiter, J2.maxiter),
                    retall=(self.retall or J2.retall),
                    testAdj=(self.testAdj or J2.testAdj),
                    testGrad=(self.testGrad or J2.testGrad),
                    testGradMinPow=max(self.testGradMinPow,
                                        J2.testGradMinPow),
                    testGradMaxPow=min(self.testGradMaxPow,
                                        J2.testGradMaxPow),
                    )
        return JSum


    #------------------------------------------------------

    def gradTest(self, x, output=True):
        J0=self.J(x)
        gradJ0=self.gradJ(x)
        test={}
        for power in xrange(self.testGradMinPow, self.testGradMaxPow, -1):
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

class PrecondJTerm(JTerm):
    
    class PrecondJTermError(Exception):
        pass


    #------------------------------------------------------
    #----| Init |------------------------------------------
    #------------------------------------------------------

    def __init__(self, maxiter=100, retall=True, testAdj=False,
                    testGrad=True, testGradMinPow=-1, testGradMaxPow=-14):

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
        return 0.5*np.dot(x,x) 

    #------------------------------------------------------

    def gradJ(self, x):
        return x

#=====================================================================
#---------------------------------------------------------------------
#=====================================================================


if __name__=='__main__':

    J1=PrecondJTerm()
    x=np.ones(10)
    
    print("====| Simple cost function |=======")
    print("First Guess:")
    print(x)
    J1.minimize(x)
    print("Analysis:")
    print(J1.x_a)


    J2=PrecondJTerm()
    print("\n\n====| Two terms cost function |====")
    print("First Guess:")
    print(x)
    JSum=J1+J2
    JSum.minimize(x)
    print("Analysis:")
    print(JSum.x_a)

