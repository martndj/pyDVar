import numpy as np
from observationOp import *
from modelCovariances import *
from observationOp import *

def costFunc(xi, x_b, var, B_sqrt_op, B_sqrt_op_T,
                H, H_T, argsH, obs, R_inv, rCTilde_sqrt):

    J_xi=0.5*np.dot(xi, xi)
    d=departure(xi, x_b, var, B_sqrt_op, H, argsH, obs, rCTilde_sqrt)
    J_o=0.5*np.dot(d,np.dot(R_inv,d))
    return J_xi+J_o

def gradCostFunc(xi, x_b, var, B_sqrt_op, B_sqrt_op_T,
                    H, H_T, argsH, obs, R_inv, rCTilde_sqrt):

    d=departure(xi, x_b, var, B_sqrt_op, H,  argsH, obs, rCTilde_sqrt)
    gradJ_o=B_sqrt_op_T(H_T(np.dot(R_inv, d), *argsH), 
                            var, rCTilde_sqrt)
    return xi+gradJ_o

def gradTest(costFunc, gradCostFunc, xi, args):
    
    maxPow=-14
    J0=costFunc(xi, *args)
    gradJ0=gradCostFunc(xi, *args)
    result={}
    for power in xrange(-1, maxPow, -1):
        eps=10.**(power)
        Jeps=costFunc(xi-eps*gradJ0, *args)
        
        n2GradJ0=np.dot(gradJ0, gradJ0)
        res=((J0-Jeps)/(eps*n2GradJ0))
        result[power]=[Jeps,n2GradJ0, res]


    return result


def printGradTest(result):
    print("----| Gradient test |------------------")
    for i in  (np.sort(result.keys())[::-1]):
        print("%4d %+25.15f"%(i, result[i][2]))

if __name__=='__main__':

    def func1(x):
        return np.dot(x,x)
    def func2(x):
        return np.dot(x,x)+9999999999999999999
    
    def grad(x):    
        return 2.*x
    
    x=np.zeros(3)
    x[0]=1.
    printGradTest(gradTest(func1, grad, x))
    printGradTest(gradTest(func2, grad, x))
        
