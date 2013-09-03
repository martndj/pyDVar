import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sciOpt

def costFunc(x):
    return np.dot(x,x)+3.

def gradF(x):
    return 2*x

x=np.ones(5)
x_a, out=sciOpt.fmin_bfgs(costFunc, x, fprime=gradF,  
                            retall=True, maxiter=1000)

