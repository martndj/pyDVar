import numpy as np
from dataAssLib import gradTest


def func(x):
    return np.dot(x,x)+9999999999999999999

def grad(x):    
    return 2.*x

x=np.zeros(3)
x[0]=1.
gradTest(func, grad, x)

    
