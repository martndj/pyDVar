import numpy as np
from pseudoSpec1D import SpectralGrid
import scipy.optimize as sciOpt

from costFunction import *

class DataAss(object):
    """
    """
    class DataAssError(Exception):
        pass

    #------------------------------------------------------
    #----| Init |------------------------------------------
    #------------------------------------------------------

    def __init__(self, s_grid, s_bkg, s_var, B_sqrt, B_sqrt_T, 
                    H, H_T, argsH, obs, R_inv, rCTilde_sqrt, maxiter=100):

        if not (isinstance(s_grid, SpectralGrid)):
            raise self.DataAssError("s_grid <SpectralGrid>")
        self.grid=s_grid

        if not (isinstance(s_bkg,np.ndarray)):
            raise self.DataAssError("s_bkg <numpy.ndarray>")
        if not (s_bkg.shape==(self.grid.N,)):
            raise self.DataAssError("s_bkg.shape<>self.grid.N")
        self.s_bkg=s_bkg

        if not (isinstance(s_var,np.ndarray)):
            raise self.DataAssError("s_var <numpy.ndarray>")
        if not (s_var.shape==(self.grid.N,)):
            raise self.DataAssError("s_var.shape<>self.grid.N")
        self.s_var=s_var

        if (not callable(B_sqrt)) or (not callable(B_sqrt_T)):
            raise self.DataAssError("B_sqrt[_T] <functions>")
        self.B_sqrt=B_sqrt
        self.B_sqrt_T=B_sqrt_T

        if (not callable(H)) or (not callable(H_T)):
            raise self.DataAssError("H[_T] <functions>")
        self.H=H
        self.H_T=H_T
        self.argsH=argsH

        if not (isinstance(obs,np.ndarray)):
            raise self.DataAssError("obs <numpy.ndarray>")
        self.obs=obs

        if not (isinstance(R_inv,np.ndarray)):
            raise self.DataAssError("R_inv <numpy.ndarray>")
        if not (R_inv.shape==self.obs.shape,):
            raise self.DataAssError("R_inv.shape<>self.obs.shape; R_inv is the diagonal of the full R inverse matrix.")
        self.R_inv=R_inv

        if (not isinstance(rCTilde_sqrt, np.ndarray)):
            raise self.DataAssError("rCTilde_sqrt <numpy.ndarray>")
        self.rCTilde_sqrt=rCTilde_sqrt

        self.maxiter=maxiter


    #------------------------------------------------------
    #----| Public methods |--------------------------------
    #------------------------------------------------------


    def minimize(self, gradientTest=True):
        costFuncArgs=(self.s_bkg, self.s_var, self.B_sqrt, self.B_sqrt_T, 
                        self.H, self.H_T, self.argsH, self.obs, self.R_inv,
                        self.rCTilde_sqrt)
       
        xi=np.zeros(self.grid.N)

        #----| Gradient test |--------------------
        if gradientTest:
            printGradTest(gradTest(costFunc, gradCostFunc, xi, 
                                    *costFuncArgs))

        #----| Minimizing |-----------------------
        xi_a=sciOpt.fmin_bfgs(costFunc, xi, fprime=gradCostFunc,  
                                args=costFuncArgs, maxiter=self.maxiter)

        #----| Final Gradient test |--------------
        if gradientTest:
            printGradTest(gradTest(costFunc, gradCostFunc, xi_a, 
                                    *costFuncArgs))

        #----| Analysis |-------------------------
        self.analysis=(self.B_sqrt(xi_a, self.s_var, self.rCTilde_sqrt)
                        +self.s_bkg)
