import numpy as np
from pseudoSpec1D import SpectralGrid #, Trajectory

class jTerm(object):
    """
        Nuclear cost function term for assimilating an information type

        x_ prefix refer to model space
        y_ to assimilated information (observation) space
    """

    class jTermError(Exception):
        pass

    #------------------------------------------------------
    #----| Init |------------------------------------------
    #------------------------------------------------------

    def __init__(self, x_grid, 
                    y_coord, obsOp, obsArgs, obsTLAdj, infoMetric,
                    maxiter=100):
        
        if not (isinstance(grid, SpectralGrid)):
            raise self.jTermError("grid <pseudoSpec1D.SpectralGrid>")
        self.x_grid=x_grid

        if  isinstance(y_coord, dict):
            self.isDict=True
            for t in y_coord.keys():
                if not isinstance(y_coord[t], np.ndarray):
                    raise self.jTermError("
                y_coord <numpy.ndarray | dict {time:<numpy.ndarray>} >")
        elif isinstance(y_coord,  np.ndarray):
            self.isDict=False
        else:
            raise self.jTermError("
                y_coord <numpy.ndarray | dict {time:<numpy.ndarray>} >")

        self.y_coord=y_coord


            

        
        
