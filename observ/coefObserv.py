import numpy as np
from pseudoSpec1D import Grid
import dVar

#-----------------------------------------------------------
#----| Correlation coefficient |----------------------------
#-----------------------------------------------------------

def corrCoef(grid, v, obs, bkg):
    if not isinstance(grid, Grid):
        raise TypeError("grid <pseudoSpec>")
    if not isinstance(obs, dVar.StaticObs):
        raise TypeError('obs <StaticObs>')
    if ((not isinstance(v, np.ndarray)) or
        (not isinstance(bkg, np.ndarray))):
        raise TypeError('v, bkg <numpy.ndarray>')
    departure=obs.values-obs.modelEquivalent(bkg, grid)
    num=obs.prosca(obs.modelEquivalent(v, grid), departure)
    denom=(obs.norm(obs.modelEquivalent(v, grid))*
            obs.norm(departure))
    return num/denom


#-----------------------------------------------------------
#----| Observability evolution XP |-------------------------
#-----------------------------------------------------------

def evolutionXP(grid, natureRun, nObs, obsSig, coeffTimes, 
                NRealisations=100):

    def oneRealisation(grid, natureRun, nObs, obsSig, coeffTimes):
        d_coeff={}
        x_bkg=grid.zeros() 
        for t in coeffTimes:
            v=natureRun.whereTime(t)
            if obsSig==0:
                RInv=100.
                obsRef=v
        
            else:
                RInv=obsSig**(-2)
                obsRef=dVar.degrad(v, 0., obsSig, seed=None)
            
            coord=np.array(dVar.rndSampling(grid, nObs, precision=2))
            values=dVar.obsOp_Coord(obsRef, grid, coord)
            obs=dVar.StaticObs(coord, values, metric=RInv,
                                obsOp=dVar.obsOp_Coord, 
                                obsOpTLMAdj=dVar.obsOp_Coord_Adj)
            
            d_coeff[t]=corrCoef(grid, v, obs, x_bkg)
        return d_coeff
    #---------------------------------------------
    def rearrangeCoeff(listCoeff, NRealisations):
        d_coeff={}
        for t in listCoeff[0].keys():
            d_coeff[t]=[]
            for i in xrange(NRealisations):
                d_coeff[t].append(l_coeff[i][t])
        return d_coeff
    #---------------------------------------------


    d_mean_var={}
    l_coeff=[]
    for i in xrange(NRealisations):
        l_coeff.append(oneRealisation(grid, natureRun, 
                                        nObs, obsSig, coeffTimes))
    d_coeff=rearrangeCoeff(l_coeff, NRealisations)
    
    for t in coeffTimes:
        d_mean_var[t]=[np.mean(d_coeff[t]),np.var(d_coeff[t])]
    return d_mean_var




