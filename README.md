1D-Var librairy (data assimilation lab)
=======================================

Martin Deshaies-Jacques

[deshaies.martin@sca.uqam.ca](mailto:deshaies.martin@sca.uqam.ca)

[www.science.martn.info](http://www.science.martn.info)

Use, share, remix!
...and criticize

Test-it with [pyfKdV](https://github.com/martndj/pyfKdV)


About Data Assimilation concepts
--------------------------------


 * [F. Bouttier, P., Courtier, Data assimilation concepts and methods, ECMWF Meteorological Training Course Lecture Series, March 1999](http://msi.ttu.ee/~elken/Assim_concepts.pdf)

Installation informations
-------------------------

### Python Dependencies
 * pseudoSpec1D (found in [pyKdV](https://github.com/martndj/pyfKdV))
 * Numpy
 * Matplotlib
 * Scipy

### Instructions
 * ./ refer to the root of dVar installation;
 * [something] means optional arguments;
 * \<something\> means you must replace it with what is appropriate to your environment.

 1. On Linux OS Debian/Ubuntu, you can install it running

        [sudo] apt-get install python python-matplotlib python-numpy python-scipy

 2. Add the path to the python module dVar to your PYTHONPATH environment variable, in Linux bash environment you can do it running.
 
        export PYTHONPATH=${PYTHONPATH}:<path to ./>

    Adding this export line to your startup script (.bashrc or .profile) is a way to do it.

 3. Link the data assimilation lab with a 1+1D model.  Here is an example using [pyfKdV](https://github.com/martndj/pyfKdV):

        git clone https://github.com/martndj/pyfKdV.git

    Follow the installation instruction, and then try this script-experiment:

        import numpy as np
        import matplotlib.pyplot as plt
        import pyKdV as kdv
        import dVar as dVar
        import random as rnd
        
        #----| Model definition |---------------
        #----| Grid configuration |---
        g=kdv.PeriodicGrid(110,300.)
        tInt=10.
        maxA=4.
            
        #----| KdV parameters |-------
        def gauss(x):
            x0=0.
            sig=5.
            return -0.1*np.exp(-((x-x0)**2)/(2*sig**2))
                
        kdvParam=kdv.Param(g, beta=1., gamma=-1, rho=gauss)
            
        model=kdv.kdvLauncher(kdvParam, maxA, dtMod=0.5)
        tlm=kdv.kdvTLMLauncher(kdvParam)
        
        #----| Nature run (truth) |-------------
        rndLFBase=kdv.rndFiltVec(g, Ntrc=g.Ntrc/4)
        soliton=kdv.soliton(g.x, 0., amp=3., beta=1., gamma=-1)
        
        x0_truth=rndLFBase+soliton
        
        #----| Observations |-------------------
        sigma=0.1
        x0_degrad=dVar.degrad(x0_truth, 0., 0.1)
        x_degrad=model.integrate(x0_degrad, tInt)
        x_truth=model.integrate(x0_truth, tInt)
        x_obs=x_truth
        
        nObsTime=3
        nPosObs=5
        R_inv=np.ones(nPosObs)*sigma**(-1)
        d_Obs={}
        for i in xrange(nObsTime):
            t=tInt*(i+1)/nObsTime
            obsPos=np.empty(nPosObs)
            for j in xrange(nPosObs):
                obsPos[j]=rnd.random()*g.L-g.L/2.
            obsValues=dVar.obsOp_Coord(x_obs.whereTime(t), g, obsPos)
            d_Obs[t]=dVar.StaticObs(obsPos, obsValues, 
                                obsOp=dVar.obsOp_Coord, 
                                obsOpTLMAdj=dVar.obsOp_Coord_Adj, 
                                metric=R_inv)
        timeObs=dVar.TimeWindowObs(d_Obs)
        
        #----| adding a background |------------
        x0_bkg=kdv.specFilt(rndLFBase, g, Ntrc=4)
        #----| preconditionning |---------------
        Lc=100.
        sigB=1.
        corr=dVar.fCorr_isoHomo(g, Lc)
        rCTilde_sqrt=dVar.rCTilde_sqrt_isoHomo(g, corr)
        var=sigB*np.ones(g.N)
        B_sqrtArgs=(var, rCTilde_sqrt)
        xi0=np.zeros(g.N)
        Jo=dVar.PrecondTWObsJTerm(timeObs, model, tlm, x0_bkg,
                                    dVar.B_sqrt_op, dVar.B_sqrt_op_Adj,
                                    B_sqrtArgs)
        Jb=dVar.TrivialJTerm()
        J2=Jo+Jb
        
        #----| Minimizing |---------------------
        print("\nMinimizing J2...")
        J2.minimize(np.zeros(g.N))
        
        #----| Integrating trajectories |-------
        x0_a=dVar.B_sqrt_op(J2.analysis, *B_sqrtArgs)
        x_a=model.integrate(x0_a, tInt)
        x_bkg=model.integrate(x0_bkg, tInt)
        
        
        nSubRow=3
        nSubLine=timeObs.nObs/nSubRow
        if timeObs.nObs%nSubRow: nSubLine+=1
        plt.figure(figsize=(12.,3.*nSubLine))
        i=0
        for t in timeObs.times:
            i+=1
            sub=plt.subplot(nSubLine, nSubRow, i)
            sub.plot(g.x, x_truth.whereTime(t), 'k')
            sub.plot(timeObs[t].interpolate(g), timeObs[t].values, 'go')
            sub.set_title("$t=%f$"%t)
            if i==timeObs.nObs:
                sub.legend(["$x_{t}$",  "$y$"], loc="lower left")
        
        plt.figure(figsize=(12.,6.))
        sub=plt.subplot(1,1,1)
        sub.plot(g.x, x0_truth, 'k:')
        sub.plot(g.x, x_truth.final(), 'k')
        sub.plot(g.x, x0_a, 'r:')
        sub.plot(g.x, x_a.final(), 'r')
        sub.plot(g.x, x0_bkg, 'b:')
        sub.plot(g.x, x_bkg.final(), 'b')
        sub.legend(["${x_t}_0$", "${x_t}_f$",
                    "${x_a}_0$","${x_a}_f$",
                    "${x_b}_0$","${x_b}_f$",
                    ], loc='best')
        plt.show()


