import numpy as np
from canonicalInjection import *
from spectralLib import *

def gauss(x, x0, sig):
    return np.exp(-((x-x0)**2)/(2*sig**2))


#-------------------------------------------------
#----| Isotropic and homogeneous covariances |----
#-------------------------------------------------

def fCorr_isoHomo(g, sig, x0=0.):
    return gauss(g.x, x0, sig)

def rCTilde_sqrt_isoHomo(g, fCorr):
    """
    Transform of the square root correlation matrix
        (homogeneous and isotropic, in 'r' representation - see 
         canonicalInjection.py)


    Diagonal representation

        [c_0, c_1.real, c_1.imag, c_2.real, c_2.imag, ...]

        <!> It is still a second order tensor and applications on it
            must be done accordingly: LCL* (not LC).
            Following manipulation reflect that.

    """
    rFTilde=g.N*c2r(np.fft.fft(fCorr))

    rCTilde=np.zeros(g.N)
    rCTilde[0]=np.abs(rFTilde[0])
    for i in xrange(1, (g.N-1)/2+1):
        # rFTilde[idx pairs] contain real coefficients
        # resulting from c2r.C.(c2r)*
        rCTilde[2*i-1]=np.abs(rFTilde[2*i-1])
        rCTilde[2*i]=np.abs(rFTilde[2*i-1])
    
    if rCTilde.min()<0.:
        raise Exception(
        "rCTilde<0: square root complex => saddle point => big problem!")
    rCTilde_sqrt=np.sqrt(rCTilde)
    return rCTilde_sqrt

def make_BisoHomo_args(grid, bkgLC, bkgSig):
    corr=fCorr_isoHomo(grid, bkgLC)
    rCTilde_sqrt=rCTilde_sqrt_isoHomo(grid, corr)
    sig=bkgSig*np.ones(grid.N)
    return (sig, rCTilde_sqrt)

#----| Fourier operators |------------------------

def ifft_Adj(x):
    N=len(x)
    xi=np.zeros(N)
    xi=np.fft.fft(x)
    xi=xi/N
    return xi

def fft_Adj(xi):
    N=len(xi)

    x=np.zeros(N)
    x=np.fft.ifft(xi).real
    x=x*N
    return x

#----| B^{1/2} operators |------------------------

def B_sqrt_isoHomo_op(xi, sig, rCTilde_sqrt, aliasing=3):
    """
        B^{1/2} operator

        sig             :   1D array of std
                            (diagonal os Sigma matrix)
        rCTilde_sqrt    :   1D array of the diagonal
                            of CTilde_sqrt (in 'r' basis)
    """
    xiR=rCTilde_sqrt*xi         #   1
    xiC=r2c(xiR)                #   2
    x1=np.fft.ifft(xiC).real    #   3
    return x1*sig               #   4


def B_sqrt_isoHomo_op_Adj(x, sig, rCTilde_sqrt, aliasing=3):
    x1=x*sig                    #   4.T
    xiC=ifft_Adj(x1)            #   3.T
    xiR=r2c_Adj(xiC)            #   2.T
    return rCTilde_sqrt*xiR     #   1.T

def B_isoHomo_op(x, sig, rCTilde_sqrt):
    return B_sqrt_isoHomo_op(B_sqrt_isoHomo_op_Adj(x, sig, rCTilde_sqrt),
                        sig, rCTilde_sqrt)

#----| B^{1/2} inverse operators |----------------

def B_sqrt_isoHomo_inv_op(x, sig, rCTilde_sqrt):
    """
        B^{-1/2} operator
        
        sig             :   1D array of std
                            (diagonal os Sigma matrix)
        rCTilde_sqrt    :   1D array of the diagonal
                            of CTilde_sqrt (in 'r' basis)

        <!> I can't explain the factor 2 in all but the 0th mode,
        but that's the 'duck-tape solution' I found so that B^{-1/2}
        is the inverse of B^{1/2}.
        
    """
    N=len(x)
    xiR2=np.zeros(N)

    x2=x*sig**(-1)                    #   1
    xiC=np.fft.fft(x2)                #   2
    xiR=r2c_Adj(xiC)                  #   3
    xiR2[0]=xiR[0]
    for i in xrange(1,N):
        xiR2[i]=2.*xiR[i]             #   4
    return rCTilde_sqrt**(-1)*xiR2    #   5
    
def B_sqrt_isoHomo_inv_op_Adj(xi, sig, rCTilde_sqrt):
    """
        B^{1/2} adjoint operator

        sig             :   1D array of std
                            (diagonal os Sigma matrix)
        rCTilde_sqrt    :   1D array of the diagonal
                            of CTilde_sqrt (in 'r' basis)
    """
    N=len(xi)
    xiR=np.zeros(N)

    xiR2=rCTilde_sqrt**(-1)*xi      #   5.T
    for i in xrange(1,N):
        xiR[i]=2.*xiR2[i]
    xiR[0]=xiR2[0]                  #   4.T
    xiC=r2c(xiR)                    #   3.T
    x2=fft_Adj(xiC)                 #   2.T
    return x2*sig**(-1)             #   1.T


def B_isoHomo_inv_op(x, sig, rCTilde_sqrt):
    return B_sqrt_isoHomo_inv_op_Adj(B_sqrt_isoHomo_inv_op(
                        x, sig, rCTilde_sqrt),
                        sig, rCTilde_sqrt)



def normBInv2(x, grid, bkgLC, bkgSig):
    """ 
        x'.B^{-1}.x
    """
    B_args=make_BisoHomo_args(grid, bkgLC, bkgSig)
    return np.dot(x, B_isoHomo_inv_op(x, *B_args))

def normBInv2Norm(x, grid, bkgLC, bkgSig):
    """ 
        x'.B^{-1}.x/N
    """
    return normBInv2(x, grid, bkgLC, bkgSig)/grid.N


#-------------------------------------------------
#----| Structure function covariances |-----------
#-------------------------------------------------

def B_sqrt_str_op(xi, sig, strVec):
    return sig*strVec*xi

def B_sqrt_str_op_Adj(x, sig, strVec):
    return sig*np.dot(strVec,x)

def B_str_op(x, sig, strVec):
    return B_sqrt_str_op(B_sqrt_str_op_Adj(x, sig, strVec),
                        sig, strVec)

#-------------------------------------------------
#----| Adapted icovariances |---------------------
#-------------------------------------------------

def normBInvAdapted2(x, grid, strVec, bkgLC, bkgSig, sigAdapted):
    (SigMat, rCTilde_sqrt)=make_BisoHomo_args(grid, bkgLC, bkgSig)
    xi1=B_sqrt_isoHomo_inv_op(x, SigMat, rCTilde_sqrt)
    xiStr=B_sqrt_isoHomo_inv_op(strVec, SigMat, rCTilde_sqrt)
    xi2=xi1-sigAdapted**2/(1.+sigAdapted**2)*np.dot(np.outer(xiStr,xiStr),xi1)
    return np.dot(x,B_sqrt_isoHomo_inv_op_Adj(xi2, SigMat, rCTilde_sqrt))

#=====================================================================
#---------------------------------------------------------------------
#=====================================================================

if __name__=='__main__':
    import random as rnd
    import matplotlib.pyplot as plt
    from pseudoSpec1D import PeriodicGrid
    rnd.seed(None)
    
    correlationTest=True

    #covType='str'
    #covType='isoHomo'
    covType='inv'

    N=11
    mu=1.
    sigRnd=1.
    sig=3.
    lCorr=3.
    
    Ng=100
    g=PeriodicGrid(Ng)

    
    x=np.empty(N, dtype='complex')
    y=np.empty(N)
    
    x[0]=rnd.gauss(mu, sigRnd)
    for i in xrange(1,(N-1)/2+1):
        x[i]=rnd.gauss(mu, sigRnd)+1j*rnd.gauss(mu,sigRnd)
        x[N-i]=x[i].real-1j*x[i].imag
    for i in xrange(N):
        y[i]=rnd.gauss(mu, sigRnd)
    
    print("Testing adjoint of r2c()")
    print(np.dot(x.conj(), r2c(y))-np.dot(r2c_Adj(x),y))
    
    print("Testing adjoint of ifft()")
    Lx_y=np.dot(np.fft.ifft(x), y.conj())
    x_LAdjy=np.dot(x, ifft_Adj(y).conj())
    print(Lx_y-x_LAdjy)

    print("Testing adjoint of fft()")
    Lx_y=np.dot(x, np.fft.fft(y).conj())
    x_LAdjy=np.dot(fft_Adj(x), y.conj())
    print(Lx_y-x_LAdjy)


    
    

    if covType in ('isoHomo', 'inv'):
        sigMatrix=sig*np.ones(g.N)
        fCorr=fCorr_isoHomo(g, lCorr)
        CTilde_sqrt=rCTilde_sqrt_isoHomo(g, fCorr)

        B_args=(sigMatrix, CTilde_sqrt)


    elif covType=='str':

        def strFunc(x, Lb):
            return 0.5*(np.exp(-0.5*(x/Lb)**2)
                       *np.cos(4.*(x/Lb)))
 
        B_args=(sig, strFunc(g.x, lCorr))
    
    
    # adjoint test
    rnd.seed(None)
    mu=0.; sigNoise=2.
    xNoise=np.zeros(g.N)
    yNoise=np.zeros(g.N)
    xi=rnd.gauss(mu, sigNoise)
    for i in xrange(g.N):
        yNoise[i]=rnd.gauss(mu, sigNoise)
        xNoise[i]=rnd.gauss(mu, sigNoise)

    if covType=='isoHomo':
        testDirect=np.dot(xNoise, B_sqrt_isoHomo_op(yNoise, *B_args))
        testAdjoint=np.dot(B_sqrt_isoHomo_op_Adj(xNoise, *B_args), 
                                yNoise)

    if covType=='str':
        testDirect=np.dot(xNoise, B_sqrt_str_op(xi,*B_args))
        testAdjoint=B_sqrt_str_op_Adj(xNoise, *B_args)*xi

    if covType=='inv':
        testDirect=np.dot(yNoise, 
                        B_sqrt_isoHomo_inv_op(xNoise, *B_args))
        testAdjoint=np.dot(B_sqrt_isoHomo_inv_op_Adj(yNoise, *B_args),
                        xNoise)

    
    print("Adjoint test with noise: <x,Gy>-<G*x,y>")
    print(testDirect-testAdjoint)
    
    if correlationTest and covType in ('isoHomo','str'):
        if covType=='isoHomo': B_op=B_isoHomo_op
        if covType=='str': B_op=B_str_op

        # correlation test
        xDirac=np.zeros(g.N)
        NDirac=Ng/4
        xDirac[NDirac]=1.
        x0Dirac=g.x[NDirac]
        xTest=B_op(xDirac, *B_args)
            
        
        plt.figure()
        plt.plot(g.x, xTest, 'b', label=r'$ B(\delta(x-x_0))$')
        if covType=='isoHomo':
            plt.plot(g.x, sig**2*fCorr_isoHomo(g, lCorr, x0Dirac), 'g', 
                    label=r'$\sigma^2f_{corr}(x-x_0)$')
        elif covType=='str':
            pass
        plt.legend(loc='best')
        plt.title(r'$\sigma=%.1f$'%sig)
        plt.show()

    elif correlationTest and covType=='inv':

        #xi=gauss(g.x, 0, 10.)
        xi=gauss(g.x, 0, 3.)+2.

        tmp2=B_sqrt_isoHomo_op(xi, *B_args)
        xiTmp2=B_sqrt_isoHomo_inv_op(tmp2, *B_args)
        tmp3=B_isoHomo_inv_op(tmp2, *B_args)
        xi2=B_sqrt_isoHomo_op_Adj(tmp3, *B_args)


        g.plot(xi)
        g.plot(xiTmp2)
        g.plot(xi2)
        plt.show()
