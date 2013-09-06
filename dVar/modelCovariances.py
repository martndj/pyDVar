import numpy as np
from canonicalInjection import *

def gauss(x, x0, sig):
    return np.exp(-((x-x0)**2)/(2*sig**2))


def fCorr_isoHomo(g, sig):
    return gauss(g.x, 0., sig)

def rCTilde_sqrt_isoHomo(g, fCorr):
    """
        Construit la matrice CTilde_sqrt isotrope et homogene
        dans la base 'r'.


        Comme celle-ci est diagonale, on la representente comme
        un vecteur (sa diagonale).

        [c_0, c_1.real, c_1.imag, c_2.real, c_2.imag, ...]

        <!> Attention, elle reste un tenseur d'ordre 2,
            il faudra cependant etre coherent dans l'application
            des transformations et changement de base (de 'r' a 'c')
            des operateurs LCL* et non LC...
            les manipulations qui suivent resultent de cela

    """
    rFTilde=g.N*c2r(np.fft.fft(fCorr))

    rCTilde=np.zeros(g.N)
    rCTilde[0]=np.abs(rFTilde[0])
    for i in xrange(1, (g.N-1)/2+1):
        # rFTilde[idx pairs] contiennent les coefs reels
        # resultant de c2r.C.(c2r)*
        # (meme si on l'ecrit comme un vecteur, il s'agit de la diagonale
        #   d'une matrice - un tenseur d'ordre 2, donc il faut appliquer
        #   les operateurs de chaque cote)
        rCTilde[2*i-1]=np.abs(rFTilde[2*i-1])
        rCTilde[2*i]=np.abs(rFTilde[2*i-1])
    
    rCTilde_sqrt=np.sqrt(rCTilde)
    return rCTilde_sqrt


def ifft_Adj(x):
    N=len(x)
    xi=np.zeros(N)
    xi=np.fft.fft(x)
    xi=xi/N
    return xi

def B_sqrt_op(xi, var, rCTilde_sqrt):
    """
        B_{1/2} operator

        var             :   1D array of variances
                            (diagonal os Sigma matrix)
        rCTilde_sqrt    :   1D array of the diagonal
                            of CTilde_sqrt (in 'r' basis)
    """
    xiR=rCTilde_sqrt*xi              #   1
    xiC=r2c(xiR)                #   2
    x1=np.fft.ifft(xiC).real    #   3
    return x1*var               #   4

def B_sqrt_op_T(x, var, rCTilde_sqrt):
    x1=x*var                    #   4.T
    xiC=ifft_Adj(x1)        #   3.T
    xiR=r2c_Adj(xiC)            #   2.T
    return rCTilde_sqrt*xiR      #   1.T



if __name__=='__main__':
    import random as rnd
    import matplotlib.pyplot as plt
    from pseudoSpec1D import SpectralGrid
    rnd.seed(0.4573216806)
    
    N=11
    mu=1.
    sig=1.
    
    
    
    x=np.empty(N, dtype='complex')
    y=np.empty(N)
    
    x[0]=rnd.gauss(mu, sig)
    for i in xrange(1,(N-1)/2+1):
        x[i]=rnd.gauss(mu, sig)+1j*rnd.gauss(mu,sig)
        x[N-i]=x[i].real-1j*x[i].imag
    for i in xrange(N):
        y[i]=rnd.gauss(mu, sig)
    
    print("Testing adjoint of r2c()")
    print(np.dot(x.conj(), r2c(y))-np.dot(r2c_Adj(x),y))
    
    print("Testing adjoint of ifft()")
    Lx_y=np.dot(np.fft.ifft(x), y.conj())
    x_LAdjy=np.dot(x, ifft_Adj(y).conj())
    print(Lx_y-x_LAdjy)



    Ng=100
    # pour sig=0.1 ca colle presque...
    g=SpectralGrid(Ng, 100., aliasing=1)
    
    
    sig=1.
    lCorr=5.
    variances=sig*np.ones(g.N)
    fCorr=fCorr_isoHomo(g, lCorr)
    CTilde_sqrt=rCTilde_sqrt_isoHomo(g, fCorr)
    
    # correlation test
    xDirac=np.zeros(g.N)
    xDirac[Ng/4]=1.
    xiTest=B_sqrt_op_T(xDirac,  variances, CTilde_sqrt)
    xTest=B_sqrt_op(xiTest, variances, CTilde_sqrt)
    
    # adjoint test
    rnd.seed(0.4573216806)
    mu=0.; sigNoise=2.
    xNoise=np.zeros(g.N)
    yNoise=np.zeros(g.N)
    for i in xrange(g.N):
        yNoise[i]=rnd.gauss(mu, sigNoise)
        xNoise[i]=rnd.gauss(mu, sigNoise)
    testDirect=np.dot(xNoise,
                        B_sqrt_op(yNoise, variances, CTilde_sqrt).conj())
    testAdjoint=np.dot(B_sqrt_op_T(xNoise, variances, CTilde_sqrt),
                        yNoise.conj())
    
    #print(testDirect)
    #print(testAdjoint)
    print("Adjoint test with noise: <x,Gy>-<G*x,y>")
    print(testDirect-testAdjoint)
    
    
    #plt.plot(g.x, xTest/sig**2)
    plt.plot(g.x, xTest)
    plt.plot(g.x, fCorr)
    plt.show()
