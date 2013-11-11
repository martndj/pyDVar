import numpy as np

def c2r(csp):
    """
    Real to complex (hermitian signal)

        ordering convention
        -------------------
        c_i=a_i+j*b_i
        c_{-i}=(c_i)*

        csp=[c_0, c_1, c_2, ..., c_{N-1}]
        rsp=[a_0, a_1, b_1, a_2, b_2, ..., a_{(N-1)/2+1}, b_{(N-1)/2+1}]
        
        csp.shape=(N+1)
        rsp.shape=(N+1)
    """
    N=len(csp)-1
    rsp=np.zeros(N+1)
    rsp[0]=csp[0].real
    for i in xrange(1,N/2+1):
        rsp[2*i-1]    =2.*csp[i].real
        rsp[2*i]      =2.*csp[i].imag
    return rsp

def r2c(rsp):
    N=len(rsp)-1
    csp=np.zeros(N+1, dtype=complex)
    csp[0]=rsp[0]
    for i in xrange(1,N/2+1):
        csp[i]     =0.5*(rsp[2*i-1]+1j*rsp[2*i])
        csp[N-i+1]   =0.5*(rsp[2*i-1]-1j*rsp[2*i])
    return csp

def r2c_Adj(csp):
    N=len(csp)-1
    rsp=np.zeros(N+1)
    for i in xrange(1, N/2+1):
        rsp[2*i-1]  =csp[i].real
        rsp[2*i]    =csp[i].imag
        # <!> carefull here: complex conjugate necessary!
    rsp[0]=csp[0].real
    return rsp


def rTrunc(rsp, Ntrc):
    rsp[2*Ntrc+1:]=0.
    return rsp
