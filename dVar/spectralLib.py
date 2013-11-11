import numpy as np
import numpy.fft as fft

class spectralLibError(Exception):
    pass

def fftOrder(N):
    """
    Return the FFT order of frequency index
        
        fftOrder(N)

        N       :   number of grid point
    """
    m=np.zeros(N)
    for i in xrange(N):
        if i <= (N-1)/2:
            m[i]=i
        else:
            m[i]=i-N
    return m

def specFilt(x, Ntrc):
    N=len(x)
    tf=fft.fft(x)
    m=fftOrder(N)
    for i in xrange(N):
        # truncature
        if np.abs(m[i])>Ntrc:
            tf[i]=0.
    f=(fft.ifft(tf)).real.copy(order='C')
    return f
