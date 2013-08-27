function [fx]=invfft77(spfx,ni)
    sptemp = zeros(ni)
    nspec = (ni+1)/2
    sptemp(1) = spfx(1)
    for k= 1:nspec
        sptemp(k+1) = spfx(k+1)
        sptemp(ni-k+1) = conj(spfx(k+1))
    end
    fx0 = ni*fft(sptemp,+1)
    fx = real(fx0)
endfunction
function [rsp]=c2ab(csp,ntrunc)
    ni    = size(csp)
    rsp   = zeros(2*ntrunc+1)
    rsp(1)= real(csp(1))
    for n =1:ntrunc
        rsp(2*n)  =  2.*real(csp(n+1))
        rsp(2*n+1)= -2.*imag(csp(n+1))
    end
endfunction

