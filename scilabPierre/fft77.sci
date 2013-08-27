function [spfx,ni,nmax]=fft77(fx,ntrunc)
    taille = size(fx)
    ni = max(taille(1),taille(2))
    fx =matrix(fx,1,ni)
    nmax =(ni-1)/2
    spfx =fft(fx,-1)/ni
    fx =matrix(fx,taille)
    //
    // Tronque les nombres d'ondes superieurs a ntrunc
    //
    spfx(ntrunc+2:ni) = 0.
    for n= 1:ntrunc
        spfx(ni-n+1) = conj(spfx(n+1))
    end
    //
endfunction

function [csp]=ab2c(rsp,ntrunc,ni)
    csp=zeros(NX,1)
    csp(1)=rsp(1)
    for n =1:ntrunc
        csp(n+1)   = 0.5*(rsp(2*n)-%i*rsp(2*n+1))
        csp(ni-n+1)= 0.5*(rsp(2*n)+%i*rsp(2*n+1))
    end
endfunction

function [rsp]=ab2c_ad(csp,ntrunc)
    ni    = size(csp)
    rsp   = zeros(2*ntrunc+1)
    rsp(1)= real(csp(1))
    for n =1:ntrunc
        rsp(2*n)  =  real(csp(n+1))
        rsp(2*n+1)=  imag(csp(n+1))
    end
endfunction
