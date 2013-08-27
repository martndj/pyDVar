function [X]=b1sur2(pxsi,nx,sigmab,rctilde,ntrunc)
    rxsitilde = rctilde.*pxsi
    xsitilde =ab2c(rxsitilde,ntrunc,nx)
    xsi2=invfft77(xsitilde,nx)
    X =xsi2*sigmab
endfunction
function [XSI]=b1sur2T(px,nx,sigmab,rctilde,ntrunc)
    xtemp = px*sigmab*nx
    xtilde = fft77(xtemp',ntrunc)
    rxtilde =c2ab(xtilde,ntrunc)
    rxtilde = rctilde.*rxtilde
    rxtilde(1)=rxtilde(1)
    for i = 2:Ntrunc+1
        rxtilde(i) = rxtilde(i)*0.5
    end
    XSI = rxtilde
endfunction
