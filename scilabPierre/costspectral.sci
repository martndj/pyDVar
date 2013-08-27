function [costj,gradient,indic]=costspectral(pxsi,indic)
    //
    //  SCA-7212  Introduction à l'assimilation de données
    //
    //            Fonction objective 1D-Var avec corrélations homogènes et isotropes
    //            UQAM Hiver 2011
    //            P. Gauthier (Février 2011)
    //            ------------------------------------------------
    //
    //            A être utilisé avec SCA7212_1Dvar.sce qui initialise les variables globales
    //            nécessaires pour évaluer cette fonction: ces variables sont définies comme
    //            globales
    //
    global U D1sur2 V H sigmab Sigma_obs yobs rctilde NX Ntrunc
    //
    // 1. Evaluation de J(x)
    //
    // a) Application de B1/2
    //
    xsi2 =b1sur2(pxsi,NX,sigmab,rctilde,Ntrunc)
    //
    // b) Application de H
    //
    w = (H*xsi2- yobs)
    //
    // c) Normalisation par sigma_obs
    //
    w =w/Sigma_obs
    //
    // d) Evaluation de la fonction objective
    //
    costj =0.5*(pxsi'*pxsi + w'*w)
    gradient = zeros(size(pxsi))
    if indic==1 then
//        mprintf('INDIC: %i J(X) = %g10 Norm = %g12 \n'..
//        ,indic,costj,sqrt(pxsi'*pxsi))
    else
        // 
        //  2. Evaluation du gradient
        //
        //  a) Normalisation par Sigma_obs
        wtilde = w/Sigma_obs
        //
        //  b) Application de la transposee (adjoint) de l'opérateur d'observation
        //
        xtemp2= zeros(NX,1),xtemp =zeros(NX,1), xtilde =zeros(NX,1)
        xtemp2 = H'*wtilde
        xsitilde =b1sur2T(xtemp2,NX,sigmab,rctilde,Ntrunc)
        //
        //   c) Calcul du gradient
        //
        gradient = pxsi + xsitilde
        normeg= gradient'*gradient
//        mprintf('INDIC: %i J(X) = %g Norme Grad J = %g \n',indic,costj,normeg)
    end
endfunction

