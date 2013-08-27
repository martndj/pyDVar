//
//  SCA-7212  Introduction à l'assimilation de données
//
//            1D-Var avec corrélations homogènes et isotropes
//            UQAM Hiver 2011
//            P. Gauthier (Février 2011)
//            ------------------------------------------------
//
//            Nécessite un second programme définissant la fonction coût:
//            -  cost1Dvar.sci
//
clear
repertoire=pwd()+'\'
exec(repertoire+'fft77.sci');
exec(repertoire+'invfft77.sci');
exec(repertoire+'b1sur2.sci');
//exec(repertoire+'cost4Dvar.sci',-1)
//exec(repertoire+'cost1Dvar.sci',-1)
exec(repertoire+'costspectral.sci',-1)
//exec(repertoire+'advectionfwd.sci',-1)
//
//        Variables globales sont initialisée ici pour être utilisées
//        par cost1Dvar.sci
//
global U D1sur2 V H sigmab Sigma_obs yobs rctilde NX Ntrunc
//
//   Caractéristiques du problème d'assimilation
//
NX = 5001
LX = 10000
Ntrunc = 200
sigmab = 1.
Lcaract = 400
x0 = LX/4
Sigma_obs = 1
ydata = 1
//
// Génération des points de grille
//
xtemp = linspace(0,LX,NX+1)
x= xtemp(1:NX)
x = matrix(x,NX,1)
//x0 = x(NX/4)
delx = x(2)-x(1)
//
//
//    1. Définition de la matrice de corrélation
//
zfacteur = 1/(2*(Lcaract**2))
//
// Définition de la corrélation utilise la distance entre
// deux points sur un cercle de latitude
// IMPORTANT: il faut que le maximum de la fonction soit placéeà
//            à l'origine
fx=zeros(NX,1)
for k=1:NX
    phi = abs(x(k) -x(1))
    if phi <= LX/2 then
        dist = phi
    else
        dist =LX - phi
    end
    fx(k)=exp(-zfacteur*(dist**2))
end
//
//
//   - Définition de Ctilde basée sur les coefficients de Fourier
//     de la fonction de corrélation
//
fxtilde = fft77(fx,Ntrunc)
rfxtilde2 =c2ab(fxtilde,Ntrunc)
//
rfxtilde =rfxtilde2
//
// Longueur caractéristique
//
K = 2*%pi/LX
K2 =K*K
norm0 = rfxtilde(1)
for i = 1:Ntrunc
    norm0 = norm0 + rfxtilde(2*i)
end
printf('C(0) = %f \n',norm0)
somme = 0.
for k =1:Ntrunc
    somme = somme -k*k*(K2*rfxtilde(2*k))
end
printf('Longueur caractéristique = %g \n\n',sqrt(-1/(somme)))
//
rfxtilde=rfxtilde2
rctilde = zeros(2*Ntrunc+1,1)
rctilde(1)=abs(rfxtilde(1))
for i = 1:Ntrunc
    temp          = abs(rfxtilde(2*i))
    rctilde(2*i)  = temp
    rctilde(2*i+1)= temp
end

rctilde =sqrt(rctilde)
//
// TEST de la fonction de corrélation
//
xtest =zeros(NX,1)
xtest(0.25*NX)=1
xsitemp = b1sur2T(xtest,NX,sigmab,rctilde,Ntrunc)
xtest2  = b1sur2(xsitemp,NX,sigmab,rctilde,Ntrunc)
//
//    3. Observations
//          a) Définition
nobs =100
xobs  = zeros(nobs,1)
alpha = zeros(nobs,1)
yobs  = zeros(nobs,1)
delxobs =3*LX/(4.*nobs)
//delxobs =6*Lcaract
H = zeros(nobs,NX)
//
//xobs =[2500; 4000; 11900]
for k = 1:nobs
    xobs(k)     = x0 + (k-1)*delxobs
    kobs        = int(xobs(k)/(delx))+1
    alpha(k)    = (xobs(k) -x(kobs))/delx
    H(k,kobs)   = alpha(k)
    H(k,kobs+1) = 1- alpha(k)
    yobs(k)     = ydata
end
//
// Minimisation
//
indic=2;
xsi0 = zeros(2*Ntrunc+1,1)
[Jf0,grad_J,indic]=costspectral(xsi0,indic);
//
// Test permettant de visualiser grad(J)
//
gradJx=b1sur2(grad_J,NX,sigmab,rctilde,Ntrunc)
//
// Test du gradient: vérification de la cohérence entre l'évaluation
// de la fonction J(x) et de son gradient
//
epsilon = 0.1
normg0 = grad_J'*grad_J
printf('   J(0) = %g Norme de grad J = %g \n',Jf0, normg0)
printf('        Epsilon        J(x)         R\n')
indic = 1
for n = 1:9
    xsi        =  xsi0 -(epsilon**n)*grad_J
    Jf         =  costspectral(xsi,indic)
    Rapport    =  -(Jf -Jf0)/((epsilon**n)*(normg0))
    printf('         %g12         %g12         %g12   %i  \n',..
    epsilon**n, Jf, Rapport,indic)
end

xsi =matrix(xsi0,2*Ntrunc+1,1)
indic=2
[Jf,xopt,gopt]=optim(costspectral,xsi,'gc',Jf0,3,'ar',100,100,0.001,imp=3);
//
// Expression de l'incrément dans l'espace physique
//
increment =b1sur2(xopt,NX,sigmab,rctilde,Ntrunc)
//
// Forme analytique pour le cas à une seule observation
//
incranl=zeros(1,NX)
for k=1:NX
    phi = abs(x(k) -xobs(1))
    if phi <= LX/2 then
        dist = phi
    else
        dist =LX - phi
    end
    incranl(k)=exp(-zfacteur*(dist**2))
end
incranl = incranl*(sigmab**2/(sigmab**2+Sigma_obs**2))*yobs(1)
clf()
plot(x,increment,'-k')
if nobs==1 then
    plot(x,incranl,'--k')
end
