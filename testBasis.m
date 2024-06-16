
%.   Unit test for basis functions
%.   Evaluate derivatives numerically

clear variables;
close all;

m = 100;
n = 7;

xx = 10*rand(1,m);
xmin = 1.542;
xmax = 7.931;

[ phi, dphi, ddphi, dddphi ] = basisFunc_spline( xx, xmin, xmax, n, splineMatrix(n) );

SDLT = 1e-8;

dphi_num = zeros(n,m);
for ii=1:m
    xx_a = xx(ii) + SDLT;
    phi_a = basisFunc_spline( xx_a, xmin, xmax, n, splineMatrix(n) );
    dphi_num(:,ii) = ( phi_a - phi(:,ii) )/SDLT;
end

ddphi_num = zeros(n,m);
for ii=1:m
    xx_a = xx(ii) + SDLT;
    [ dum, dphi_a ] = basisFunc_spline( xx_a, xmin, xmax, n, splineMatrix(n) );
    ddphi_num(:,ii) = ( dphi_a - dphi(:,ii) )/SDLT;
end

dddphi_num = zeros(n,m);
for ii=1:m
    xx_a = xx(ii) + SDLT;
    [ dum1, dum2, ddphi_a ] = basisFunc_spline( xx_a, xmin, xmax, n, splineMatrix(n) );
    dddphi_num(:,ii) = ( ddphi_a - ddphi(:,ii) )/SDLT;
end

max(max(abs(dphi-dphi_num)))
max(max(abs(ddphi-ddphi_num)))
max(max(abs(dddphi-dddphi_num)))
