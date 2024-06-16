
%.   Unit test for splines
%.   Comparison with built-in MATLAB splines

clear variables;
close all;

N = 7;

%. nodes
xi = 0:(N-1);
yi = rand(1,N);

%. sample points
xx = -2:0.1:(N+1);

%. MATLAB solution
yy = spline(xi,yi,xx);

%. basis
val = basisFunc_spline( xx, 0, N-1, N, splineMatrix(N) );

%. solution via basis func.
yq = yi*val;

max(max(abs(yy-yq)))

%. polynomial solution - match when N==4
pp = polyfit(xi,yi,3);
pv = polyval(pp,xx);

figure(1);
hold on;
plot(xx,yy,'b-','LineWidth',1);
plot(xx,yq,'y--','LineWidth',1);
plot(xx,pv,'ms','LineWidth',1);
hold off;
