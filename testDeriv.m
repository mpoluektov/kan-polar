
%.   Unit test for derivatives
%.   Evaluate derivatives numerically

clear variables;
close all;

xmin = 0.72;
xmax = 1.59;

m = 3;
n = 7;
q = 4;
p = 5;

Ni = 100;
xi = xmin + (xmax-xmin)*rand(Ni,m);

ymin = 0.31;
ymax = 2.68;
[ fnB0, fnT0 ] = buildKA_init( m, n, q, p, ymin, ymax );

L = @(s,u,du,ddu,z) u*du(2)^2 + 2*s(1)*s(2)*du(1) - s(3)*(u^3)*sin(du(3)) + ddu(1) + u*ddu(2) + sin(ddu(3)) + ddu(4)*ddu(5)/s(1) - 4*ddu(6);
dLdu = @(s,u,du,ddu,z) du(2)^2 - s(3)*(3*u^2)*sin(du(3)) + ddu(2);
dLddu = @(s,u,du,ddu,z) [2*s(1)*s(2) u*2*du(2) -s(3)*(u^3)*cos(du(3))];
dLdddu = @(s,u,du,ddu,z) [1 u cos(ddu(3)) ddu(5)/s(1) ddu(4)/s(1) -4];

[ yhat_all, dyhat_all, ddyhat_all, fnB, fnT, RMSE, t_min_all, t_max_all, L_all, LgradB_all, LgradT_all ] = buildKA_basisF( xi, zeros(Ni,1), zeros(Ni,1), 1, 0, 0, 1, xmin, xmax, ymin, ymax, L, dLdu, dLddu, dLdddu, fnB0, fnT0 );

SDLT = 1e-8;

LgradB_num = zeros(size(LgradB_all));
for ii=1:n
    for jj=1:p*m
        fnB0_a = fnB0;
        fnB0_a(ii,jj) = fnB0_a(ii,jj) + SDLT;
        [ dum1, dum2, dum3, dum4, dum5, dum6, dum7, dum8, L_all_a ] = buildKA_basisF( xi, zeros(Ni,1), zeros(Ni,1), 1, 0, 0, 1, xmin, xmax, ymin, ymax, L, dLdu, dLddu, dLdddu, fnB0_a, fnT0 );
        LgradB_num(:,(jj-1)*n+ii) = ( L_all_a - L_all )/SDLT;
    end
end

LgradT_num = zeros(size(LgradT_all));
for ii=1:q
    for jj=1:p
        fnT0_a = fnT0;
        fnT0_a(ii,jj) = fnT0_a(ii,jj) + SDLT;
        [ dum1, dum2, dum3, dum4, dum5, dum6, dum7, dum8, L_all_a ] = buildKA_basisF( xi, zeros(Ni,1), zeros(Ni,1), 1, 0, 0, 1, xmin, xmax, ymin, ymax, L, dLdu, dLddu, dLdddu, fnB0, fnT0_a );
        LgradT_num(:,(jj-1)*q+ii) = ( L_all_a - L_all )/SDLT;
    end
end

dyhat_num = zeros(Ni,m);
for ii=1:m
    xi_a = xi;
    xi_a(:,ii) = xi_a(:,ii) + SDLT;
    [ yhat_all_a ] = buildKA_basisF( xi_a, zeros(Ni,1), zeros(Ni,1), 1, 0, 0, 1, xmin, xmax, ymin, ymax, L, dLdu, dLddu, dLdddu, fnB0, fnT0 );
    dyhat_num(:,ii) = ( yhat_all_a - yhat_all )/SDLT;
end

ddyhat_num = zeros(Ni,m^2);
for ii=1:m
    xi_a = xi;
    xi_a(:,ii) = xi_a(:,ii) + SDLT;
    [ dum1, dyhat_all_a ] = buildKA_basisF( xi_a, zeros(Ni,1), zeros(Ni,1), 1, 0, 0, 1, xmin, xmax, ymin, ymax, L, dLdu, dLddu, dLdddu, fnB0, fnT0 );
    ddyhat_num(:,(1:m)+(ii-1)*m) = ( dyhat_all_a - dyhat_all )/SDLT;
end

%. remove double-calc. deriv. for m==3
ddyhat_num(:,[4 7 8]) = [];

max(max(abs(LgradB_all-LgradB_num)))
max(max(abs(LgradT_all-LgradT_num)))
max(max(abs(dyhat_all-dyhat_num)))
max(max(abs(ddyhat_all-ddyhat_num)))
