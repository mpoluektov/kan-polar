function [ yhat_all, LgradB_all, LgradM_all, LgradT_all, t_min, t_max, s_min, s_max ] = modelKAdeep_basisC( x, xmin, xmax, ymin, ymax, fnB, fnM, fnT )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

N = size(x,1);
m = size(x,2);
n = size(fnB,1);
h = size(fnM,1);
q = size(fnT,1);
p = size(fnB,2)/m;
r = size(fnM,2)/p;

tmin = ymin;
tmax = ymax;
smin = ymin;
smax = ymax;

%. proj. matrices
Crq = kron(eye(r),ones(q,1));
Cpnm = kron(eye(p),ones(1,n*m));
Crhp = kron(eye(r),ones(1,h*p));
Cprh = kron(eye(p*r),ones(h,1));
Crp = kron(ones(r,1),eye(p));

Mn = splineMatrix(n);
Mh = splineMatrix(h);
Mq = splineMatrix(q);

fnB_r = reshape(fnB,n*m,p);
fnM_r = reshape(fnM,h*p,r);
fnT_r = fnT(:);

%. calc. bottom
xr = reshape(x.',1,N*m);
[ phi, dphi, ddphi ] = basisFunc_spline( xr, xmin, xmax, n, Mn );
phi_r = reshape(phi,n*m,N);
phi_re = repmat(phi_r,p,1);
t = phi_r.' * fnB_r;

%. calc. middle
tr = reshape(t.',1,N*p);
[ xi, dxi, ddxi ] = basisFunc_spline( tr, tmin, tmax, h, Mh );
xi_r = reshape(xi,h*p,N);
dxi_r = reshape(dxi,h*p,N);
xi_re = repmat(xi_r,r,1);
dxi_re = repmat(dxi_r,r,1);
s = xi_r.' * fnM_r;

%. calc. top
sr = reshape(s.',1,N*r);
[ psi, dpsi, ddpsi ] = basisFunc_spline( sr, smin, smax, q, Mq );
psi_r = reshape(psi,q*r,N);
dpsi_r = reshape(dpsi,q*r,N);
yhat_all = psi_r.' * fnT_r;

%. deriv.
top = dpsi_r.' * diag(fnT_r.') * Crq;
interm = dxi_re.' * diag(fnM_r(:).') * Cprh;
middle = ( interm .* kron(top,ones(1,p)) ) * Crp;
LgradB_all = ( middle * Cpnm ) .* phi_re.';
LgradM_all = ( top * Crhp ) .* xi_re.';
LgradT_all = psi_r.';

t_min = min(t);
t_max = max(t);
s_min = min(s);
s_max = max(s);

end
