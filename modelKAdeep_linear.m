function [ yhat_all, LgradB_all, LgradM_all, LgradT_all, t_min, t_max, s_min, s_max ] = modelKAdeep_linear( x, xmin, xmax, ymin, ymax, fnB, fnM, fnT )
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
Cph = kron(eye(p),ones(h,1));
Crq = kron(eye(r),ones(q,1));
Cpnm = kron(eye(p),ones(1,n*m));
Crhp = kron(eye(r),ones(1,h*p));

fnB_r = reshape(fnB,n*m,p);
fnM_r = reshape(fnM,h*p,r);
fnT_r = fnT(:);

%. calc. bottom
xt = x.';
xr = xt(:);
b = 1 + (n-1) * ( xr - xmin )/( xmax - xmin );
kL = floor(b);
kR = ceil(b);
indOutL = ( kL <= 0 );
indOutR = ( kR >= n+1 );
kL(indOutL) = 1;
kR(indOutL) = 2;
kL(indOutR) = n-1;
kR(indOutR) = n;
psi = b - kL;
om_psi = 1 - psi;

rw = repmat( (1:N), m, 1 );
rwr = rw(:);
sh = repmat( n*(0:1:(m-1)).', N, 1 );
V = accumarray( [rwr kL+sh; rwr kR+sh], [om_psi; psi], [N n*m] );
t = V * fnB_r;

%. calc. middle
tt = t.';
tr = tt(:);
d = 1 + (h-1) * ( tr - tmin )/( tmax - tmin );
uL = floor(d);
uR = ceil(d);
indOutL = ( uL <= 0 );
indOutR = ( uR >= h+1 );
uL(indOutL) = 1;
uR(indOutL) = 2;
uL(indOutR) = h-1;
uR(indOutR) = h;
xi = d - uL;
om_xi = 1 - xi;

rw = repmat( (1:N), p, 1 );
rwr = rw(:);
sh = repmat( h*(0:1:(p-1)).', N, 1 );
aai = [rwr uL+sh; rwr uR+sh];
Q = accumarray( aai, [om_xi; xi], [N h*p] );
Qd = accumarray( aai, [-ones(N*p,1); ones(N*p,1)], [N h*p] );
s = Q * fnM_r;

%. calc. top
st = s.';
sr = st(:);
c = 1 + (q-1) * ( sr - smin )/( smax - smin );
vL = floor(c);
vR = ceil(c);
indOutL = ( vL <= 0 );
indOutR = ( vR >= q+1 );
vL(indOutL) = 1;
vR(indOutL) = 2;
vL(indOutR) = q-1;
vR(indOutR) = q;
phi = c - vL;
om_phi = 1 - phi;

rw = repmat( (1:N), r, 1 );
rwr = rw(:);
sh = repmat( q*(0:1:(r-1)).', N, 1 );
aai = [rwr vL+sh; rwr vR+sh];
U = accumarray( aai, [om_phi; phi], [N q*r] );
Ud = accumarray( aai, [-ones(N*r,1); ones(N*r,1)], [N q*r] );
yhat_all = U * fnT_r;

%. deriv.
top = Ud * diag(fnT_r.') * Crq * (q-1) / ( smax - smin );
interm = Qd * ( kron(fnM_r,ones(1,p)) .* repmat(Cph,1,r) ) * (h-1) / ( tmax - tmin );
middle = ( interm .* kron(top,ones(1,p)) ) * kron(ones(r,1),eye(p));
V_ex = repmat(V,1,p);
Q_ex = repmat(Q,1,r);
LgradB_all = ( middle * Cpnm ) .* V_ex;
LgradM_all = ( top * Crhp ) .* Q_ex;
LgradT_all = U;

t_min = min(t);
t_max = max(t);
s_min = min(s);
s_max = max(s);

end
