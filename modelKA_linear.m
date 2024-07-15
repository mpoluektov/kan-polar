function [ yhat_all, LgradB_all, LgradT_all, t_min, t_max ] = modelKA_linear( x, xmin, xmax, ymin, ymax, fnB, fnT )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

N = size(x,1);
m = size(x,2);
n = size(fnB,1);
q = size(fnT,1);
p = size(fnT,2);

tmin = ymin;
tmax = ymax;

%. proj. matrices
Cpq = kron(eye(p),ones(q,1));
Cpnm = kron(eye(p),ones(1,n*m));

fnB_r = reshape(fnB,m*n,p);
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

%. calc. top
tt = t.';
tr = tt(:);
c = 1 + (q-1) * ( tr - tmin )/( tmax - tmin );
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

ry = repmat( (1:N), p, 1 );
ryr = ry(:);
sh = repmat( q*(0:1:(p-1)).', N, 1 );
aai = [ryr vL+sh; ryr vR+sh];
U = accumarray( aai, [om_phi; phi], [N q*p] );
Ud = accumarray( aai, [-ones(N*p,1); ones(N*p,1)], [N q*p] );
yhat_all = U * fnT_r;

%. deriv.
top = Ud * diag(fnT_r.') * Cpq * (q-1) / ( tmax - tmin );
V_ex = repmat(V,1,p);
LgradB_all = ( top * Cpnm ) .* V_ex;
LgradT_all = U;

t_min = min(t);
t_max = max(t);

end
