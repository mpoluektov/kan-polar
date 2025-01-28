function [ yhat_all, fnB, fnM, fnT, RMSE, t_min_all, t_max_all, s_min_all, s_max_all ] = buildKAdeep_basisA( x, y, lab, identID, verifID, alp, Nrun, xmin, xmax, ymin, ymax, fnB0, fnM0, fnT0 )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

%. num. of records
N = size(x,1);

%. num. of inputs
m = size(x,2);

%. limits
tmin = ymin;
tmax = ymax;
smin = ymin;
smax = ymax;

%. init. operators
fnB = fnB0;
fnM = fnM0;
fnT = fnT0;
n = size(fnB,1);
h = size(fnM,1);
q = size(fnT,1);
p = size(fnB,2)/m;
r = size(fnM,2)/p;

err_all = zeros(N,1);
RMSE = zeros(Nrun,1);
PC = zeros(Nrun,1);
t_min_all = zeros(Nrun,p);
t_max_all = zeros(Nrun,p);
s_min_all = zeros(Nrun,r);
s_max_all = zeros(Nrun,r);

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

fnB0_r = reshape(fnB0,n*m,p);
fnM0_r = reshape(fnM0,h*p,r);
fnT0_r = fnT0(:);

%. calc. bottom - basis
xr = reshape(x.',1,N*m);
[ phi ] = basisFunc_spline( xr, xmin, xmax, n, Mn );
phi_r = reshape(phi,n*m,N);
phi_re = repmat(phi_r,p,1);

for jj=1:Nrun

    %. calc. bottom - interm.
    t = phi_r.' * fnB_r;
    t_all = t;
    
    %. calc. middle
    tr = reshape(t.',1,N*p);
    [ xi, dxi ] = basisFunc_spline( tr, tmin, tmax, h, Mh );
    xi_r = reshape(xi,h*p,N);
    dxi_r = reshape(dxi,h*p,N);
    xi_re = repmat(xi_r,r,1);
    dxi_re = repmat(dxi_r,r,1);
    s = xi_r.' * fnM_r;
    s_all = s;
    
    %. calc. top
    sr = reshape(s.',1,N*r);
    [ psi, dpsi ] = basisFunc_spline( sr, smin, smax, q, Mq );
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

    for ii=1:N
        %. calc.
        if ( lab(ii) == identID )||( lab(ii) == verifID )

            yy = y(ii);
            yhat = yhat_all(ii);
            LgradB = LgradB_all(ii,:);
            LgradM = LgradM_all(ii,:);
            LgradT = LgradT_all(ii,:);

            yhat_e = yhat + LgradT*(fnT_r-fnT0_r) + LgradM*(fnM_r(:)-fnM0_r(:)) + LgradB*(fnB_r(:)-fnB0_r(:));
            Lnum_e = yhat_e - yy;

            %. export
            err_all(ii) = abs(Lnum_e);
        end
        
        %. ident.
        if ( lab(ii) == identID )
            chi = sum(LgradB.^2) + sum(LgradM.^2) + sum(LgradT.^2);
            fnB_r = fnB_r - alp * Lnum_e * reshape(LgradB,n*m,p)/chi;
            fnM_r = fnM_r - alp * Lnum_e * reshape(LgradM,h*p,r)/chi;
            fnT_r = fnT_r - alp * Lnum_e * LgradT.'/chi;
        end
    end

    fnB0_r = fnB_r;
    fnM0_r = fnM_r;
    fnT0_r = fnT_r;

    inds = ( lab == verifID );
    RMSE(jj) = sqrt( sum( err_all(inds).^2 )/sum(inds) )/(ymax-ymin);
    t_min_all(jj,:) = min(t_all(inds,:));
    t_max_all(jj,:) = max(t_all(inds,:));
    s_min_all(jj,:) = min(s_all(inds,:));
    s_max_all(jj,:) = max(s_all(inds,:));

    PCt = corrcoef( y(inds), yhat_all(inds) );
    PC(jj) = PCt(1,2);

    printProgr = 1;
    if ( printProgr == 1 )
        fprintf( '  pass %04.0f out of %04.0f completed, RMSE=%.4f, Pearson=%.4f\n', jj, Nrun, RMSE(jj), PC(jj) );
    end
end

fnB = reshape(fnB_r,n,m*p);
fnM = reshape(fnM_r,h,p*r);
fnT = reshape(fnT_r,q,r);

end
