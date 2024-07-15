function [ yhat_all, fnB, fnT, RMSE, t_min_all, t_max_all, LgradB_all, LgradT_all ] = buildKA_basisA( x, y, lab, identID, verifID, alp, Nrun, xmin, xmax, ymin, ymax, fnB0, fnT0 )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

%. num. of records
N = size(x,1);

%. num. of inputs
m = size(x,2);

%. limits
tmin = ymin;
tmax = ymax;

%. init. operators
fnB = fnB0;
fnT = fnT0;
n = size(fnB,1);
q = size(fnT,1);
p = size(fnT,2);

err_all = zeros(N,1);
RMSE = zeros(Nrun,1);
t_min_all = zeros(Nrun,p);
t_max_all = zeros(Nrun,p);

%. proj. matrices
Cpq = kron(eye(p),ones(q,1));
Cpnm = kron(eye(p),ones(1,n*m));

Mn = splineMatrix(n);
Mq = splineMatrix(q);

fnB_r = reshape(fnB,n*m,p);
fnT_r = fnT(:);

fnB0_r = reshape(fnB0,n*m,p);
fnT0_r = fnT0(:);

%. calc. bottom - basis
xr = reshape(x.',1,N*m);
[ phi ] = basisFunc_spline( xr, xmin, xmax, n, Mn );
phi_r = reshape(phi,n*m,N);
phi_re = repmat(phi_r.',1,p);

for jj=1:Nrun

    %. calc. bottom - interm.
    t = phi_r.' * fnB_r;
    t_all = t;
    
    %. calc. top
    tr = reshape(t.',1,N*p);
    [ psi, dpsi ] = basisFunc_spline( tr, tmin, tmax, q, Mq );
    psi_r = reshape(psi,q*p,N);
    yhat_all = psi_r.' * fnT_r;
    
    %. deriv.
    dpsi_r = reshape(dpsi,q*p,N);
    top = dpsi_r.' * diag(fnT_r.') * Cpq;
    LgradB_all = ( top * Cpnm ) .* phi_re;
    LgradT_all = psi_r.';

    for ii=1:N
        %. calc.
        if ( lab(ii) == identID )||( lab(ii) == verifID )

            yy = y(ii);
            yhat = yhat_all(ii);
            LgradB = LgradB_all(ii,:);
            LgradT = LgradT_all(ii,:);

            yhat_e = yhat + LgradT*(fnT_r-fnT0_r) + LgradB*(fnB_r(:)-fnB0_r(:));
            Lnum_e = yhat_e - yy;

            %. export
            err_all(ii) = abs(Lnum_e);
        end
        
        %. ident.
        if ( lab(ii) == identID )
            chi = sum(LgradB.^2) + sum(LgradT.^2);
            fnB_r = fnB_r - alp * Lnum_e * reshape(LgradB,n*m,p)/chi;
            fnT_r = fnT_r - alp * Lnum_e * LgradT.'/chi;
        end
    end

    fnB0_r = fnB_r;
    fnT0_r = fnT_r;

    inds = ( lab == verifID );
    RMSE(jj) = sqrt( sum( err_all(inds).^2 )/sum(inds) )/(ymax-ymin);
    t_min_all(jj,:) = min(t_all(inds,:));
    t_max_all(jj,:) = max(t_all(inds,:));

    printProgr = 1;
    if ( printProgr == 1 )
        if ( jj > 1 )
            fprintf( repmat( '\b', 1, 34 ) );
        end
        fprintf( '  pass %04.0f out of %04.0f completed\n', jj, Nrun );
    end
end

fnB = reshape(fnB_r,n,p*m);
fnT = reshape(fnT_r,q,p);

end
