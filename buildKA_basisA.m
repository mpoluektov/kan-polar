function [ yhat_all, fnB, fnT, RMSE, t_min_all, t_max_all, t_all ] = buildKA_basisA( x, y, lab, identID, verifID, alp, nrmse, Nrun, xmin, xmax, ymin, ymax, fnB0, fnT0 )
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

%. num. of param.
Npar = n*m*p + q*p;

err_all = zeros(N,1);
yhat_all = zeros(N,1);
RMSE = zeros(Nrun,1);
PC = zeros(Nrun,1);
t_min_all = zeros(Nrun,p);
t_max_all = zeros(Nrun,p);

%. proj. matrices
Cpq = kron(eye(p),ones(q,1));
Cpnm = kron(eye(p),ones(1,n*m));

Mn = splineMatrix(n);
Mq = splineMatrix(q);

fnB_r = reshape(fnB,n*m,p);
fnT_r = fnT(:);

%. calc. bottom - basis
xr = reshape(x.',1,N*m);
[ phi ] = basisFunc_spline( xr, xmin, xmax, n, Mn );
phi_r = reshape(phi,n*m,N);
phi_re = repmat(phi_r.',1,p);

for jj=1:Nrun

    %. normalise
    if ( nrmse == 1 )
        fnB = reshape(fnB_r,n,p*m);
        fnB_nrm = normBlock( fnB, m, tmin, tmax );
        fnB_r = reshape(fnB_nrm,n*m,p);
    end

    %. calc. bottom - interm.
    t = phi_r.' * fnB_r;
    t_all = t;
    
    %. calc. top - prediction
    tr = reshape(t.',1,N*p);
    [ psi, dpsi ] = basisFunc_spline( tr, tmin, tmax, q, Mq );
    psi_r = reshape(psi,q*p,N);
    dpsi_r = reshape(dpsi,q*p,N);
    yhat_pred = psi_r.' * fnT_r;
    
    %. deriv.
    top = dpsi_r.' * diag(fnT_r.') * Cpq;
    LgradB_all = ( top * Cpnm ) .* phi_re;
    LgradT_all = psi_r.';

    %. param. update
    fnUpd = zeros(Npar,1);

    for ii=1:N
        %. calc.
        if ( lab(ii) == identID )||( lab(ii) == verifID )
            yy = y(ii);
            LgradB = LgradB_all(ii,:);
            LgradT = LgradT_all(ii,:);

            %. calc. top - correction
            yhat = yhat_pred(ii) + [ LgradB LgradT ] * fnUpd;
            Lnum = yhat - yy;

            %. export
            err_all(ii) = abs(Lnum);
            yhat_all(ii) = yhat;
        end
        
        %. ident.
        if ( lab(ii) == identID )
            chi = sum(LgradB.^2) + sum(LgradT.^2);
            fnUpd = fnUpd - alp * Lnum * [ LgradB LgradT ].'/chi;
        end
    end

    fnB_r = fnB_r + reshape(fnUpd(1:(n*m*p),1),n*m,p);
    fnT_r = fnT_r + fnUpd((n*m*p+1):Npar,1);

    inds = ( lab == verifID );
    RMSE(jj) = sqrt( sum( err_all(inds).^2 )/sum(inds) )/(ymax-ymin);
    t_min_all(jj,:) = min(t_all(inds,:));
    t_max_all(jj,:) = max(t_all(inds,:));

    PCt = corrcoef( y(inds), yhat_all(inds) );
    PC(jj) = PCt(1,2);

    printProgr = 1;
    if ( printProgr == 1 )
        fprintf( '  pass %04.0f out of %04.0f completed, RMSE=%.4f, Pearson=%.4f\n', jj, Nrun, RMSE(jj), PC(jj) );
    end
end

fnB = reshape(fnB_r,n,p*m);
fnT = reshape(fnT_r,q,p);

end
