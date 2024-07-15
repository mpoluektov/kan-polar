function [ yhat_all, fnB, fnT, RMSE, t_min_all, t_max_all, LgradB_all, LgradT_all ] = buildKA_basisC( x, y, lab, identID, verifID, alp, Nrun, xmin, xmax, ymin, ymax, fnB0, fnT0 )
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
t_all = zeros(N,p);
yhat_all = zeros(N,1);
RMSE = zeros(Nrun,1);
t_min_all = zeros(Nrun,p);
t_max_all = zeros(Nrun,p);
LgradB_all = zeros(N,n*p*m);
LgradT_all = zeros(N,q*p);

%. proj. matrix
Cpq = kron(eye(p),ones(q,1));

Mn = splineMatrix(n);
Mq = splineMatrix(q);

fnB_r = reshape(fnB,n*m,p);
fnT_r = fnT(:);

for jj=1:Nrun
    for ii=1:N
        %. calc.
        if ( lab(ii) == identID )||( lab(ii) == verifID )
            xx = x(ii,:);
            yy = y(ii);

            %. calc. bottom
            [ phi, dphi, ddphi ] = basisFunc_spline( xx, xmin, xmax, n, Mn );
            t = phi(:).'*fnB_r;

            %. calc. top
            [ psi, dpsi, ddpsi, dddpsi ] = basisFunc_spline( t, tmin, tmax, q, Mq );
            yhat = psi(:).'*fnT_r;
            Lnum = yhat - yy;

            %. deriv.
            dpsiEx = diag(dpsi(:)) * Cpq;
            top = fnT_r.' * dpsiEx;
            der = phi(:) * top;
            LgradB = der(:).';
            LgradT = psi(:).';

            %. export
            err_all(ii) = abs(Lnum);
            t_all(ii,:) = t;
            yhat_all(ii) = yhat;
            LgradB_all(ii,:) = LgradB;
            LgradT_all(ii,:) = LgradT;
        end
        
        %. ident.
        if ( lab(ii) == identID )
            chi = sum(LgradB.^2) + sum(LgradT.^2);
            fnB_r = fnB_r - alp * Lnum * reshape(LgradB,n*m,p)/chi;
            fnT_r = fnT_r - alp * Lnum * LgradT.'/chi;
        end
    end

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
