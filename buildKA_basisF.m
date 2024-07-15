function [ yhat_all, dyhat_all, ddyhat_all, fnB, fnT, RMSE, t_min_all, t_max_all, L_all, LgradB_all, LgradT_all ] = buildKA_basisF( x, y, lab, identID, verifID, alp, Nrun, xmin, xmax, ymin, ymax, L, dLdu, dLddu, dLdddu, fnB0, fnT0 )
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
dyhat_all = zeros(N,m);
ddyhat_all = zeros(N,m*(m+1)/2);
RMSE = zeros(Nrun,1);
t_min_all = zeros(Nrun,p);
t_max_all = zeros(Nrun,p);
L_all = zeros(N,1);
LgradB_all = zeros(N,n*p*m);
LgradT_all = zeros(N,q*p);

%. proj. matrices
%. second deriv. order (11,12,..,1m,22,23,..,2m,..,mm)
Cmn = kron(eye(m),ones(n,1));
Cpq = kron(eye(p),ones(q,1));
ins = reshape(tril(true(m)),m^2,1);
Prows = kron(eye(m),ones(m,1));
Prows(~ins,:) = [];
Pcols = kron(ones(m,1),eye(m));
Pcols(~ins,:) = [];
Pdiag = Prows.*Pcols;
PexR = eye(m^2);
PexR(~ins,:) = [];
transpInd = reshape(reshape(1:(m^2),m,m).',m^2,1);
PexL = PexR(:,transpInd);
Pexcl = PexL + PexR;

Mn = splineMatrix(n);
Mq = splineMatrix(q);

fnB_r = reshape(fnB,n*m,p);
fnT_r = fnT(:);

d_dyhat_d_fnB = zeros(m,n*p*m);
d_dyhat_d_fnT = zeros(m,q*p);
d_ddyhat_d_fnB = zeros(m*(m+1)/2,n*p*m);
d_ddyhat_d_fnT = zeros(m*(m+1)/2,q*p);

for jj=1:Nrun
    for ii=1:N
        %. calc.
        if ( lab(ii) == identID )||( lab(ii) == verifID )
            xx = x(ii,:);
            yy = y(ii);

            %. calc. Bottom
            [ phi, dphi, ddphi ] = basisFunc_spline( xx, xmin, xmax, n, Mn );
            t = phi(:).'*fnB_r;

            %. calc. Top
            [ psi, dpsi, ddpsi, dddpsi ] = basisFunc_spline( t, tmin, tmax, q, Mq );
            yhat = psi(:).'*fnT_r;

            %. deriv. first
            dphiEx = diag(dphi(:)) * Cmn;
            ddphiEx = diag(ddphi(:)) * Cmn;
            dpsiEx = diag(dpsi(:)) * Cpq;
            ddpsiEx = diag(ddpsi(:)) * Cpq;
            dddpsiEx = diag(dddpsi(:)) * Cpq;
            bot = fnB_r.' * dphiEx;
            dbot = fnB_r.' * ddphiEx;
            top = fnT_r.' * dpsiEx;
            dtop = fnT_r.' * ddpsiEx;
            ddtop = fnT_r.' * dddpsiEx;
            dyhat = top * bot;
            der = phi(:) * top;
            d_yhat_d_fnB = der(:).';
            d_yhat_d_fnT = psi(:).';
            dtb = repmat(dtop,m,1) .* bot.';
            dtdb = repmat(dtop,m,1) .* dbot.';
            
            evalFirstGrad = 1;
            if ( evalFirstGrad == 1 )
                aa1 = kron( dtb, phi(:).' );
                aa2 = kron( top, dphiEx.' );
                d_dyhat_d_fnB = aa1 + aa2;
                d_dyhat_d_fnT = bot.' * dpsiEx.';
            end
            
            %. deriv. second
            bb = ( Prows * bot.' ).*( Pcols * bot.' );
            ddtbb = repmat(ddtop,m*(m+1)/2,1) .* bb;
            ddyhat = top * dbot * Pdiag.' + dtop * bb.';
            
            evalSecondGrad = 1;
            if ( evalSecondGrad == 1 )
                cc1 = kron( dtb, dphiEx.' );
                cc2 = kron( top, ddphiEx.' );
                cc3 = kron( dtdb, phi(:).' );
                cc4 = kron( ddtbb, phi(:).' );
                d_ddyhat_d_fnB = Pexcl * cc1 + Pdiag * cc2 + Pdiag * cc3 + cc4;
                d_ddyhat_d_fnT = bb * ddpsiEx.' + Pdiag * dbot.' * dpsiEx.';
            end

            %. eqs.
            Lnum = L(xx,yhat,dyhat,ddyhat,yy);
            LgradB = dLdu(xx,yhat,dyhat,ddyhat,yy) * d_yhat_d_fnB + dLddu(xx,yhat,dyhat,ddyhat,yy) * d_dyhat_d_fnB + dLdddu(xx,yhat,dyhat,ddyhat,yy) * d_ddyhat_d_fnB;
            LgradT = dLdu(xx,yhat,dyhat,ddyhat,yy) * d_yhat_d_fnT + dLddu(xx,yhat,dyhat,ddyhat,yy) * d_dyhat_d_fnT + dLdddu(xx,yhat,dyhat,ddyhat,yy) * d_ddyhat_d_fnT;
            
            %. export
            err_all(ii) = abs(Lnum);
            t_all(ii,:) = t;
            yhat_all(ii) = yhat;
            dyhat_all(ii,:) = dyhat;
            ddyhat_all(ii,:) = ddyhat;
            L_all(ii) = Lnum;
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
    RMSE(jj) = sqrt( sum( err_all(inds).^2 )/sum(inds) );
    t_min_all(jj,:) = min(t_all(inds,:));
    t_max_all(jj,:) = max(t_all(inds,:));

    printProgr = 0;
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
