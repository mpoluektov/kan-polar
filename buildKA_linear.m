function [ yhat_all, fnB, fnT, RMSE, t_min_all, t_max_all ] = buildKA_linear( x, y, lab, identID, verifID, alp, Nrun, xmin, xmax, ymin, ymax, fnB0, fnT0 )
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

toLinB = n*(0:1:(p*m-1));
toLinT = q*(0:1:(p-1));

for jj=1:Nrun
    for ii=1:N
        %. calc.
        if ( lab(ii) == identID )||( lab(ii) == verifID )

            %. calc. bottom
            b = 1 + (n-1) * ( x(ii,:) - xmin )/( xmax - xmin );
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
        
            kL_ex = repmat( kL, 1, p );
            kR_ex = repmat( kR, 1, p );
            psi_ex = repmat( psi, 1, p );
            om_psi_ex = repmat( om_psi, 1, p );
            
            indL = kL_ex + toLinB;
            indR = kR_ex + toLinB;
    
            outB = om_psi_ex .* fnB(indL) + psi_ex .* fnB(indR);
            outB_r = reshape( outB, m, [] );
            t = sum( outB_r, 1 );
            t_all(ii,:) = t;
        
            %. calc. top
            c = 1 + (q-1) * ( t - tmin )/( tmax - tmin );
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
    
            idxL = vL + toLinT;
            idxR = vR + toLinT;
            
            outT = om_phi .* fnT(idxL) + phi .* fnT(idxR);
            yhat = sum(outT);
            yhat_all(ii) = yhat;
        
            %. error
            D = y(ii) - yhat;
            err_all(ii) = abs(D);
        end
    
        %. ident.
        if ( lab(ii) == identID )
            
            %. ident. bottom
            coefB = ( fnT(idxR) - fnT(idxL) ) * (q-1) / ( tmax - tmin );
            coefB_e = repmat( coefB, m, 1 );
            coefB_er = reshape( coefB_e, 1, [] );
            chiB = sum( (coefB_er .* om_psi_ex).^2 + (coefB_er .* psi_ex).^2 );
            chiT = sum( om_phi.^2 + phi.^2 );
            chiAll = chiB + chiT;
            fnB(indL) = fnB(indL) + alp * D * (coefB_er .* om_psi_ex)/chiAll;
            fnB(indR) = fnB(indR) + alp * D * (coefB_er .* psi_ex)/chiAll;

            %. ident. top
            fnT(idxL) = fnT(idxL) + alp * D * om_phi/chiAll;
            fnT(idxR) = fnT(idxR) + alp * D * phi/chiAll;
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

end
