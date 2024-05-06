function [ yhat_all, fnB, fnT, RMSE, t_min_all, t_max_all ] = solveMinGauss( x, y, lab, identID, verifID, alp, lam, Nrun, xmin, xmax, ymin, ymax, fnB0, fnT0 )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

m = size(x,2);

fnB = fnB0;
fnT = fnT0;
n = size(fnB,1);
q = size(fnT,1);
p = size(fnT,2);

RMSE = zeros(Nrun,1);
t_min_all = zeros(Nrun,p);
t_max_all = zeros(Nrun,p);

for jj=1:Nrun
    
    [ yhat_all, dum2, dum3, dum4, dum5, dum6, LgradB_all, LgradT_all ] = buildKA_basisC( x, y, lab, 0, identID, 0, 1, xmin, xmax, ymin, ymax, fnB, fnT );
    L_all = yhat_all - y;
    
    ind = (lab == identID);
    F = L_all(ind,:);
    J = [ LgradB_all(ind,:) LgradT_all(ind,:) ];
    A = J.' * J;
    b = J.' * F;
    Ar = A + lam*eye(n*p*m+q*p);

    dlt = -Ar\b;
    dltB = reshape(dlt(1:(n*p*m)),n,[]);
    dltT = reshape(dlt((n*p*m+1):end),q,[]);

    fnB = fnB + alp*dltB;
    fnT = fnT + alp*dltT;

    [ yhat_all, dum2, dum3, RMSE(jj), t_min_all(jj,:), t_max_all(jj,:) ] = buildKA_basisC( x, y, lab, 0, verifID, 0, 1, xmin, xmax, ymin, ymax, fnB, fnT );
    
    printProgr = 1;
    if ( printProgr == 1 )
        if ( jj > 1 )
            fprintf( repmat( '\b', 1, 34 ) );
        end
        fprintf( '  pass %04.0f out of %04.0f completed\n', jj, Nrun );
    end
end

end
