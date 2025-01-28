function [ yhat_all, fnB, fnM, fnT, RMSE, t_min_all, t_max_all, s_min_all, s_max_all ] = solveMinGaussDeep( x, y, lab, identID, verifID, alp, lam, Nrun, xmin, xmax, ymin, ymax, fnB0, fnM0, fnT0 )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

m = size(x,2);

fnB = fnB0;
fnM = fnM0;
fnT = fnT0;
n = size(fnB,1);
h = size(fnM,1);
q = size(fnT,1);
p = size(fnB,2)/m;
r = size(fnM,2)/p;

RMSE = zeros(Nrun,1);
PC = zeros(Nrun,1);
t_min_all = zeros(Nrun,p);
t_max_all = zeros(Nrun,p);
s_min_all = zeros(Nrun,r);
s_max_all = zeros(Nrun,r);

for jj=1:Nrun
    
    indI = ( lab == identID );
    indV = ( lab == verifID );

    %. training
    [ yhat_all, LgradB_all, LgradM_all, LgradT_all ] = modelKAdeep_basisC( x(indI,:), xmin, xmax, ymin, ymax, fnB, fnM, fnT );
    L_all = yhat_all - y(indI,:);
    
    F = L_all;
    J = [ LgradB_all LgradM_all LgradT_all ];
    A = J.' * J;
    b = J.' * F;
    Ar = A + lam*eye(n*p*m+h*r*p+q*r);

    dlt = -Ar\b;
    dltB = reshape(dlt(1:(n*p*m)),n,[]);
    dltM = reshape(dlt((n*p*m+1):(n*p*m+h*r*p)),h,[]);
    dltT = reshape(dlt((n*p*m+h*r*p+1):end),q,[]);

    fnB = fnB + alp*dltB;
    fnM = fnM + alp*dltM;
    fnT = fnT + alp*dltT;

    %. validation
    [ yhat_all, dum1, dum2, dum3, t_min, t_max, s_min, s_max ] = modelKAdeep_basisC( x(indV,:), xmin, xmax, ymin, ymax, fnB, fnM, fnT );
    err_all = abs( yhat_all - y(indV,:) );
    RMSE(jj) = sqrt( mean( err_all.^2 ) )/(ymax-ymin);
    t_min_all(jj,:) = t_min;
    t_max_all(jj,:) = t_max;
    s_min_all(jj,:) = s_min;
    s_max_all(jj,:) = s_max;
    
    PCt = corrcoef( y(indV,:), yhat_all );
    PC(jj) = PCt(1,2);

    printProgr = 1;
    if ( printProgr == 1 )
        fprintf( '  pass %04.0f out of %04.0f completed, RMSE=%.4f, Pearson=%.4f\n', jj, Nrun, RMSE(jj), PC(jj) );
    end
end

end
