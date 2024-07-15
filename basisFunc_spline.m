function [ val, dval, ddval, dddval ] = basisFunc_spline( xx, xmin, xmax, n, M )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

%. xx must be a row
tt = (n-1)*(xx-xmin)/(xmax-xmin);

ind = floor(tt)+1;
ind(ind<=0) = 1;
ind(ind>=n) = n-1;

I = eye(n);
QL = I(:,ind);
QR = I(:,ind+1);
MQL = M(:,ind);
MQR = M(:,ind+1);
T = repmat( tt-ind+1, n, 1 );
ZL = MQL - QR + QL;
ZR = -MQR + QR - QL;
ZI = (1-T).*ZL + T.*ZR;

V = (1-T).*QL + T.*QR + T.*(1-T).*ZI;
val = V;
if (nargout > 1)
    dV = (1-2*T).*ZI + T.*(1-T).*(ZR-ZL) + QR - QL;
    dval = dV*(n-1)/(xmax-xmin);
end
if (nargout > 2)
    ddV = 2*(1-2*T).*(ZR-ZL) - 2*ZI;
    ddval = ddV*(n-1)^2/(xmax-xmin)^2;
end
if (nargout > 3)
    dddV = -6*(ZR-ZL);
    dddval = dddV*(n-1)^3/(xmax-xmin)^3;
end

end
