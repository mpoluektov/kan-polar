function [ fnB ] = normBlock( fnB0, m, tmin, tmax )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

n = size(fnB0,1);
p = size(fnB0,2)/m;

min_fnB = min(fnB0);
min_fnB_r = reshape( min_fnB, m, p );
minB = sum(min_fnB_r);
minB_e = repmat( reshape(repmat(minB,m,1),1,p*m), n, 1 );
max_fnB = max(fnB0);
max_fnB_r = reshape( max_fnB, m, p );
maxB = sum(max_fnB_r);
maxB_e = repmat( reshape(repmat(maxB,m,1),1,p*m), n, 1 );

fnB = tmin/m + ( fnB0 - minB_e/m )*( tmax - tmin )./( maxB_e - minB_e );

end
