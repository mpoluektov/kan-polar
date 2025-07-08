function [ fnB, fnT ] = buildKAvect_init( m, n, q, p, r, ymin, ymax )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

%. limits
tmin = ymin;
tmax = ymax;

%. init. operators
amplB = (tmax-tmin)/m;
amplT = (ymax-ymin)/p;
fnB = tmin/m + rand( n, m*p ) * amplB;
fnT = ymin/p + rand( q, p*r ) * amplT;

end
