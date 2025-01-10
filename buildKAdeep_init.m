function [ fnB, fnM, fnT ] = buildKAdeep_init( m, n, h, q, p, r, ymin, ymax )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

%. limits
tmin = ymin;
tmax = ymax;
smin = ymin;
smax = ymax;

%. init. operators
amplB = (tmax-tmin)/m;
amplM = (smax-smin)/p;
amplT = (ymax-ymin)/r;
fnB = tmin/m + rand( n, m*p ) * amplB;
fnM = smin/p + rand( h, p*r ) * amplM;
fnT = ymin/r + rand( q, r ) * amplT;

end
