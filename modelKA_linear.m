function [ yhat_all ] = modelKA_linear( x, xmin, xmax, ymin, ymax, fnB, fnT )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

y = zeros(size(x,1),1);
lab = ones(size(x,1),1);
identID = 0;
verifID = 1;
yhat_all = buildKA_linear( x, y, lab, identID, verifID, 0, 1, xmin, xmax, ymin, ymax, fnB, fnT );

end

