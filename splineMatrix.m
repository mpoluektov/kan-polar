function [ M ] = splineMatrix( n )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

e = ones(n,1);
A = spdiags([e 4*e e],[-1 0 1],n,n);
B = spdiags([-3*e 3*e],[-1 1],n,n);

%. natural
% A(1,1) = 2;
% A(n,n) = 2;
% B(1,1) = -3;
% B(n,n) = 3;
%. not-a-knot
A(1,[1 2 3]) = [1 0 -1];
A(n,[n-2 n-1 n]) = [1 0 -1];
B(1,[1 2 3]) = [-2 4 -2];
B(n,[n-2 n-1 n]) = [-2 4 -2];

M = full(A\B).';

end
