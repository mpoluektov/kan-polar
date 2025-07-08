
%.   Kolmogorov-Arnold model for machine learning
%.   This code demonstrates building a model with a vector output; an example for the single output can be found in mainTriang.m
%.   See (Poluektov and Polar, arXiv:2305.08194, May 2023)
%.   Code has been written by Michael Poluektov (University of Dundee, Department of Mathematical Sciences and Computational Physics)

%.   The computational example is a synthetic dataset - for each record, the inputs are the coordinates of three points in 2D
%.   and the output are the lengths of the medians of the triangle that is formed by the points. The points belong to unit square.
%.   K.-A. regression model is built and the RMSE as a function of the iteration number is plotted.
%.   There is one possible combination of the model variant and the identification method (so far). 

clear variables;
close all;

%% generate data

%. total number of input-output records - training and validation
N = 1.2e4;

%. number of records to be used for training
Nid = 1e4 + 1;

%. number of inputs
m = 6;

%. inputs
x = rand(N,m);

%. outputs
sides = [ sqrt( (x(:,3)-x(:,1)).^2 + (x(:,4)-x(:,2)).^2 )  sqrt( (x(:,5)-x(:,3)).^2 + (x(:,6)-x(:,4)).^2 )  sqrt( (x(:,1)-x(:,5)).^2 + (x(:,2)-x(:,6)).^2 ) ];
ma = sqrt( 2*sides(:,2).^2 + 2*sides(:,3).^2 - sides(:,1).^2 )/2;
mb = sqrt( 2*sides(:,3).^2 + 2*sides(:,1).^2 - sides(:,2).^2 )/2;
mc = sqrt( 2*sides(:,1).^2 + 2*sides(:,2).^2 - sides(:,3).^2 )/2;
y = [ ma mb mc ];

%. number of outputs
r = size(y,2);

%. label records to be used for training and validation
lab = ones(N,1);
lab(Nid:end) = 2;
identID = 1;
verifID = 2;

%% numerical param.

%. damping factor for iterative parameter update (also called learning rate)
alp = 0.5;

%. Tikhonov regularisation parameter for Gauss-Newton method 
lam = 1;

%. num. of runs through data
Nrun = 100;

%. limits
xmin = 0;
xmax = 1;
ymin = 0;
ymax = sqrt(2);

%. num. of nodes bottom
n = 7;

%. num. of nodes top
q = 10;

%. num. of bottom operators, 2*m+1 for classical K.-A.
p = 13;

%% build K.-A.

tic;

%. initialise
[ fnB0, fnT0 ] = buildKAvect_init( m, n, q, p, r, ymin, ymax );

%. build model
modelMethod = 8;
if (modelMethod == 8)
    
    %. basis functions - piecewise-linear, identification method - Newton-Kaczmarz, standard
    [ yhat_all, fnB, fnT, RMSE, t_min_all, t_max_all ] = buildKAvect_linear( x, y, lab, identID, verifID, alp, Nrun, xmin, xmax, ymin, ymax, fnB0, fnT0 );

end

toc;

%% plot

figure(1);
hold on;
plot(log(RMSE)/log(10));
hold off;
xlabel('number of passes');
ylabel('log_{10}(RMSE)');

figure(2);
hold on;
for jj=1:p
    plot(t_min_all(:,jj),'b');
    plot(t_max_all(:,jj),'r');
end
hold off;
xlabel('number of passes');
ylabel('min/max of intermediate var.');
