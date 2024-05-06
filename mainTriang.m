
%.   Kolmogorov-Arnold model for machine learning
%.   See (Poluektov and Polar, arXiv:2305.08194, May 2023)
%.   Code has been written by Michael Poluektov (University of Dundee, Department of Mathematical Sciences and Computational Physics)

%.   The computational example is a synthetic dataset - for each record, the inputs are the coordinates of three points in 2D
%.   and the output is the area of the triangle that is formed by the points. The points belong to unit square.
%.   K.-A. regression model is built and the RMSE as a function of the iteration number is plotted.
%.   There are three possible combinations of the model variant and the identification method. 

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
hper = ( sides(:,1) + sides(:,2) + sides(:,3) )/2;
y = sqrt( hper .* ( hper - sides(:,1) ) .* ( hper - sides(:,2) ) .* ( hper - sides(:,3) ) );

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
ymax = 0.5;

%. num. of nodes bottom
n = 7;

%. num. of nodes top
q = 7;

%. num. of bottom operators, 2*m+1 for classical K.-A.
p = 13;

%% build K.-A.

tic;

%. initialise
[ fnB0, fnT0 ] = buildKA_init( m, n, q, p, ymin, ymax );

%. build model
modelMethod = 1;
if (modelMethod == 1)
    
    %. basis functions - cubic splines, identification method - Gauss-Newton
    [ yhat_all, fnB, fnT, RMSE, t_min_all, t_max_all ] = solveMinGauss( x, y, lab, identID, verifID, alp, lam, Nrun, xmin, xmax, ymin, ymax, fnB0, fnT0 );
    
elseif (modelMethod == 2)
    
    %. basis functions - cubic splines, identification method - Newton-Kaczmarz
    [ yhat_all, fnB, fnT, RMSE, t_min_all, t_max_all ] = buildKA_basisC( x, y, lab, identID, verifID, alp, Nrun, xmin, xmax, ymin, ymax, fnB0, fnT0 );
    
elseif (modelMethod == 3)

    %. basis functions - piecewise-linear, identification method - Newton-Kaczmarz
    [ yhat_all, fnB, fnT, RMSE, t_min_all, t_max_all ] = buildKA_linear( x, y, lab, identID, verifID, alp, Nrun, xmin, xmax, ymin, ymax, fnB0, fnT0 );

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
