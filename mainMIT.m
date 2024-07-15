
%.   Kolmogorov-Arnold model for machine learning
%.   See (Poluektov and Polar, arXiv:2305.08194, May 2023)
%.   Code has been written by Michael Poluektov (University of Dundee, Department of Mathematical Sciences and Computational Physics)

%.   The computational example is a synthetic dataset - function fitting to compare execution time with (Z. Liu et al., arXiv:2404.19756, 2024)

clear variables;
close all;

%% generate data

N = 2e3;
Nid = 1e3 + 1;

m = 2;
x = 2*rand(N,m)-1;

y = exp( sin(pi*x(:,1)) + x(:,2).^2 );

%. label records to be used for training and validation
lab = ones(N,1);
lab(Nid:end) = 2;
identID = 1;
verifID = 2;

%% numerical param.

%. damping factor for iterative parameter update (also called learning rate)
alp = 1;

%. Tikhonov regularisation parameter for Gauss-Newton method 
lam = 0.01;

%. num. of runs through data
Nrun = 50;

%. limits
xmin = -1;
xmax = 1;
ymin = exp(-1);
ymax = exp(2);

%. num. of nodes bottom
n = 6;

%. num. of nodes top
q = 6;

%. num. of bottom operators, 2*m+1 for classical K.-A.
p = 5;

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
    
    %. basis functions - cubic splines, identification method - Newton-Kaczmarz, standard
    [ yhat_all, fnB, fnT, RMSE, t_min_all, t_max_all ] = buildKA_basisC( x, y, lab, identID, verifID, alp, Nrun, xmin, xmax, ymin, ymax, fnB0, fnT0 );
    
elseif (modelMethod == 3)
    
    %. basis functions - cubic splines, identification method - Newton-Kaczmarz, accelerated
    [ yhat_all, fnB, fnT, RMSE, t_min_all, t_max_all ] = buildKA_basisA( x, y, lab, identID, verifID, alp, Nrun, xmin, xmax, ymin, ymax, fnB0, fnT0 );
    
elseif (modelMethod == 4)

    %. basis functions - piecewise-linear, identification method - Newton-Kaczmarz, standard
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
