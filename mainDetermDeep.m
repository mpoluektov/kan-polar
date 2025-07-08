
%.   Kolmogorov-Arnold model for machine learning
%.   This code demonstrates building three-layered (deep) model; an example for the classical version (two layers - inner and outer functions) can be found in mainTriang.m
%.   See (Poluektov and Polar, arXiv:2305.08194, May 2023)
%.   Code has been written by Michael Poluektov (University of Dundee, Department of Mathematical Sciences and Computational Physics)

%.   The computational example is a synthetic dataset - for each record, the inputs are the components 
%.   of a 4x4 matrix and the output is the determinant of the matrix.
%.   K.-A. regression model is built and the RMSE as a function of the iteration number is plotted.
%.   There are three possible combinations of the model variant and the identification method. 

clear variables;
close all;

%% generate data

%. total number of input-output records - training and validation
N = 1.2e5;

%. number of records to be used for training
Nid = 1e5 + 1;

%. dimensionality
dim = 4;

%. inputs
m = dim^2;
x = rand(N,m);

%. outputs
y = zeros(N,1);
for ii=1:N
    xx = reshape(x(ii,:).',dim,dim);
    y(ii) = det(xx);
end

%. label records to be used for training and validation
lab = ones(N,1);
lab(Nid:end) = 2;
identID = 1;
verifID = 2;

%% numerical param.

%. damping factor for iterative parameter update (also called learning rate)
alp = 0.5;

%. damping factor for pre-training
alppre = 0.2;

%. Tikhonov regularisation parameter for Gauss-Newton method 
lam = 1;

%. num. of runs through data for main training
Nrun = 30;

%. num. of runs through data for pre-training
Npre = 8;

%. limits
xmin = 0;
xmax = 1;
ymin = min(y);
ymax = max(y);

%. num. of nodes bottom
n = 4;

%. num. of nodes middle
h = 7;

%. num. of nodes top
q = 10;

%. num. of bottom operators
p = 16;

%. num. of middle operators
r = 8;

%% build K.-A.

tic;

%. initialise and/or pre-train
modelPre = 2;
if (modelPre == 0)

    %. initialise randomly
    [ fnB0, fnM0, fnT0 ] = buildKAdeep_init( m, n, h, q, p, r, ymin, ymax );

elseif (modelPre == 1)

    %. initialise pre-training
    [ fnB0_pre1, fnT0_pre1 ] = buildKA_init( m, n, h, p, ymin, ymax );
    [ fnB0_pre2, fnT0_pre2 ] = buildKA_init( p, h, q, r, ymin, ymax );

    %. pre-training for three-layer model, basis functions - cubic splines, identification method - Newton-Kaczmarz, standard
    [ yhat_all_pre1, fnB0, fnTdum, RMSE_pre1, t_min_all_pre1, t_max_all_pre1, t_all_pre1 ] = buildKA_basisC( x, y, lab, identID, verifID, alppre, Npre, xmin, xmax, ymin, ymax, fnB0_pre1, fnT0_pre1 );
    [ yhat_all_pre2, fnM0, fnT0, RMSE_pre2, t_min_all_pre2, t_max_all_pre2, t_all_pre2 ] = buildKA_basisC( t_all_pre1, y, lab, identID, verifID, alppre, Npre, ymin, ymax, ymin, ymax, fnB0_pre2, fnT0_pre2 );
    
elseif (modelPre == 2)

    %. initialise pre-training
    [ fnB0_pre1, fnT0_pre1 ] = buildKA_init( m, n, h, p, ymin, ymax );
    [ fnB0_pre2, fnT0_pre2 ] = buildKA_init( p, h, q, r, ymin, ymax );

    %. pre-training for three-layer model, basis functions - piecewise-linear, identification method - Newton-Kaczmarz, standard
    [ yhat_all_pre1, fnB0, fnTdum, RMSE_pre1, t_min_all_pre1, t_max_all_pre1, t_all_pre1 ] = buildKA_linear( x, y, lab, identID, verifID, alppre, Npre, xmin, xmax, ymin, ymax, fnB0_pre1, fnT0_pre1 );
    [ yhat_all_pre2, fnM0, fnT0, RMSE_pre2, t_min_all_pre2, t_max_all_pre2, t_all_pre2 ] = buildKA_linear( t_all_pre1, y, lab, identID, verifID, alppre, Npre, ymin, ymax, ymin, ymax, fnB0_pre2, fnT0_pre2 );

end

%. build model
modelMethod = 7;
if (modelMethod == 5)
    
    %. three-layer model, basis functions - cubic splines, identification method - Gauss-Newton
    [ yhat_all, fnB, fnM, fnT, RMSE, t_min_all, t_max_all, s_min_all, s_max_all ] = solveMinGaussDeep( x, y, lab, identID, verifID, alp, lam, Nrun, xmin, xmax, ymin, ymax, fnB0, fnM0, fnT0 );
    
elseif (modelMethod == 6)

    %. three-layer model, basis functions - cubic splines, identification method - Newton-Kaczmarz, accelerated
    [ yhat_all, fnB, fnM, fnT, RMSE, t_min_all, t_max_all, s_min_all, s_max_all ] = buildKAdeep_basisA( x, y, lab, identID, verifID, alp, Nrun, xmin, xmax, ymin, ymax, fnB0, fnM0, fnT0 );
    
elseif (modelMethod == 7)

    %. three-layer model, basis functions - piecewise-linear, identification method - Newton-Kaczmarz, standard
    [ yhat_all, fnB, fnM, fnT, RMSE, t_min_all, t_max_all, s_min_all, s_max_all ] = buildKAdeep_linear( x, y, lab, identID, verifID, alp, Nrun, xmin, xmax, ymin, ymax, fnB0, fnM0, fnT0 );

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
ylabel('min/max of first hidden var.');

figure(3);
hold on;
for jj=1:r
    plot(s_min_all(:,jj),'b');
    plot(s_max_all(:,jj),'r');
end
hold off;
xlabel('number of passes');
ylabel('min/max of second hidden var.');
