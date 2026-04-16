
%.   Kolmogorov-Arnold network (KAN) as a machine learning model
%.   Cite paper [Poluektov and Polar, Machine Learning, 114(8):185, 2025]
%.   Code has been written by Michael Poluektov (current affiliation - School of Computing and Mathematical Sciences, University of Greenwich, UK)
%.   Email: m.poluektov@greenwich.ac.uk

%.   The computational example is a synthetic dataset - for each record, the inputs are the coordinates of three points in 2D
%.   and the output is the area of the triangle that is formed by the points. The points belong to the unit square.
%.   Two-layer KAN is built and the RMSE as a function of the iteration number is plotted.
%.   There are four possible combinations of the model variant and the training method. 

%.   The trained model can be used to make a prediction on a new dataset. 
%.   For the spline version of the model, use function 'modelKA_basisC'.
%.   For the piecewise-linear version of the model, use function 'modelKA_linear'.

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

%% numerical parameters

%. numerical damping for iterative parameter update (also called "learning rate")
alp = 0.5;

%. Tikhonov regularisation parameter for Gauss-Newton method 
lam = 1;

%. normalise intermediate variables for Newton-Kaczmarz method (0 or 1, default - 1)
nrmse = 1;

%. number of runs through data (also called "epochs")
Nrun = 100;

%. limits
xmin = 0;
xmax = 1;
ymin = 0;
ymax = 0.5;

%. number of nodes bottom
n = 7;

%. number of nodes top
q = 7;

%. number of bottom blocks, 2*m+1 for classical two-layer KAN
p = 13;

%% build model

tic;

%. initialise
[ fnB0, fnT0 ] = buildKA_init( m, n, q, p, ymin, ymax );

%. build model
modelMethod = 1;
if (modelMethod == 1)

    %. two-layer model; basis functions - cubic splines; training method - Gauss-Newton
    [ yhat_all, fnB, fnT, RMSE, t_min_all, t_max_all ] = solveMinGauss( x, y, lab, identID, verifID, alp, lam, Nrun, xmin, xmax, ymin, ymax, fnB0, fnT0 );

elseif (modelMethod == 2)

    %. two-layer model; basis functions - cubic splines; training method - Newton-Kaczmarz standard
    [ yhat_all, fnB, fnT, RMSE, t_min_all, t_max_all ] = buildKA_basisC( x, y, lab, identID, verifID, alp, nrmse, Nrun, xmin, xmax, ymin, ymax, fnB0, fnT0 );

elseif (modelMethod == 3)

    %. two-layer model; basis functions - cubic splines; training method - Newton-Kaczmarz accelerated
    [ yhat_all, fnB, fnT, RMSE, t_min_all, t_max_all ] = buildKA_basisA( x, y, lab, identID, verifID, alp, nrmse, Nrun, xmin, xmax, ymin, ymax, fnB0, fnT0 );

elseif (modelMethod == 4)

    %. two-layer model; basis functions - piecewise-linear; training method - Newton-Kaczmarz standard
    [ yhat_all, fnB, fnT, RMSE, t_min_all, t_max_all ] = buildKA_linear( x, y, lab, identID, verifID, alp, nrmse, Nrun, xmin, xmax, ymin, ymax, fnB0, fnT0 );

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
ylabel('min/max of intermediate variables');

%% inference using the trained model

if (modelMethod == 1)||(modelMethod == 2)||(modelMethod == 3)

    %. basis functions - cubic splines
    yhat_final = modelKA_basisC( x, xmin, xmax, ymin, ymax, fnB, fnT );

elseif (modelMethod == 4)

    %. basis functions - piecewise-linear
    yhat_final = modelKA_linear( x, xmin, xmax, ymin, ymax, fnB, fnT );

end
