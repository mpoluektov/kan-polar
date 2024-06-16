
%.   Kolmogorov-Arnold model for data-driven solution of partial differential equations
%.   See (Poluektov and Polar, arXiv:2305.08194, updated version, June 2024)
%.   Code has been written by Michael Poluektov (University of Dundee, Department of Mathematical Sciences and Computational Physics)

%.   The computational example is a second-order PDE. Sets of boundary and internal points 
%.   are generated and the K.-A. model that approximates the solution is trained. 
%.   The plots show the slices of the analytical and the numerical solutions.

clear variables;
close all;

%% define PDE

%. boundary data
L_b = @(s,u,du,ddu,z) u-z;
dLdu_b = @(s,u,du,ddu,z) 1;
dLddu_b = @(s,u,du,ddu,z) [0 0];
dLdddu_b = @(s,u,du,ddu,z) [0 0 0];

%. first-order PDE
% L_i = @(s,u,du,ddu,z) 2*s(1)*s(2)*du(1)+du(2)-u;
% dLdu_i = @(s,u,du,ddu,z) -1;
% dLddu_i = @(s,u,du,ddu,z) [2*s(1)*s(2) 1];
% dLdddu_i = @(s,u,du,ddu,z) [0 0 0];

%. second-order PDE
L_i = @(s,u,du,ddu,z) ddu(1) + 2*s(1)*s(2)*ddu(2) + ddu(3) + 2*s(1)*du(1) - du(2);
dLdu_i = @(s,u,du,ddu,z) 0;
dLddu_i = @(s,u,du,ddu,z) [2*s(1) -1];
dLdddu_i = @(s,u,du,ddu,z) [1 2*s(1)*s(2) 1];

%. exact solution
uex = @(s) s(:,1).*exp(s(:,2)-s(:,2).^2);

%. fit to exact solution
% L = @(s,u,du,ddu,z) u-s(1)*exp(s(2)-s(2)^2);
% dLdu = @(s,u,du,ddu,z) 1;
% dLddu = @(s,u,du,ddu,z) [0 0];
% dLdddu = @(s,u,du,ddu,z) [0 0 0];

%% init. operators

m = 2;
n = 7;
q = 7;
p = 5;

alp = 0.2;

xmin = 0;
xmax = 2;
ymin = 0;
ymax = 2*exp(0.25);
[ fnB, fnT ] = buildKA_init( m, n, q, p, ymin, ymax );

%. number of passes
Nr = 1e4;

%. number of points per boundary per pass
Nb = 1;

%. number of internal points per pass
Ni = 20;

%% sovle

tic;

%. for first-order PDE
% ovb = ones(2*Nb,1);

%. for second-order PDE
ovb = ones(4*Nb,1);

ovi = ones(Ni,1);

RMSEb = zeros(Nr,1);
RMSEi = zeros(Nr,1);
for jj = 1:Nr

    %. points boundary for first-order PDE
%     xb = zeros(2*Nb,2);
%     xb(1:Nb,1) = xmin*ones(Nb,1);
%     xb(1:Nb,2) = xmin + (xmax-xmin)*rand(Nb,1);
%     xb((Nb+1):(2*Nb),1) = xmin + (xmax-xmin)*rand(Nb,1);
%     xb((Nb+1):(2*Nb),2) = xmin*ones(Nb,1);
%     yb = uex(xb);

    %. points boundary for second-order PDE
    xb = zeros(4*Nb,2);
    xb(1:Nb,1) = xmin*ones(Nb,1);
    xb((Nb+1):(2*Nb),1) = xmax*ones(Nb,1);
    xb(1:(2*Nb),2) = xmin + (xmax-xmin)*rand(2*Nb,1);
    xb((2*Nb+1):(4*Nb),1) = xmin + (xmax-xmin)*rand(2*Nb,1);
    xb((2*Nb+1):(3*Nb),2) = xmin*ones(Nb,1);
    xb((3*Nb+1):(4*Nb),2) = xmax*ones(Nb,1);
    yb = uex(xb);

    %. calc. boundary
    [ dum1, dum2, dum3, fnB_b, fnT_b, RMSEb(jj) ] = buildKA_basisF( xb, yb, ovb, 1, 1, alp, 1, xmin, xmax, ymin, ymax, L_b, dLdu_b, dLddu_b, dLdddu_b, fnB, fnT );
    
    %. points internal
    xi = xmin + (xmax-xmin)*rand(Ni,2);

    %. calc. internal
    [ dum1, dum2, dum3, fnB_i, fnT_i, RMSEi(jj) ] = buildKA_basisF( xi, ovi, ovi, 1, 1, alp, 1, xmin, xmax, ymin, ymax, L_i, dLdu_i, dLddu_i, dLdddu_i, fnB_b, fnT_b );

    fnB = fnB_i;
    fnT = fnT_i;

    printProgr = 1;
    iterSkip = 100;
    if ( printProgr == 1 )&&( rem(jj,iterSkip) == 0 )
        if ( jj > iterSkip )
            fprintf( repmat( '\b', 1, 38 ) );
        end
        fprintf( '  pass %06.0f out of %06.0f completed\n', jj, Nr );
    end
end

toc;

%% save

fn = 'testPDE_2nd';
save( fn, 'fnB', 'fnT', 'RMSEb', 'RMSEi', 'Nr', 'Nb', 'Ni' );

%% plot

figure(1);
hold on;

col = {'r','b','k'};
xs = (0:0.1:2).';
for jj=1:3
    xs(:,2) = jj/2;
    ys_num = modelKA_basisC( xs, xmin, xmax, ymin, ymax, fnB, fnT );
    plot( xs(:,1), ys_num, '-', 'Color', col{jj} );
    ys_an = uex(xs);
    plot( xs(:,1), ys_an, 's', 'Color', col{jj} );
end

hold off;
