function BO_uncertainty_demo()
% A simple 1d numerical illustration for the posterior of the maximiser of the objective function f 
% in Bayesian optimisation (BO).
% 
% Suppose, as typically in BO, that unknown function f to minimise is modelled using GP prior.
% Here we compute by simulation the posterior of
%
% 1) argmax_x f(x),   2) max_x f(x)
%
% given some n observations of the function f i.e. D = {(f(x_i),x_i)}_{i=1}^n in a simple 1d case.
%
% Quantity 1 are used as quantitity-of-interest in Entropy Search BO method (Entropy Search for 
% Information-Efficient Global Optimization 2013 JMLR) while Max-value entropy search BO method 
% (Max-value Entropy Search for Efficient Bayesian Optimization 2017 ICML) is based on quantity 2.
%
% The illustration here (for quantity 1) is similar to Figure 2 of the above Entropy Search paper. 

close all; 
%rng(12345);

% select the true function f, parameter x
n_grid = 1000;
x_grid = linspace(-10,10,n_grid)';
sigma_n = 0.01; % stdev of the Gaussian observation error
%sigma_n = 3;
f_true = @(x)0.25*abs(x).*sin(x./2 + 1).^2 - 1; % true objective function (that we try to minimise in BO)
x_bds = [x_grid(1); x_grid(end)];
f_grid = f_true(x_grid);
[f_min_true,x_min_ind] = min(f_grid);
x_min = x_grid(abs(f_grid-f_grid(x_min_ind))<=10*eps); % do this better...

% The locations where the evaluations are made. In BO, these would be selected iteratively using 
% some acquisition function based on the GP model to minimise the queries of the function f.
% Here our goal is to simply visualise the uncertainty of 1) argmax of the unknown function f, 2)
% max of f, so no acquisition functions etc. are implemented here.
n_tr = 8;
%n_tr = 20;
x_tr = x_bds(1)+(x_bds(2)-x_bds(1))*rand(n_tr,1); % select points uniformly...
sigma_n_true = sigma_n;
y_tr = f_true(x_tr) + sigma_n_true*randn(size(x_tr)); % evaluate function f at these points

% set up and fit the GP (uses zero mean GP with squared-exp cov and fixed GP hypers)
l = 1.75;
sigma_f = 0.25;
A = l^2;
invA = 1/A;

% GP mean and variance for predicting f at test points x_grid
K = sqexp(x_tr,x_tr,invA,sigma_f) + (sigma_n^2 + 1e-9)*eye(n_tr);
L = chol(K,'lower');
a = L'\(L\y_tr);
kx = sqexp(x_grid,x_tr,invA,sigma_f);
m_f_tr = kx*a;
c_f_tr = sqexp(x_grid,x_grid,invA,sigma_f) - kx*(L'\(L\kx'));
v_f_tr = diag(c_f_tr); 
s_f_tr = sqrt(v_f_tr);

% compute the posterior of x_s = argmax_x f(x) and f_s = max_x f(x) using simulation and a tight 
% discretization of the x-space.
nsimul = 2000; % how many simuls from GP
jitter = 1e-9;
L_grid = chol(c_f_tr + jitter*eye(size(c_f_tr)),'lower');
rr = randn(n_grid,nsimul);
f_draws = NaN(n_grid,nsimul);
xss = NaN(1,nsimul);
fss = NaN(1,nsimul);
for i = 1:nsimul
    f_draws(:,i) = m_f_tr + L_grid*rr(:,i); % simulate sample path from GP
    % minimise the sample path using the grid approximation
    [fss(i),xmin_indi] = min(f_draws(:,i));
    xss(i) = x_grid(xmin_indi);
end


%% 1/3: plot true function and GP fit
figure(1);
set(gcf,'Position',[100 100 1800 900]);
subplot(3,3,1:3); 
hold on;
h(1) = plot(x_grid,f_grid,'-k'); % true function 
plot(x_bds,f_min_true*[1,1],'--g'); % min f line
lw = 1; truecol = 'b';
h(2) = plot(x_grid,m_f_tr,['-',truecol],'LineWidth',lw); % GP mean
plot(x_grid,m_f_tr + 1.96*s_f_tr,['--',truecol],'LineWidth',lw); % gp upper 95% CI
plot(x_grid,m_f_tr - 1.96*s_f_tr,['--',truecol],'LineWidth',lw); % gp lower 95% CI
plot(x_tr,y_tr,'*k','MarkerSize',10); % function evaluations
hold off;
xlabel('x'); ylabel('f(x)');
legend(h,{'true function f','GP mean'},'Location','northwest');
box on;
fl = ylim();

%% 2/3: visualise the posterior of argmax_x f(x)
subplot(3,3,4:6);
xss_f = ksdensity(xss,x_grid,'Bandwidth',l/10); % bandwidth fixed to a certain value
hold on;
plot(x_grid,xss_f,'-r');
xlim(x_bds);
ylim([0,max(xss_f)]);
xlabel('x');
set(gca,'ytick',[]);
title('posterior of argmax_x f(x)');
plot(x_min,zeros(size(x_min)),'kx','MarkerSize',16); % true minimiser(s)
plot(xss,zeros(size(xss)),'k+'); % simulated minimisers
hold off;
box on;

%% 3/3 visualise the posterior of max_x f(x)
subplot(3,3,8);
fss_f = ksdensity(fss,x_grid);
hold on;
plot(x_grid,fss_f,'-r');
plot(fss,zeros(size(fss)),'k+'); % simulated minimum points
hold off;
xlim(fl);
ylim([0,max(fss_f)]);
xlabel('f(x)');
set(gca,'ytick',[]);
title('posterior of max_x f(x)');
box on;
end




