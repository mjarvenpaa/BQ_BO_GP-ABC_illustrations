function [] = simple_bq()
% A simple BQ-implementation and test case in 1d. 
%
% We compute probabilistically the integral 
%
% I_1 = int{f(x)pi(x)}dx, 
%
% where pi(x) is a known Gaussian density, f is assumed costly to evaluate and where the 
% integration is over the real numbers. 
%
% In the second part, we consider the ratio of integrals i.e. the formula 
%
% I_2 = [int{xf(x)pi(x)}dx] / [int{f(x)pi(x)}dx].
%

close all;
%rng(12345);

%% INTEGRAL I_1:
%%%%%%%%%%%%%%%%

% function f and density pi
n_grid = 500;
x_grid = linspace(0,10,n_grid)';
mu = 4;
f_true = @(x)exp(-0.5*(x-mu).^2/0.5^2)/2; % true (likelihood) function f(x)
x_bds = [x_grid(1); x_grid(end)];

b = 5; % mean and variance for the gaussian density pi(x)
B = 1.5^2;
invB = 1/B;

% select the locations and evaluate the (noisy and expensive) function f
n_tr = 20;
x_tr = x_bds(1)+(x_bds(2)-x_bds(1))*rand(n_tr,1);
sigma_n_true = 0.001;
y_tr = f_true(x_tr) + sigma_n_true*randn(size(x_tr));


% set up and fit the GP (uses zero mean GP with squared-exp cov and fixed GP hypers)
l = 1;
sigma_f = 0.25;
sigma_n = sigma_n_true;
A = l^2;
invA = 1/A;


% compute mean and variance of the integral over f
id = 1;
K = sqexp(x_tr,x_tr,invA,sigma_f) + (sigma_n^2 + 1e-9)*eye(n_tr);
L = chol(K,'lower');
z = sqexp(x_tr,b,inv(A+B),sigma_f)/sqrt(det(id+invA*B));
a = L'\(L\y_tr);
m_intf = z'*a;
v_intf1 = sigma_f^2/sqrt(det(id+2*invA*B));
v_intf = v_intf1 - z'*(L'\(L\z));

% GP mean and variance for predicting at test points
kx = sqexp(x_grid,x_tr,invA,sigma_f);
m_f_tr = kx*a;
c_f_tr = sqexp(x_grid,x_grid,invA,sigma_f) - kx*(L'\(L\kx'));
v_f_tr = diag(c_f_tr); % could be computed directly...
s_f_tr = sqrt(v_f_tr);

% compute 'true' integral value using trapz and tight discretisation
f_true_grid = f_true(x_grid);
pi_grid = normpdf(x_grid,b,sqrt(B));
f_pi_true_grid = f_true_grid.*pi_grid;
intf_true = trapz(x_grid,f_pi_true_grid);


% visualise integral I_1
if 1
    figure(1);
    subplot(1,2,1); % plots posterior over the integral of f
    hold on;
    plot(intf_true,0,'xr'); % true integral value
    plot(m_intf,0,'*k');
    intf_grid = linspace(0,max(f_true_grid),1000)';
    plot(intf_grid,normpdf(intf_grid,m_intf,sqrt(v_intf)),'-k'); 
    hold off;
    xlabel('integral (evidence)');
    box on;
    
    subplot(1,2,2); % plots posterior over f
    hold on;
    plot(x_grid,f_true_grid,'-r');
    plot(x_tr,y_tr,'*k'); % data points
    plot(x_grid,pi_grid,'-m'); % density pi
    plot(x_grid,m_f_tr,'-k'); % gp mean
    plot(x_grid,m_f_tr + 1.96*s_f_tr,'--k'); % gp upper 95% CI
    plot(x_grid,m_f_tr - 1.96*s_f_tr,'--k'); % gp lower 95% CI
    plot(x_bds,[0,0],'-b'); % zero line
    hold off;
    xlabel('x');
    ylabel('likelihood');
    box on;
    set(gcf,'Position',[25 590 1200 450]);
end


%% INTEGRAL I_2: EXPECTATION USING TAYLOR SERIES APPROX. AND BQ:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% exact value using trapz and tight discretisation
xf_pi_true_grid = x_grid.*f_true_grid.*pi_grid;
int1 = trapz(x_grid,xf_pi_true_grid);
int2 = intf_true;
intf_true = int1/int2;

% expectation and variance of the integral int{r(x)f(x)pi(x)}dx in the nominator
iaib = invA + invB;
Gk1 = iaib\(invA*x_tr + invB*b);
zr = z.*Gk1;
m_intfr = zr'*a;
edk1 = (iaib\invA)/(inv(A + B) + invB);
v_ri1 = v_intf1*(edk1 + b^2);
v_intfr = v_ri1 - zr'*(L'\(L\zr));
s_intfr = sqrt(v_intfr);

% covariance between int{f(x)pi(x)}dx and int{xf(x)pi(x)}dx
cov_ffr = v_intf1*b - z'*(L'\(L\zr));

% analytical Taylor approx. for the ratio integral
m_ri = m_intfr/m_intf - cov_ffr/m_intf^2 + m_intfr*v_intf/m_intf^3;
v_ri = v_intfr/m_intf^2 - 2*m_intfr*cov_ffr/m_intf^3 + m_intfr^2*v_intf/m_intf^4;
v_ri = max(0,v_ri);
s_ri = sqrt(v_ri);

% expectation and variance of the expectation - by simulation from GP
nsimul = 1000; % how many simuls from GP
jitter = 1e-9;
L_grid = chol(c_f_tr + jitter*eye(size(c_f_tr)),'lower');
rr = randn(n_grid,nsimul);
f_draws = NaN(n_grid,nsimul);
fpi_draws = NaN(n_grid,nsimul);
uis = NaN(1,nsimul);
lis = NaN(1,nsimul);
es = NaN(1,nsimul);
% mtpps_j = bsxfun(@plus, mt, tauL*randn(n_is,N))';
for i = 1:nsimul
    f_draws(:,i) = m_f_tr + L_grid*rr(:,i); % simulate sample path from GP
    fpi_draws(:,i) = f_draws(:,i).*pi_grid;
    uis(i) = trapz(x_grid, x_grid.*fpi_draws(:,i));
    lis(i) = trapz(x_grid, fpi_draws(:,i));
    es(i) = uis(i)/lis(i);
end
fl_me = median(bsxfun(@rdivide,f_draws,lis),2); % median value
fpil_me = median(bsxfun(@rdivide,fpi_draws,lis),2); % median value

% visualise integral I_2
if 1
    figure(2);
    subplot(1,3,1); % 1/3: plots posterior over [int xf(x)pi(x)dx]/[int f(x)pi(x)dx]
    ri_grid = linspace(0.95*min(es),1.05*max(es),1000);
    min(es), max(es)
    ri_eval_grid = ksdensity(es,ri_grid);
    hold on;
    plot(intf_true,0,'xk'); % true integral value
    plot(ri_grid,ri_eval_grid,'-r'); % computed using simulation
    % computed using analytical approx.:
    plot(m_ri,0,'*k'); 
    plot(ri_grid,normpdf(ri_grid,m_ri,s_ri),'-k'); 
    hold off;
    xlabel('integral (expectation)');
    box on;
    
    subplot(1,3,2); % 2/3: plots posterior draws of [f]/[int f(x)pi(x)dx]
    hold on;
    n_plot = 75;
    for i = 1:min(n_plot,nsimul)
        plot(x_grid,f_draws(:,i)/lis(i),'-k');
    end
    plot(x_grid,fl_me,'-g'); % median
    plot(x_bds,[0,0],'-b'); % zero line
    hold off;
    xlabel('x');
    ylabel('likelihood/evidence');
    box on;
    
    subplot(1,3,3); % 3/3: plots posterior draws of [f*pi]/[int f(x)pi(x)dx]
    hold on;
    n_plot = 75;
    for i = 1:min(n_plot,nsimul)
        plot(x_grid,fpi_draws(:,i)/lis(i),'-k');
    end
    plot(x_grid,fpil_me,'-g'); % median
    plot(x_bds,[0,0],'-b'); % zero line
    hold off;
    xlabel('x');
    ylabel('likelihood x prior/evidence');
    box on;
    
    set(gcf,'Position',[25 25 1800 450]);
end


% compare also int{xf(x)pi(x)}dx computed analytically v.s. by simulation 
if 0
    figure(3);
    hold on;
    rf_grid = linspace(0.95*min(uis),1.05*max(uis),1000);
    plot(rf_grid,normpdf(rf_grid,m_intfr,s_intfr),'-k');
    rf_eval_grid = ksdensity(uis,rf_grid);
    plot(rf_grid,rf_eval_grid,'-r'); % computed using simulation
    plot(int1,0,'xk'); % true integral value
    xlabel('integral (nominator)');
    hold off;
    box on;
    set(gcf,'Position',[1500 590 600 450]);
end
end




