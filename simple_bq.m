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
% I_2 = [int{xf(x)pi(x)}dx]/[int{f(x)pi(x)}dx].
%

close all;

%% INTEGRAL I_1:
%%%%%%%%%%%%%%%%

% function f and density pi
n_grid = 500;
x_grid = linspace(0,10,n_grid)';
f_true = @(x)exp(-0.5*(x-4).^2/1^2)/2; % true (likelihood) function f(x)
x_bds = [x_grid(1); x_grid(end)];

b = 5; % mean and variance for the gaussian density pi(x)
B = 1^2;

% select the locations and evaluate the (noisy and expensive) function f
n_tr = 127;
x_tr = x_bds(1)+(x_bds(2)-x_bds(1))*rand(n_tr,1);
sigma_n_true = 0.01;
y_tr = f_true(x_tr) + sigma_n_true*randn(size(x_tr));


% set up and fit the GP (uses zero mean GP with squared-exp cov and fixed GP hypers)
l = 2;
sigma_f = 0.5;
sigma_n = sigma_n_true;
A = l^2;
invA = 1/A;


% compute mean and variance of the integral over f
id = 1;
K = sqexp(x_tr,x_tr,invA,sigma_f) + (sigma_n^2 + 1e-9)*eye(n_tr);
L = chol(K,'lower');
z = sqexp(x_tr,b,inv(A+B),sigma_f)/sqrt(det(id+invA*B));
a = L'\(L\y_tr(:));
m_intf = z'*a;
v_intf = sigma_f^2/sqrt(det(id+2*invA*B)) - z'*(L'\(L\z));

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
    plot(intf_true,0,'*r'); % true integral value
    plot(m_intf,0,'*k');
    intf_grid = linspace(0,max(f_true_grid),1000)';
    plot(intf_grid,normpdf(intf_grid,m_intf,sqrt(v_intf)),'-k'); % density pi
    hold off;
    xlabel('integral (evidence)');
    box on;
    
    subplot(1,2,2); % plots posterior over f
    hold on;
    plot(x_grid,f_true_grid,'-r');
    plot(x_tr,y_tr,'*k');
    plot(x_grid,pi_grid,':r');
    plot(x_grid,m_f_tr,'-k'); % gp mean
    plot(x_grid,m_f_tr + 1.96*s_f_tr,'--k'); % gp upper 95% CI
    plot(x_grid,m_f_tr - 1.96*s_f_tr,'--k'); % gp lower 95% CI
    plot(x_bds,[0,0],'-b'); % zero line
    hold off;
    xlabel('x');
    box on;
    
    set(gcf,'Position',[50 250 1200 600]);
end

%return;

%% INTEGRAL I_2: EXPECTATION USING TAYLOR SERIES APPROX. AND BQ:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% exact value using trapz and tight discretisation
xf_pi_true_grid = x_grid.*f_true_grid.*pi_grid;
int1 = trapz(x_grid,xf_pi_true_grid);
int2 = intf_true;
intf_true = int1/int2;

% expectation and variance of the expectation - analytical approx. formulas
%...

% expectation and variance of the expectation - by simulation from GP
nsimul = 500; % how many simuls from GP
jitter = 1e-9;
L_grid = chol(c_f_tr + jitter*eye(size(c_f_tr)),'lower');
rr = randn(n_grid,nsimul);
sps = NaN(n_grid,nsimul);
ris = NaN(1,nsimul);
% mtpps_j = bsxfun(@plus, mt, tauL*randn(n_is,N))';
for i = 1:nsimul
    sps(:,i) = m_f_tr + L_grid*rr(:,i); % simulate sample path from GP
    pp_grid = sps(:,i).*pi_grid;
    ui = trapz(x_grid,x_grid.*pp_grid);
    li = trapz(x_grid,pp_grid);
    ris(i) = ui/li;
end

% visualise integral I_2
if 1
    figure(2);
    subplot(1,2,1); % plots posterior over [int xf(x)pi(x)dx]/[int f(x)pi(x)dx]
    ri_grid = linspace(0.95*min(ris),1.05*max(ris),1000);
    ri_eval_grid = ksdensity(ris,ri_grid);
    hold on;
    plot(intf_true,0,'*r'); % true integral value
    plot(ri_grid,ri_eval_grid,'-r'); % computed using simulation
    hold off;
    % computed using analytical approx. 
    
    box on;
    
    % print results
    %ris
    %intf_true
    
    
    subplot(1,2,2); % plots posterior over [f]/[int f(x)pi(x)dx]
    hold on;
    plot(x_grid,f_true_grid,'-r');
    plot(x_tr,y_tr,'*k');
    plot(x_grid,pi_grid,':r');
    plot(x_grid,m_f_tr,'-k'); % gp mean
    plot(x_grid,m_f_tr + 1.96*s_f_tr,'--k'); % gp upper 95% CI
    plot(x_grid,m_f_tr - 1.96*s_f_tr,'--k'); % gp lower 95% CI
    plot(x_bds,[0,0],'-b'); % zero line
    hold off;
    xlabel('x');
    box on;
    
    set(gcf,'Position',[50 250 1200 600]);
end

end


function c = sqexp(x1,x2,invS,sigma_f)
% computes the squared exp cov matrix between rows in x1 and x2

% n1 = length(x1);
% n2 = length(x2);
% dis = NaN(n1,n2);
% for i = 1:n1
%     for j = 1:n2
%         dis(i,j) = (x1(i)-x2(j))^2*invS;
%     end
% end

fu = @(x,y) (x-y).^2*invS;
dis = bsxfun(fu,x1(:),x2(:)');
c = sigma_f^2*exp(-0.5*dis);
end


