function [] = random_fourier_features_test()
% A simple 1d test case for random Fourier features v.s. standard GP

close all;
%rng(12345);

% function f
n_grid = 500;
x_grid = linspace(-10,10,n_grid)';
f_true = @(x)min(4,x.^2); % true (likelihood) function f(x)
x_bds = [x_grid(1); x_grid(end)];

% select the locations and evaluate the (noisy and expensive) function f
n_tr = 20;
x_tr = x_bds(1)+(x_bds(2)-x_bds(1))*rand(n_tr,1);
sigma_n_true = 10*0.001;
y_tr = f_true(x_tr) + sigma_n_true*randn(size(x_tr));

% set up and fit the GP (uses zero mean GP with squared-exp cov and fixed GP hypers)
l = 1;
sigma_f = 3;
sigma_n = sigma_n_true;
A = l^2;
invA = 1/A;

% GP mean and variance for predicting at test points
K = sqexp(x_tr,x_tr,invA,sigma_f) + (sigma_n^2 + 1e-9)*eye(n_tr);
L = chol(K,'lower');
a = L'\(L\y_tr);
kx = sqexp(x_grid,x_tr,invA,sigma_f);
m_f_tr = kx*a;
c_f_tr = sqexp(x_grid,x_grid,invA,sigma_f) - kx*(L'\(L\kx'));
v_f_tr = diag(c_f_tr); % could be computed directly...
s_f_tr = sqrt(v_f_tr);

% approximate GP mean and variance using random Fourier features
m = 1000;
W = 0 + 1/l^2*randn(m,1);
b = 2*pi*rand(m,1);
alpha = sigma_f^2;
bf = sqrt(2*alpha/m)*cos(W*x_tr(:)' + b)';
phi = sqrt(2*alpha/m)*cos(W*x_grid(:)' + b);
A = bf'*bf + sigma_n^2*eye(m,m);
m_rff = phi'*(A\(bf'*y_tr));
v_rff = diag(phi'*(A\phi)*sigma_n^2);
s_rff = sqrt(v_rff);

% compute other things
f_true_grid = f_true(x_grid);


% visualise
figure(1);
set(gcf,'Position',[25 700 1200 450]);
subplot(1,2,1);
hold on;
plot(x_grid,f_true_grid,'-r'); % true function
plot(x_tr,y_tr,'*k'); % data points
plot(x_grid,m_f_tr,'-k'); % gp mean
plot(x_grid,m_f_tr + 1.96*s_f_tr,'--k'); % gp upper 95% CI
plot(x_grid,m_f_tr - 1.96*s_f_tr,'--k'); % gp lower 95% CI
plot(x_bds,[0,0],'-b'); % zero line
hold off;
box on;
yl = ylim();

subplot(1,2,2);
hold on;
plot(x_grid,f_true_grid,'-r'); % true function
plot(x_tr,y_tr,'*k'); % data points
plot(x_grid,m_rff,'-k'); % gp mean
plot(x_grid,m_rff + 1.96*s_rff,'--k'); % gp upper 95% CI
plot(x_grid,m_rff - 1.96*s_rff,'--k'); % gp lower 95% CI
plot(x_bds,[0,0],'-b'); % zero line
ylim(yl);
hold off;
end




