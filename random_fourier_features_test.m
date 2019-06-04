function [] = random_fourier_features_test()
% A simple 1d test case for random Fourier features v.s. standard GP

close all;
%rng(12345);

% function f
n_grid = 500;
x_grid = linspace(-10,10,n_grid)';
f_true = @(x)x.*sin(1/2*x+1)-1; % true (likelihood) function f(x)
x_bds = [x_grid(1); x_grid(end)];

% select the locations and evaluate the (noisy and expensive) function f
n_tr = 20;
x_tr = x_bds(1)+(x_bds(2)-x_bds(1))*rand(n_tr,1);
sigma_n_true = 0.001; % NON-NOISY CASE
sigma_n_true = 1; % NOISY CASE
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
W = 1/l^2*randn(m,1);
b = 2*pi*rand(m,1);
alpha = sigma_f^2;
bf = sqrt(2*alpha/m)*cos(W*x_tr(:)' + b)';
phi = sqrt(2*alpha/m)*cos(W*x_grid(:)' + b);
A = bf'*bf + sigma_n^2*eye(m,m);
m_rff = phi'*(A\(bf'*y_tr));
c_rff = phi'*(A\phi)*sigma_n^2;
v_rff = diag(c_rff);
s_rff = sqrt(v_rff);

% compute other things
f_true_grid = f_true(x_grid);

% compute some exact and approximate sample paths
ns = 15;
sp_exact = NaN(ns,n_grid);
sp_approx = NaN(ns,n_grid);
rr = randn(n_grid,ns);
choltol = 1e-9;
Le = chol(c_f_tr + choltol*eye(size(c_f_tr)),'lower');
La = chol(c_rff + choltol*eye(size(c_rff)),'lower');
for i = 1:ns
    sp_exact(i,:) = m_f_tr + Le*rr(:,i);
    sp_approx(i,:) = m_rff + La*rr(:,i);
end


%% visualise GP fits
figure(1);
set(gcf,'Position',[25 700 1200 900]);
subplot(2,2,1);
title('exact');
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

subplot(2,2,2);
title('RFF approx.');
hold on;
plot(x_grid,f_true_grid,'-r'); % true function
plot(x_tr,y_tr,'*k'); % data points
plot(x_grid,m_rff,'-k'); % gp mean
plot(x_grid,m_rff + 1.96*s_rff,'--k'); % gp upper 95% CI
plot(x_grid,m_rff - 1.96*s_rff,'--k'); % gp lower 95% CI
plot(x_bds,[0,0],'-b'); % zero line
ylim(yl);
hold off;
box on;

%% sample paths from the GP
subplot(2,2,3);
hold on;
for i = 1:ns
    plot(x_grid,sp_exact(i,:),'-k');
end
plot(x_tr,y_tr,'*k'); % data points
plot(x_bds,[0,0],'-b'); % zero line
hold off;
box on;

subplot(2,2,4);
hold on;
for i = 1:ns
    plot(x_grid,sp_approx(i,:),'-k');
end
plot(x_tr,y_tr,'*k'); % data points
plot(x_bds,[0,0],'-b'); % zero line
hold off;
box on;

%% compare the true SE-kernel to the rff-kernel (inner product)
if 1
    figure(2);
    set(gcf,'Position',[1300 700 550 400]);
    nxx = 1000;
    xx = linspace(0,5*l,nxx);
    y_sek = sigma_f^2*exp(-0.5*xx.^2/l^2);
    y_rff = 2*alpha/m*sum(cos(zeros(1,nxx) + b).*cos(W*xx(:)' + b),1);
    
    hold on;
    plot([xx(1),xx(end)],[0,0],'-k'); % zero line
    plot(xx,y_sek,'-r');
    plot(xx,y_rff,'-b');
    ylim([-1,1.05*sigma_f^2]);
    hold off;
    box on;
    title('SE kernel (red), RFF SE kernel approx. (blue)');
end



