function [] = GPdiscrepancy_lik_vs_unnorm_lik()
% Test using simulation in a simple 1d scenario how the uncertainties in the likelihood
% compare to the normalised likelihood when the GP is placed on the discrepancy.

close all; 
%rng(12345);

% true likelihood exp(f) and log-likelihood f, parameter x
n_grid = 500;
x_grid = linspace(-10,10,n_grid)';
mu = 1;
sigma_n = 3; % stdev of the Gaussian discrepancy
epsilon = 0; % ABC threshold
d_true = @(x)0.5*(x-mu).^2 + sigma_n; % true mean of discrepancy
f_true = @(x)normcdf_fast((epsilon-d_true(x))/sigma_n); % true ABC likelihood
x_bds = [x_grid(1); x_grid(end)];

% select the locations and evaluate the discrepancy/likelihood
n_tr = 50;
if 1
    x_tr = x_bds(1)+(x_bds(2)-x_bds(1))*rand(n_tr,1);
else
    x_tr = min(x_bds(2),max(x_bds(1),mu + 2*randn(n_tr,1)));
end
sigma_n_true = sigma_n;
y_tr = d_true(x_tr) + sigma_n_true*randn(size(x_tr));

% set up and fit the GP (uses zero mean GP with squared-exp cov and fixed GP hypers)
l = 1.5;
sigma_f = 10;
A = l^2;
invA = 1/A;

% compute mean and variance of the integral over f
K = sqexp(x_tr,x_tr,invA,sigma_f) + (sigma_n^2 + 1e-9)*eye(n_tr);
L = chol(K,'lower');
a = L'\(L\y_tr);

% GP mean and variance for predicting log-likelihood f at test points x_grid
kx = sqexp(x_grid,x_tr,invA,sigma_f);
m_d_tr = kx*a;
c_d_tr = sqexp(x_grid,x_grid,invA,sigma_f) - kx*(L'\(L\kx'));
v_d_tr = diag(c_d_tr); % could be computed directly...
s_d_tr = sqrt(v_d_tr);

% mean and quantiles for likelihood from the discrepancy-GP 
med_f_tr = normcdf_fast((epsilon-m_d_tr)/sigma_n);
mean_f_tr = normcdf_fast((epsilon-m_d_tr)./sqrt(sigma_n^2 + v_d_tr));
uc_f_tr = normcdf_fast((s_d_tr*norminv(0.975)-m_d_tr+epsilon)/sigma_n);
lc_f_tr = normcdf_fast((s_d_tr*norminv(0.025)-m_d_tr+epsilon)/sigma_n);

% sample from the GP, exponentiate and normalise
nsimul = 2000; % how many simuls from GP
jitter = 1e-9;
L_grid = chol(c_d_tr + jitter*eye(size(c_d_tr)),'lower');
rr = randn(n_grid,nsimul);
d_draws = NaN(n_grid,nsimul);
f_draws = NaN(n_grid,nsimul);
x_f_draws = NaN(n_grid,nsimul);
norm_f_draws = NaN(n_grid,nsimul);
xnorm_f_draws = NaN(n_grid,nsimul);
Zs = NaN(1,nsimul);
uis = NaN(1,nsimul);
es = NaN(1,nsimul);
for i = 1:nsimul
    % note: numerical over/overflows not handled!
    d_draws(:,i) = m_d_tr + L_grid*rr(:,i); % simulate sample path from GP
    f_draws(:,i) = normcdf_fast((epsilon-d_draws(:,i))/sigma_n); 
    Zs(i) = trapz(x_grid,f_draws(:,i)); % sampled evidence
    x_f_draws(:,i) = x_grid.*f_draws(:,i);
    norm_f_draws(:,i) = f_draws(:,i)/Zs(i);
    xnorm_f_draws(:,i) = x_grid.*norm_f_draws(:,i);
    uis(i) = trapz(x_grid, x_f_draws(:,i)); % sampled unnormalised expectation
    es(i) = uis(i)/Zs(i); % sampled expectation
end
med_norm_f = median(norm_f_draws,2); % median value
mean_norm_f = mean(norm_f_draws,2); % median value
uc_norm_f = quantile(norm_f_draws,0.975,2); 
lc_norm_f = quantile(norm_f_draws,0.025,2); 

% exact value
int_true = mu;

% visualise
if 1
    PLOT_SPS = 1;
    truecol = 'g';
    lw = 1.3;
    aa = 0.99;
    LOGP = 0;
    
    figure(1);
    set(gcf,'Position',[25 590 1800 1000]);
    suptitle('GP prior on discrepancy:');
    
    %% discrepancy (follows GP)
    subplot(4,2,1); 
    hold on;
    n_plot = 75;
    for i = 1:min(n_plot,nsimul)
        plot(x_grid,d_draws(:,i),'-k'); % draws
    end
    plot(x_tr,y_tr,'*k'); % discrepancy realisations
    plot(x_bds,epsilon*[1,1],'--b'); % epsilon
    plot(x_grid,m_d_tr,['-',truecol],'LineWidth',lw); % GP mean
    plot(x_grid,m_d_tr + 1.96*s_d_tr,['-',truecol],'LineWidth',lw); % gp upper 95% CI
    plot(x_grid,m_d_tr - 1.96*s_d_tr,['-',truecol],'LineWidth',lw); % gp lower 95% CI
    hold off;
    title('discrepancy');
    xlabel('x');
    box on;
    
    %% posterior of the expectation ratio integral
    subplot(4,2,2); 
    ri_grid = linspace(x_bds(1),x_bds(2),1000);
    ri_eval_grid = ksdensity(es,ri_grid);
    hold on;
    plot(ri_grid,ri_eval_grid,'-r'); % computed using simulation
    esp = es(1:min(length(es),100));
    plot(esp,zeros(size(esp)),'*k');
    plot(int_true,0,'xr'); % true expectation
    hold off;
    xlim(x_bds);
    title('expectation');
    xlabel('integral (expectation)');
    box on;
    min(es),max(es)
    
    
    %% unnormalised likelihood 
    subplot(4,2,3); 
    hold on;
    if PLOT_SPS
        n_plot = 75;
        for i = 1:min(n_plot,nsimul)
            plot(x_grid,f_draws(:,i),'-k');
        end
    end
    f_tr = normcdf_fast((epsilon-y_tr)/sigma_n);
    plot(x_tr,f_tr,'*k'); % data points (y exp-transformed)
    plot(x_grid,med_f_tr,[':',truecol],'LineWidth',1.5); % f median
    plot(x_grid,mean_f_tr,['-',truecol],'LineWidth',lw); % f mean 
    plot(x_grid,uc_f_tr,['--',truecol],'LineWidth',lw); % f upper 95% CI
    plot(x_grid,lc_f_tr,['--',truecol],'LineWidth',lw); % f lower 95% CI
    hold off;
    title('likelihood (=acceptance probability)');
    xlabel('x');
    box on;
    
    %% posterior of normalised likelihood by simulation
    subplot(4,2,4); 
    hold on;
    if PLOT_SPS
        n_plot = 75;
        for i = 1:min(n_plot,nsimul)
            plot(x_grid,norm_f_draws(:,i),'-k');
        end
    end
    plot(x_grid,med_norm_f,[':',truecol],'LineWidth',1.5); % normalised f median
    plot(x_grid,mean_norm_f,['-',truecol],'LineWidth',lw); % normalised f median
    plot(x_grid,uc_norm_f,['--',truecol],'LineWidth',lw); % normalised f upper 95% CI
    plot(x_grid,lc_norm_f,['--',truecol],'LineWidth',lw); % normalised f lower 95% CI
    hold off;
    title('likelihood (normalised)');
    xlabel('x');
    box on;
    
    %% posterior of x * normalised likelihood by simulation
    subplot(4,2,5); 
    hold on;
    if PLOT_SPS
        n_plot = 75;
        for i = 1:min(n_plot,nsimul)
            plot(x_grid,xnorm_f_draws(:,i),'-k');
        end
    end
    hold off;
    title('x * likelihood (normalised)');
    xlabel('x');
    box on;
    
    %% posterior of upper integral i.e. x * likelihood by simulation
    subplot(4,2,6);
    if LOGP
        uis = log(-min(uis)+uis+1);
    end
    ui_grid = linspace(min(uis),1.1*quantile(uis,aa),1000);
    ui_eval_grid = ksdensity(uis,ui_grid);
    hold on;
    plot(ui_grid,ui_eval_grid,'-r'); % computed using simulation
    uisp = uis(1:min(length(uis),75));
    plot(uisp,zeros(size(uisp)),'*k');
    hold off;
    xlabel('integral (unnormalised expectation)');
    box on;
    
    %% posterior of evidence by simulation
    subplot(4,2,7);
    if LOGP
        Zs = log(Zs);
    end
    zs_grid = linspace(min(Zs),1.1*max(Zs),1000);
    z_eval_grid = ksdensity(Zs,zs_grid);
    hold on;
    plot(zs_grid,z_eval_grid,'-r'); % computed using simulation
    Zsp = Zs(1:min(length(Zs),75));
    plot(Zsp,zeros(size(Zsp)),'*k');
    hold off;
    xlabel('evidence');
    box on;
end
end