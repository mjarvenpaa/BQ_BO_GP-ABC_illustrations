function [] = GPloglik_lik_vs_unnorm_lik()
% Test using simulation in a simple 1d scenario how the uncertainties in the likelihood
% compare to the normalised likelihood when the GP is placed on the log-likelihood.

close all; 
%rng(12345);

% true likelihood exp(f) and log-likelihood f, parameter x
n_grid = 500;
x_grid = linspace(-10,10,n_grid)';
mu = 0;
v = 4*0.5^2;
expf_true = @(x)exp(-0.5*(x-mu).^2/v)/2; % true likelihood function exp(f(x))
f_true = @(x)log(expf_true(x)); % true log-likelihood f(x)
x_bds = [x_grid(1); x_grid(end)];

% select the locations and evaluate the likelihood/log-likelihood
n_tr = 15;
x_tr = x_bds(1)+(x_bds(2)-x_bds(1))*rand(n_tr,1);
sigma_n_true = 100*0.0005;
y_tr = f_true(x_tr) + sigma_n_true*randn(size(x_tr));

% set up and fit the GP (uses zero mean GP with squared-exp cov and fixed GP hypers)
l = 3*0.5;
sigma_f = 8;
sigma_n = sigma_n_true;
A = l^2;
invA = 1/A;

% compute mean and variance of the integral over f
K = sqexp(x_tr,x_tr,invA,sigma_f) + (sigma_n^2 + 1e-9)*eye(n_tr);
L = chol(K,'lower');
a = L'\(L\y_tr);

% GP mean and variance for predicting log-likelihood f at test points x_grid
kx = sqexp(x_grid,x_tr,invA,sigma_f);
m_f_tr = kx*a;
c_f_tr = sqexp(x_grid,x_grid,invA,sigma_f) - kx*(L'\(L\kx'));
v_f_tr = diag(c_f_tr); % could be computed directly...
s_f_tr = sqrt(v_f_tr);

% mean and quantiles for likelihood from the log-GP 
med_expf_tr = exp(m_f_tr);
uc_expf_tr = exp(m_f_tr + norminv(0.975)*s_f_tr);
lc_expf_tr = exp(m_f_tr + norminv(0.025)*s_f_tr);

% sample from the GP, exponentiate and normalise
nsimul = 2000; % how many simuls from GP
jitter = 1e-9;
L_grid = chol(c_f_tr + jitter*eye(size(c_f_tr)),'lower');
rr = randn(n_grid,nsimul);
f_draws = NaN(n_grid,nsimul);
expf_draws = NaN(n_grid,nsimul);
x_expf_draws = NaN(n_grid,nsimul);
norm_expf_draws = NaN(n_grid,nsimul);
xnorm_expf_draws = NaN(n_grid,nsimul);
Zs = NaN(1,nsimul);
uis = NaN(1,nsimul);
es = NaN(1,nsimul);
for i = 1:nsimul
    % note: numerical over/overflows not handled!
    f_draws(:,i) = m_f_tr + L_grid*rr(:,i); % simulate sample path from GP
    expf_draws(:,i) = exp(f_draws(:,i)); 
    Zs(i) = trapz(x_grid,expf_draws(:,i)); % sampled evidence
    x_expf_draws(:,i) = x_grid.*expf_draws(:,i);
    norm_expf_draws(:,i) = expf_draws(:,i)/Zs(i);
    xnorm_expf_draws(:,i) = x_grid.*norm_expf_draws(:,i);
    uis(i) = trapz(x_grid, x_expf_draws(:,i)); % sampled unnormalised expectation
    es(i) = uis(i)/Zs(i); % sampled expectation
end
med_norm_expf = median(norm_expf_draws,2); % median value
uc_norm_expf = quantile(norm_expf_draws,0.975,2); 
lc_norm_expf = quantile(norm_expf_draws,0.025,2); 

% exact value
int_true = mu;

% visualise
if 1
    PLOT_SPS = 1;
    truecol = 'g';
    lw = 1.3;
    aa = 0.99;
    LOGP = 1;
    
    figure(1);
    set(gcf,'Position',[25 590 1800 1000]);
    suptitle('GP prior on log-likelihood:');
    
    %% log-likelihood (follows GP)
    subplot(4,2,1); 
    hold on;
    n_plot = 75;
    for i = 1:min(n_plot,nsimul)
        plot(x_grid,f_draws(:,i),'-k'); % draws
    end
    plot(x_tr,y_tr,'*k'); % data points
    plot(x_grid,m_f_tr,['-',truecol],'LineWidth',lw); % GP mean
    plot(x_grid,m_f_tr + 1.96*s_f_tr,['-',truecol],'LineWidth',lw); % gp upper 95% CI
    plot(x_grid,m_f_tr - 1.96*s_f_tr,['-',truecol],'LineWidth',lw); % gp lower 95% CI
    hold off;
    title('log-likelihood');
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
    %min(es),max(es)
    
    
    %% unnormalised likelihood (follows log-GP)
    subplot(4,2,3); 
    hold on;
    if PLOT_SPS
        n_plot = 75;
        for i = 1:min(n_plot,nsimul)
            plot(x_grid,expf_draws(:,i),'-k');
        end
    end
    f_tr = exp(y_tr);
    plot(x_tr,f_tr,'*k'); % data points (y exp-transformed)
    plot(x_grid,med_expf_tr,['-',truecol],'LineWidth',lw); % expf median
    plot(x_grid,uc_expf_tr,['--',truecol],'LineWidth',lw); % expf upper 95% CI
    plot(x_grid,lc_expf_tr,['--',truecol],'LineWidth',lw); % expf lower 95% CI
    hold off;
    title('likelihood');
    xlabel('x');
    box on;
    
    %% posterior of normalised likelihood by simulation
    subplot(4,2,4); 
    hold on;
    if PLOT_SPS
        n_plot = 75;
        for i = 1:min(n_plot,nsimul)
            plot(x_grid,norm_expf_draws(:,i),'-k');
        end
    end
    plot(x_grid,med_norm_expf,['-',truecol],'LineWidth',lw); % normalised expf median
    plot(x_grid,uc_norm_expf,['--',truecol],'LineWidth',lw); % normalised expf upper 95% CI
    plot(x_grid,lc_norm_expf,['--',truecol],'LineWidth',lw); % normalised expf lower 95% CI
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
            plot(x_grid,xnorm_expf_draws(:,i),'-k');
        end
    end
    hold off;
    title('x * likelihood (normalised)');
    xlabel('x');
    box on;
    
    %% posterior of upper integral i.e. x * likelihood by simulation
    subplot(4,2,6);
    if 0 && LOGP
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


