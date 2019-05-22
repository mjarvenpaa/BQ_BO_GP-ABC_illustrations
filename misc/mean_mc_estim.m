function [ev,ex] = mean_mc_estim(f,x)
% Computes the Monte Carlo estimate for posterior expectation and marginal likelihood 
% (evidence)of a 1D posterior. 
% f contains the evaluations of the unnormalised likelihood and x the corresponding points
% sampled from the prior.

f = f(:);
x = x(:);
ev = mean(f);
ex = sum(x.*f)/sum(f);
end


