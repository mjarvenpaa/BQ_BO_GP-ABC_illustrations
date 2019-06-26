function rnd_ratio_distribution()
% Simulates points from the density of x/y where (x,y) have joint Gaussian distribution. 
% (In some special cases, e.g. if both means=0, this is in fact a general Cauchy distribution.)

close all;
%rng(12345);

n_simul = 10000;
mu = 0*5*[3; 3];
v1 = 1^2;
v2 = 1^2;
roo = 0.9; % correlation coefficient

% simulate
cc = roo*sqrt(v1*v2);
S = [v1 cc; cc v2];
cholS = chol(S,'lower'); 
rr = randn(2,n_simul);
xy_samples = NaN(2,n_simul);
xpy_samples = NaN(1,n_simul);
for i = 1:n_simul
    xy_samples(:,i) = mu + cholS*rr(:,i);
    xpy_samples(i) = xy_samples(1,i)/xy_samples(2,i);
end

% visualise
figure(1);
set(gcf,'Position',[25 600 1400 450]);
mm = [min(xpy_samples),max(xpy_samples)];
xy = linspace(mm(1),mm(2),1000);
fxy = ksdensity(xpy_samples,xy);

subplot(1,3,1);
plot(xy,fxy);
xlabel('x/y');

subplot(1,3,2);
histogram(xpy_samples,60);
xlabel('x/y');

subplot(1,3,3);
plot(xy_samples(1,:),xy_samples(2,:),'b*');
xlabel('x');
ylabel('y');

end

