function c = sqexp(x1,x2,invS,sigma_f)
% Computes the squared exp cov matrix between points in x1 and x2. Works only in 1d case i.e. invS 
% must be scalar, x1 and x2 vectors.

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

