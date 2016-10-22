function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

J = (1/(2*m)) * ((X*theta - y)' * (X*theta - y)); 

R = (lambda/(2*m)) * sum(theta(2:end,:).^2);
J = J + R;

% calculate gradient
%L = eye(length(theta));
%L(1,1) = 0;
%grad = pinv(X'*X + lambda*L) * (X'*y);

for j=1:length(theta)
	tmp =0;
	reg = 0;
	sig = X * theta;
	for k=1:m
		%sig = sigmoid(X(k,:) * theta);
		%sig = theta(1) + X(k,2)*theta(2);		
		tmp = tmp + ((sig(k) - y(k))* X(k,j));
	end
	
		
		
	if(j > 1)
		reg = (lambda/m)*theta(j);
	end
	
	grad(j) = ((1/m) * tmp) + reg;
end









% =========================================================================

grad = grad(:);

end
