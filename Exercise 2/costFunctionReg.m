function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Part 1, Calculating Cost
tmp = 0;
for i=1:m
  sig = sigmoid(X(i,:) * theta);
  tmp = tmp + ((-y(i) * log(sig)) - ((1-y(i)) * log(1-sig)));
end
% Regularize Theta.
reg = 0;
for i=2:length(theta)
	reg = reg + (theta(i)^2);
end

J = ((1 / m)  * tmp) + ((lambda/(2 * m))*reg);

% Part 2, Calculating Gradient Descent.
% Gradient Descent
for j=1:length(theta)
	tmp =0;
	reg = 0;
	for k=1:m
		sig = sigmoid(X(k,:) * theta);
		tmp = tmp + ((sig - y(k))* X(k,j));
	end
	if(j > 1)
		reg = (lambda/m)*theta(j);
	end
	
	grad(j) = ((1/m) * tmp) + reg;
end
%disp(grad)

% =============================================================

end
