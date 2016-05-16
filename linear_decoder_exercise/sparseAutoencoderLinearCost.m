function [cost,grad,features] = sparseAutoencoderLinearCost(theta, visibleSize, hiddenSize, ...
                                                            lambda, sparsityParam, beta, data)
% -------------------- YOUR CODE HERE --------------------
% Instructions:
%   Copy sparseAutoencoderCost in sparseAutoencoderCost.m from your
%   earlier exercise onto this file, renaming the function to
%   sparseAutoencoderLinearCost, and changing the autoencoder to use a
%   linear decoder.

%将最后一层修改
W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

%{
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));
%}

m = size(data,2);
%1.forward propagation
Z2 = bsxfun(@plus,W1 * data , b1);
A2 = sigmoid(Z2);
features = A2;
Z3 = bsxfun(@plus,W2 * A2 , b2);
Y = Z3;

cost_term = norm(data - Y,'fro')^2 / (2 * m) ;
weight_decay = lambda / 2 * (sum(sum(W1 .^2)) + sum(sum(W2.^2)) );

rhoi = sum(A2,2)./m;
rho = repmat(sparsityParam,hiddenSize,1);
sparsity = beta .* sum(rho .* log(rho ./ rhoi) + (1 - rho).*log((1 - rho) ./ (1 - rhoi)));
cost=cost_term + weight_decay + sparsity;

rhoi = repmat(sum(A2 ,2) ./ m,1,m);
rho = repmat(rho , 1 , m);
sparsity_penalty = beta .* ( - (rho ./ rhoi) + ( (1 - rho) ./ (1 - rhoi) ) );
delta3 = (Y - data);
delta2 = (W2' * delta3 + sparsity_penalty) .* A2 .* (1 - A2);


W2grad = 1/m * delta3 * A2' + lambda * W2;
b2grad = 1/m * sum(delta3,2);
W1grad = 1/m * delta2 * data' + lambda * W1;
b1grad = 1/m * sum(delta2,2);

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

% -------------------- YOUR CODE HERE --------------------                                    

end
function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end
