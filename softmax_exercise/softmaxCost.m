function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)
%一点不理解的是，截距项为何没有，就算添加也一样？？？
% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;
%每个theta是一行
thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.


M = theta*data;  % M的每一列就是一个样本所对应的thta*data(:, i)的值
M = bsxfun(@minus, M, max(M, [],1)); %减去每列的最大值以防止溢出，不会改变结果
M = exp(M);   % 
p = bsxfun(@rdivide, M, sum(M));  %得到概率矩阵

cost = -1/numCases .* sum(groundTruth(:)'*log(p(:))) + lambda/2 *sum(theta(:).^2);    % cost function
thetagrad = -1/numCases .* (groundTruth - p) * data' + lambda * theta;               % grad 

% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

