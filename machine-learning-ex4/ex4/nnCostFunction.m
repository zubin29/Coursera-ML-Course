function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));
yExpanded = zeros(m,num_labels);
countingVec = (1:num_labels);

X=[ones(size(X,1),1) X];

a2 = sigmoid(X*Theta1');
a2 = [ones(size(a2,1),1) a2];
a3 = sigmoid(a2*Theta2');

for i=1:m
    yExpanded(i,:) = (countingVec == y(i));
end



costVec = zeros(m,1);
for i=1:m
    costVec(i) = yExpanded(i,:)*log(a3(i,:))' + (1-yExpanded(i,:))*log(1-a3(i,:))';
end

Theta1_temp=Theta1;
Theta1_temp(:,1) =0;

Theta2_temp=Theta2;
Theta2_temp(:,1) =0;

J = -sum(costVec)/m + (1.0/(2*m)*lambda*(sum(sum(Theta1_temp.^2)) + sum(sum(Theta2_temp.^2))));

%[hval,h] = max(a3,[],2);

%%Backpropagation code:-


for i=1:m
    a1 = X(i,:)';
    z2=Theta1*a1;
    a2=sigmoid(z2);
    a2=[1;a2];
    z3=Theta2*a2;
    a3=sigmoid(z3);
    
    
    del3 = a3 - (countingVec == y(i))';
   
    del2 = Theta2'*del3 .* sigmoidGradient([0;z2]);
    del2=del2(2:end);
    
    Delta1 = Delta1 + del2*(a1');
    Delta2 = Delta2 + del3*a2';
end

Theta1_Temp = Theta1;
Theta1_Temp(:,1) = 0;
Theta2_Temp = Theta2;
Theta2_Temp(:,1) = 0;
Theta1_grad = Delta1/m + (lambda/m)*Theta1_temp;
Theta2_grad = Delta2/m + (lambda/m)*Theta2_temp;




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
