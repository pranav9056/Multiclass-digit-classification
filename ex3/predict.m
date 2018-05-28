function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

theta1 = Theta1';
z1  = X*theta1;
a1 = sigmoid(z1);
a1 = [ones(m, 1) a1];
theta2 = Theta2';
z2 = a1 * theta2;
a2 = sigmoid(z2);
[val,p] = max(a2,[],2);

end
