# MNIST NeuralNetwork From Scratch
Functions:

(1)	guess_the_params(size)
Responsible for initializing the weights and biases of the neural network.
“size” parameter represents the dimensions of the image in pixels.

(2) relu(z)
Also known as rectified linear unit (ReLu) which is an activation function that enables nonlinear relations in a neural network. By replacing any negative values with 0.
“z” parameter represents the input of the next neuron.

(3) softmax(z)
Responsible for converting a vector of X real numbers in a probability distribution of K probabilities proportional to the exponential of the input numbers.
“z” parameter represents a vector with the outcome form each neuron. 

(4) propagation (w1, b1, w2, b2, x)
Forword propagation is a function used to calculate the final output of the neural network by passing the weights and biases thought each layer of the network.
w1: First weight parameter 
b1: First bias parameter 
w2: Second weight parameter 
b2: Second bias parameter
x: Data to apply the weights and biases

(5) derivative_relu(z)
Variation of the rectified linear unit (ReLu) function that returns if the slope of z is positive or negative.
“z” parameter represents the input of the next neuron.

(6) one_hot_encoding(y)	
Responsible for managing categorical data.
“y” parameter represents a numpy array with the nominal data.

(7) back_propagation(X, Y,  A1, A2, W2, Z1, m)
Responsible for minimizing the loss, during training, between the neurons output and the actual value 
X: Data input
Y: Labels of the data / Target output 
A1: The output from the activation function from the previous layer 
A2: The output from the activation function from two layer before
W2: The second weight parameter passed between the previous layer and the current layer
Z1: The output from the previous layer before applying the activation function 
m: The shape of the image

(8) update_params(w1, b1, w2, b2, der_w1, der_b1, der_w2, der_b2, l_rate)
Responsible for applying the new weight and bias values to the neurons.
w1: First weight output from the network
b1: First bias output from the network
w2: Second weight output from the network
b2: Second bias output from the network
der_w1: represents the slope of the weights by calculating the derivative 
der_b1: represents the slope of the biases by calculating the derivative
der_w2: represents the slope of the weights by calculating the derivative
der_b2: represents the slope of the biases by calculating the derivative
l_rate: the learning rate of each neuron

(9) get_predicitons(a2)
Returns the neuron with the highest probability

(10) gradient_descent(x, y, l_rate, epochs)
Main function that combines all the previous functions to represent an almost complete neural network.
x: Data input 
y: Labels of the data / Target output
l_rate: learning rate / the rate that the weights and biases is going to change between each iteration
epochs: number of iterations for the network



 
