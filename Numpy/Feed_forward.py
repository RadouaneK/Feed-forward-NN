import numpy as np 


#################### activation functions##################
def sigmoid(Z):
	A = 1 / (1 + np.exp(-Z))
	return A

def relu(Z):
	A = np.maximum(0,Z)
	return A

def sigmoid_gradient(Z):
	A = sigmoid(Z)
	DaDz = np.multiply(1 - A, A)
	return DaDz

def relu_gradient(Z):
	if Z>0:
		DaDz = 1
	elif Z<=0:
		DaDz = 0
	return DaDz		

def sigmoid_back(A_gradient, Z):
	DaDz = sigmoid_gradient(Z)
	Z_gradient = np.dot(A_gradient, DaDz.T)
	return Z_gradient	

def relu_back(A_gradient, Z):
	DaDz = relu_gradient(Z)
	Z_gradient = np.dot(A_gradient, DaDz.T)
	return Z_gradient
################### Initialize parameters ###############
def initialize_parameters(layer_dimentions):

	L = len(layer_dimentions)
	parameters = {}
	for i in range(1,L):
		parameters['W'+str(i)] = np.random.randn(layer_dimentions[i], layer_dimentions[i - 1]) * 0.01
		parameters['b'+ str(i)] = np.zeros((layer_dimentions[i], 1))

	return parameters
################### Forward propagation ########################
def linear_activation(A_previous, weights, b, activation):

	Z = np.dot(weights, A_previous) + b

	if activation == "sigmoid":
		A = sigmoid(Z)
	elif activation == "relu":
		A = relu(Z)

	cache = (A_previous, weights, b, Z)
	return A, cache

def forward_propagation(X, parameters):
	A = X
	L = len(parameters) // 2 
	caches = []

	for i in range(1,L):
		A_previous = A
		A, cache = linear_activation(A_previous, parameters['W'+str(i)], parameters['b'+str(i)], "relu")
		caches.append(cache)

	predicted_output, cache = linear_activation(A, parameters['W'+str(L)], parameters['b'+str(L)], "sigmoid")
	caches.append(cache)
	return predicted_output, caches
####################### computing Cost #############################
def cost_func(predicted_output, Y):
	m = Y.shape[0]
	cost = np.sum(np.multiply(Y, np.log(predicted_output)) + np.multiply(1 - Y, np.log(1 - predicted_output)))/m
	return cost
###################### backpropagation #############################
def backward(A_gradient, cache, activation):
	A_previous, weights, b, Z = cache
	m = A_previous.shape[1]

	if activation == "relu":
		Z_gradient = relu_back(A_gradient, Z)
	elif activation == "sigmoid":
		Z_gradient = sigmoid_back(A_gradient, Z)

	A_previous_gradient = np.dot(weights.T, Z_gradient)
	W_gradient = np.dot(Z_gradient, A_previous.T)/m
	b_gradient = np.sum(Z_gradient, axis=1, keepdims=True)/m

	return A_previous_gradient, W_gradient, b_gradient

def backward_propagation(predicted_output,Y, caches):
	grads = {}
	L = len(caches) 
	m = AL.shape[1]
	Y = Y.reshape(AL.shape)
	
	pred_out_grad = np.devide(1-Y, 1 - predicted_output) - np.devide(Y, predicted_output)

	grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = backward(pred_out_grad, caches[-1], "sigmoid")
	
	for i in reversed(range(L-1)):
		dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA{}".format(i + 2)], caches[i], "relu")
		grads["dA" + str(i + 1)] = dA_prev_temp
		grads["dW" + str(i + 1)] = dW_temp
		grads["db" + str(i + 1)] = db_temp

	return grads

def update_params(learning_rate, grads, parameters):
	L = len(parameters) //2

	for i in range(1,L+1):
		parameters["W" + str(i)] = parameters["W" + str(i)] - learning_rate * grads["dW" + str(i)]
		parameters["b" + str(i)] = parameters["b" + str(i)] -learning_rate * grads["db" + str(i)]
	return parameters

def data_file(file):
	dataset= np.loadtxt(open(file), delimiter=",", skiprows=0)
	inputs = dataset[:, 1:4]
	inputs = inputs.T
	Y = dataset[:, -1]
	Y = Y==2
	Y = 1*Y
	X_training, X_test, y_training, y_test = train_test_split(inputs, Y, test_size=0.2, random_state=42)
	y_training = y_training.transpose()
	return X_training, X_test, y_training, y_test
def model(X, Y, layer_dimentions, learning_rate, epoch = 3000):

	#inialize the parameters for the neural networks
	parameters = initialize_parameters(layer_dimentions)

	for i in range(1, epoch+1):
		predicted_output, caches = forward_propagation(X, parameters)
		cost = cost_func(predicted_output, Y)
		grads = backward_propagation(predicted_output, Y, caches)
		parameters = update_params(learning_rate, grads, parameters)

		if i!= 1 and i % 100 !=0 :
			continue

		print ("Cost after iteration %i: %f" % (i, cost))

	return parameters
"""
X_training, X_test, y_training, y_test = data_file("all_data.txt")

parameters = model(X_training, y_training, [3,50,2], 0.01)
"""

# TODO: do the training part



	





 