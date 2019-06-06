#Loading the packages
import numpy as np
import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt 
from test import balanced_data, populate_train_val
from keras.utils.np_utils import to_categorical   

def initialize_parameters(L_dim):
    """
    Initializing weight parameters for the model randomly
    L_dim = list containingthe dimentions of the feed forward neural network
    *note that the first element of thearray is the feature number and last element is the number of classes
    output:
    parameters: a dictionnary of tensors containing the weights and biases
    """
    tf.set_random_seed(1)
    L = len(L_dim)
    parameters = {}
    for i in range(1,L):
        parameters['W'+str(i)] = tf.get_variable('W'+str(i), [L_dim[i-1], L_dim[i]], initializer = tf.contrib.layers.xavier_initializer(seed =1))
        parameters['b'+str(i)] = tf.get_variable('b'+str(i), [1, L_dim[i]], initializer = tf.zeros_initializer())

    return parameters

def create_placeholders(n_x, n_y):

    #Create the placeholders for tensorflow session

    X = tf.placeholder(tf.float32, [None, n_x], name = 'X')
    Y = tf.placeholder(tf.float32, [None, n_y], name = 'Y')
    return X,Y

def forward_propagation(X, parameters):
 
    L = len(parameters)//2
    A = X
    
    for i in range(L):
        A_prev = A
        Z = tf.add(tf.matmul(A_prev, parameters['W'+str(i+1)]) , parameters['b'+str(i+1)])
        A = tf.nn.relu(Z)
    return Z

def compute_cost(Z, Y, lamda =4.):
    
    #compute the cost for the model using softmax cross entropy
 
   
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z, labels = Y))
    L2 = tf.multiply(lamda , sum(tf.nn.l2_loss(a) for a in tf.trainable_variables()))
    cost = tf.add(cost, L2)
    return cost

def mini_batches(X, Y, mb_size = 64):
    """
    Create a list of mini-batches from inputs X and Y
    """
    minibatches = []
    m = X.shape[0]

    permutation = list(np.random.permutation(m))
    X = X[permutation,:]
    Y = Y[permutation,:].reshape((m, Y.shape[1]))

    # Partition of the data
    mb_nbr = int(m/mb_size)
    for i in range(0, mb_nbr):
        mb_X = X[i*mb_size:(i+1)*mb_size, :]
        mb_Y = Y[i*mb_size:(i+1)*mb_size, :]
        mini_batch = (mb_X, mb_Y)
        minibatches.append(mini_batch)

    if m%mb_size != 0:
        mb_X = X[mb_size*mb_nbr:,:]
        mb_Y = Y[mb_size*mb_nbr:,:]
        mini_batch = (mb_X, mb_Y)
        minibatches.append(mini_batch)
    return minibatches

def model(X_training, X_test, y_training, y_test, L_dim, minibatch_size = 128, learning_rate = 0.001, number_of_epochs = 10000):
    """
    train the model and with specified dimentions
    L_dim: list, dimensions of the layers   
    """
    tf.reset_default_graph() 
  
    cache = dict(train_cost=[], test_cost=[], train_acc =[], test_acc=[] )

    m = X_training.shape[0] # number of samples
    
    X, Y = create_placeholders(L_dim[0], L_dim[-1])
    parameters = initialize_parameters(L_dim)
    Z = forward_propagation(X, parameters)
    cost = compute_cost(Z, Y)

    correct_prediction = tf.equal(tf.argmax(Z,1), tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))
    

    #learning rate decay
    global_step = tf.Variable(0)
    decayed_learning_rate = tf.train.exponential_decay(learning_rate, global_step, 10000, 0.96, staircase = True)
    #backpropagation phase
    optimizer = tf.train.AdamOptimizer(learning_rate = decayed_learning_rate).minimize(cost, global_step = global_step)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(1, number_of_epochs+1):
            epoch_cost = 0.
            minibatches = mini_batches(X_training, y_training, minibatch_size)
            num_minibatches = int(m / minibatch_size)
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                epoch_cost += minibatch_cost / num_minibatches

            if  i % 100 == 0:
                train_acc = sess.run(accuracy, feed_dict={X:X_training, Y:y_training})
                test_acc, test_cost = sess.run([accuracy,cost], feed_dict={X:X_test, Y:y_test})
                l_rate = sess.run(decayed_learning_rate)
                print ("epoch:%i train_cost: %f train_acc: %f test_cost: %f test_acc: %f " % (i, epoch_cost, train_acc, test_cost, test_acc))
                cache['train_cost'].append(epoch_cost)
                cache['train_acc'].append(train_acc)
                cache['test_cost'].append(test_cost)
                cache['test_acc'].append(test_acc)
                
        print ("Train Accuracy Final:", accuracy.eval({X: X_training, Y: y_training}))
        print ("Test Accuracy Final:", accuracy.eval({X: X_test, Y: y_test}))
    

    plt.figure(figsize=(12, 8))
    plt.plot(np.array(cache['train_cost']), "r-", label="Train")
    plt.plot(np.array(cache['test_cost']), "b-", label="Test")
    plt.plot(np.array(cache['train_acc']), "r-")
    plt.plot(np.array(cache['test_acc']), "b-")
    plt.legend(loc='upper right', shadow=True)
    plt.ylabel('Cost and Accuracy')
    plt.xlabel('Epoch')
    plt.ylim(0,2)
    plt.grid()
    plt.show()
#EOF 

# example on how to run the model
data = pd.read_csv('data CSV test file.csv')
data = np.array(data)
data = data[:,1:]

#split the data into training and validation sets
train, test = populate_train_val(data, train_size = 0.7, stratify = False)

# we should balance only the training set
train = balanced_data(train)


# splitting data and labels
X_training = train[:,1:]
X_test = test[:,1:]
y_training = train[:,0]
y_test = test[:,0]

#one hot encoding
y_training = to_categorical(y_training, 2)
y_test = to_categorical(y_test, 2)

# training
model(X_training, X_test,y_training, y_test, L_dim = [X_training.shape[1],10,10,2],minibatch_size = 512, learning_rate = 0.001,  number_of_epochs = 10000)
