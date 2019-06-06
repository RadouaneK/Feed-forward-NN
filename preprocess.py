from smote import Smote
import numpy as np
import pandas as pd

def balanced_data(data, k = 5):
	"""
	This function looks for the minority class, generate synthetic data using 
	SMOTE algorithm and returns resampled data.

	inputs:
		data: numpy array, shape = [nbr_ samples, nbr_features + 1], the first row represents the labels
		k : int, optional (default = 5)
            Number of neighbors to use by default for k_neighbors queries

    outputs:
    		data_resampled: numpy array, shape = [nbr_ samples_new, nbr_features + 1]
	"""

	# separate data according to their classes
	data_0 = data[data[:,0] == 0]
	data_1 = data[data[:,0] == 1]

	#get data of the minority class
	if len(data_0) < len(data_1):
		min_data = data_0
		maj_data = data_1
	else:
		min_data = data_1
		maj_data = data_0

	#Percentage of new syntethic samples 
	N = (len(maj_data) / len(min_data)) * 100
	# generate synthetic data
	syn_data = Smote(data=min_data[:, 1:], N = N, k=k).over_sampling()
	# labels for the synthetic data
	syn_data_labels = np.ones((len(syn_data), 1)) * (1 - maj_data[0,0]) 
	# concatenate all data together
	syn_data = np.concatenate((syn_data_labels, syn_data), axis = 1)
	data_resampled = np.concatenate((maj_data, syn_data), axis = 0)

	return data_resampled


def randomize_data(data, seed = 0):
	"""
	Shuffling the data randomly 
	inputs:
		data: numpy array, shape = [nbr_ samples, nbr_features + 1], the first row represents the labels
		seed: int, to seed the random number generator
	outputs: 
		data: numpy array of the shuffled data
	"""
	m = data.shape[0]
	np.random.seed(seed)
	permutation = list(np.random.permutation(m))
	data = data[permutation,:]

	return data


def populate_train_val(data, train_size = 0.75, k = 5, stratify = False, seed = 0):

	"""
	Split the data to train/val sets
	inputs:
		data: numpy array, shape = [nbr_ samples, nbr_features + 1], the first row represents the labels
		train_size: float, between 0.0 and 1.0, the proportion of the dataset to include in the train split (default = 0.75)
		k : int, optional (default = 5)
		stratify: boolean, set it to True if the data is imbalanced
		seed: int, to seed the random number generator
	outputs: 
		x_train, x_val: numpy arrays, the training and validation data, both are shuffled
	"""

	if stratify:
		# separate data according to their labels
		data_0 = data[data[:,0] == 0] 
		data_1 = data[data[:,0] == 1]

		# training lengths from data_0 and data_1
		index_0 = int(len(data_0) * train_size)
		index_1 = int(len(data_1) * train_size)

		x_train = np.concatenate((data_0[:index_0], data_1[:index_1]), axis = 0)
		x_val = np.concatenate((data_0[index_0:], data_1[index_1:]), axis = 0)

		x_train = randomize_data(x_train, seed = seed)
		x_val = randomize_data(x_val, seed = seed)

	else:
		data = randomize_data(data, seed = seed)
		index = int(len(data) * train_size)
		x_train = data[:index]
		x_val = data[index:]

	return x_train, x_val




###############################################################################

#Testing the output of each function, please uncomment and run the code
"""
data = pd.read_csv('data CSV test file.csv')
data = np.array(data)
# must remove the first column
data = data[:,1:]

data_resampled = balanced_data(data, k = 5)
data_random = randomize_data(data, seed = 0)
x_train, x_test = populate_train_val(data, train_size = 0.7, stratify = True)
"""








