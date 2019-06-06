from sklearn.neighbors import NearestNeighbors
import random


class Smote():
    """
    SMOTE algorithm: Oversampling the minority class.
    inputs:
        data: numpy array, minority class samples
        N: Percentage of new syntethic samples
        k: int, int, optional (default = 5)
            Number of neighbors to use by default for k_neighbors queries
    output:
        synthetic: python list containing the synthetic data 
    """


    def __init__(self, data, N , k = 5):
        self.data = data
        self.k = k
        self.T = len(self.data)
        self.N = N
        self.newIndex = 0
        self.synthetic = []
        self.neighbors = NearestNeighbors(n_neighbors=self.k).fit(self.data)

    def over_sampling(self):
        if self.N < 100:
            self.T = (self.N / 100) * self.T
            self.N = 100
        self.N = int(self.N / 100)

        for i in range(0, self.T):
            nn_array = self.compute_k_nearest(i)
            self.populate(self.N, i, nn_array)
        return self.synthetic

    def compute_k_nearest(self, i):
        nn_array = self.neighbors.kneighbors([self.data[i]], self.k, return_distance=False)
        if len(nn_array) is 1:
            return nn_array[0]
        else:
            return []

    def populate(self, N, i, nn_array):
        while N is not 0:
            nn = random.randint(0, self.k - 1)
            self.synthetic.append([])
            for attr in range(0, len(self.data[i])):
                dif = self.data[nn_array[nn]][attr] - self.data[i][attr]
                gap = random.random()
                self.synthetic[self.newIndex].append(self.data[i][attr] + gap * dif)
            self.newIndex += 1
            N -= 1
        return
