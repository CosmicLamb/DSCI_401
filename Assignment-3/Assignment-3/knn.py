# First create a new file called “knn.py” 
# inside this file define a Python class** called KNN containing 
# predict and fit methods. 
# be able to create a KNN instance by specifying a value for k as well as a distance function 
# You should also implement a fit method (which does nothing in the case of KNN) and a predict method. 
# Here is some sample code showing how your class should be used:

#    import knn # Import your knn class

#    def euclidean_dist(x1, x2):
#        # .... compute euclidean distance between vectors x1 and x2

#    data_x, data_y = ... # Get X and y data
#    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.3, random_state = 4)

#    knn = KNN(3, euclidean) # Create a 3-NN algorithm with Euclidean distance
#    knn.fit(x_train, y_train)
#    y_hat = knn.predict(x_test)

#    # .... Print out error metrics between y_test and y_hat

# Your predict method should check to see whether the incoming data X is a pandas data frame or a numpy array
# if it is a data frame 
# 	 you can convert to a numpy array by using X.values() method.
# then perform your calculations. 
# any distance function should take two row vectors and return the distance between them 
# examples of distance functions are Euclidean, Manhattan Edge, and Jaccard Distance.


# You should use your KNN method in your notebook in part 2 as one of the methods you try. Simply import your
# “knn.py” file (store in the same folder as your notebook and import as shown above) and then use KNN like you
# would any other algorithm. You may also want to systematically try different values of k to arrive at the best model.

import numpy as np
import pandas as pd
from scipy import stats

class KNN:
	def __init__(self, k, distance):
		self.k = k
		self.distance = distance
	

	def predict(self, X):
		
		# Make sure we're working in numpy
		if isinstance(X, pd.DataFrame):
			X = np.asarray(X)

		# And now for the main act
		if isinstance(X, np.ndarray):
			
			Y = []

			# Go through all of the test data X
			for i in range(len(X)):
				
				print("Calculating distances from position" + str(i+1) + "...\n")
				# Calculate all the distance
				response = []
				dist = [] 
				for j in np.nditer(self.x_train):
					dist.append(distance(X[i], self.x_train[j]))
					response.append(self.y_train[j])
				
				print("Analyzing nearby positions...\n")
				# Take the k smallest distance values
				smallest = []
				while len(smallest) < k:
					minimum = distance[0]
					for r in range(len(distance)):
						if distance[r] < minimum:
							minimum = r
					dist.remove(distance[r])
					response.remove(response[r])
					smallest.append(self.y_train[r])

				# Average their response variables & set as i's response variable
				Y.append(stats.mode(smallest))

		return (Y)

	def fit(self, x_train, y_train):
		self.x_train = x_train
		self.ytrain = y_train
