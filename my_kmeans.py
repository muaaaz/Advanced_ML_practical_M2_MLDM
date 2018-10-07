#
# kmeans based on numpy
#

import numpy as np
import matplotlib.pyplot as plt

fig = 1

def kmeans(X, K, maxIters = 10, plot_progress = None):
	global fig
	size = len(X)
	centroids_indexes = np.random.choice(np.arange(size), K)
	centroids = X[centroids_indexes,:]
	
	for i in range(maxIters):
		# cluster Assignment step
		C = np.array([np.argmin([np.dot(x_i-y_k, x_i-y_k) for y_k in centroids]) for x_i in X])
		# Move centroids
		centroids = [X[C == k].mean(axis = 0) for k in range(K)]
		if plot_progress:
			fig += 1 
			plt.figure(fig, figsize=(8,6))
			plt.clf()
			plt.scatter(X[:,0], X[:,1], c=C, s=25, edgecolor='k')
			plt.show()
			
	return np.array(centroids) , C

K2_SIGMA = 0.007
K3_alpha = 2
K3_constatnt = 1
def polynomial_kernel(X, Y):
	return (X.T.dot(Y) + 1) ** 2

#def gaussian_kernel(X, Y):
#	return exp( (-1 * (norm(X - Y) ** 2)) / (2 * (K2_SIGMA ** 2)) )

#def sigmoid_kernel(X, Y):
#	return tanh(K3_alpha*X.T.dot(Y) + K3_constatnt)


def k_matrix(A, kernel):

	n = A.shape[0]
	d = A.shape[1]
	K = np.ones((n, n))
	for i in range(n):
		for j in range(n):
			K[i, j] = kernel(A[i], A[j])

	K_SUM = K.sum() / (n ** 2)
	K_SUMROWS = K.sum(axis=1) / n

	K_ = np.ones((n, n))
	for i in range(n):
		for j in range(n):
			K_[i, j] = K[i, j] - K_SUMROWS[i] - K_SUMROWS[j] + K_SUM
	
	#ONE_OVER_N = ones((d, d)) / n	
	#C_COV = COV - 2 * ONE_OVER_N.dot(COV) + ONE_OVER_N.dot(COV.dot(ONE_OVER_N)) 	
	
	return K_

def kkmeans(X, K, maxIters = 10, plot_progress = None):
	global fig
	size = len(X)
	centroids_indexes = np.random.choice(np.arange(size), K)
	centroids = X[centroids_indexes,:]
	
	for i in range(maxIters):
		# cluster Assignment step

		#C = np.array([np.argmin([np.dot(x_i-y_k, x_i-y_k) for y_k in centroids]) for x_i in X])
		G = k_matrix(X, polynomial_kernel)
		#C = np.array([np.argmin([G[x_i,y_k] for y_k in range(len(centroids))]) for x_i in range(len(X))])
		C = np.zeros(len(X))
		for x_i in range(len(X)):
			idx = 0
			current_kernel = G[x_i,0]
			for y_k in range(1,len(centroids)):
				if G[x_i, y_k] > current_kernel:
					current_kernel = G[x_i, y_k]
					idx = y_k
			C[x_i] = idx
				
		# Move centroids
		centroids = [X[C == k].mean(axis = 0) for k in range(K)]
	
	if plot_progress:
		fig += 1 
		plt.figure(fig, figsize=(8,6))
		plt.clf()
		plt.scatter(X[:,0], X[:,1], c=C, s=25, edgecolor='k')
		plt.show()
			
	return np.array(centroids) , C



if __name__ == "__main__":
	data = np.loadtxt("data.data")
	centroids, C = kkmeans(data, 3, maxIters=150, plot_progress=True)
	print(centroids)