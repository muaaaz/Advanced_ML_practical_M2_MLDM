import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering

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


from math import sqrt

K2_SIGMA = sqrt( 1.0/ (2.0*0.3) )
K3_alpha = 2
K3_constatnt = 1
def polynomial_kernel(X, Y):
	return (X.T.dot(Y) + 1) ** 2

def gaussian_kernel(X, Y):
	return np.exp( (-1 * (np.linalg.norm(X - Y) ** 2)) / (2 * (K2_SIGMA ** 2)) )

def rbf_kernel(v1, v2, sigma=1.0):
    return np.exp((-1 * np.linalg.norm(v1 - v2) ** 2) / (2 * sigma ** 2))

def sigmoid_kernel(X, Y):
	return np.tanh(K3_alpha*X.T.dot(Y) + K3_constatnt)

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
	
	ONE_OVER_N = ones((d, d)) / n	
	#C_COV = COV - 2 * ONE_OVER_N.dot(COV) + ONE_OVER_N.dot(COV.dot(ONE_OVER_N)) 	
	return K_

	#return K

def kkmeans_kernel(x,y,kernel):
	return kernel(x,x) + kernel(y,y) - 2*kernel(x,y)

def kkmeans(X, K,kernel, maxIters=10 ,plot_progress = None):
	global fig
	size = len(X)
	centroids_indexes = np.random.choice(np.arange(size), K)
	centroids = X[centroids_indexes,:]
	
	#centroids = [[245.033,-313.628],[0,0]]
	for i in range(maxIters):
		print(centroids)
		# cluster Assignment step
		#print(i)
		#C = np.array([np.argmin([kkmeans_kernel(x_i,y_k, kernel) for y_k in centroids]) for x_i in X])
		#G = k_matrix(X, kernel)
		#C = np.array([np.argmin([G[x_i,y_k] for y_k in range(len(centroids))]) for x_i in range(len(X))])
		C = np.zeros(len(X))
		for x_i in range(len(X)):
			idx = 0
			current_kernel_distance = kkmeans_kernel(X[x_i],centroids[0],kernel)
			#current_kernel_distance = kernel(X[x_i],centroids[0])
			for y_k in range(len(centroids)):
				temp = kkmeans_kernel(X[x_i],centroids[y_k],kernel)
				#temp = kernel(X[x_i],centroids[y_k])
				if temp < current_kernel_distance:
				#if temp > current_kernel_distance:
					current_kernel_distance = temp
					idx = y_k
			C[x_i] = idx

	
		centroids = [X[C == k].mean(axis = 0) for k in range(K)]
			
	return np.array(centroids) , C

	
if __name__ == "__main__":
	data = np.loadtxt("data_k2.data")
	centroids, C = kkmeans(data, 2, gaussian_kernel ,maxIters=50, plot_progress=True)
	fig += 1 
	plt.figure(fig, figsize=(8,6))
	plt.clf()
	plt.scatter(data[:,0], data[:,1], c=C, s=25, edgecolor='k')
	plt.show()
	
	clustering = SpectralClustering(n_clusters=2 ,affinity = "rbf", gamma=0.3, coef0  = 1,assign_labels="discretize", random_state=0).fit(data)
	fig += 1 
	plt.figure(fig, figsize=(8,6))
	plt.clf()
	plt.scatter(data[:,0], data[:,1], c=clustering.labels_, s=25, edgecolor='k')
	plt.show()
	