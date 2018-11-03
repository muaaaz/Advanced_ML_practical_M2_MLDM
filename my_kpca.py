#
# kpca based on numpy
#

import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from numpy import ones, exp, loadtxt, tanh
from numpy.linalg import eig, norm
from sklearn.preprocessing import normalize, scale
import numpy as np

VERBOSE = False
def __DEBUG(msg):
	if VERBOSE: print(msg)

fig = 1
K2_SIGMA = 0.007

def liner_kernel(X,Y):
	return np.dot(X,Y)

def polynomial_kernel(X, Y):
	return (X.T.dot(Y) + 1) ** 2

def gaussian_kernel(X, Y):
	return exp( (-1 * (norm(X - Y) ** 2)) / (2 * (K2_SIGMA ** 2)) )

def k_matrix(A, kernel):

	n = A.shape[0]
	d = A.shape[1]
	K = ones((n, n))
	for i in range(n):
		for j in range(n):
			K[i, j] = kernel(A[i], A[j])

	K_SUM = K.sum() / (n ** 2)
	K_SUMROWS = K.sum(axis=1) / n

	K_ = ones((n, n))
	for i in range(n):
		for j in range(n):
			K_[i, j] = K[i, j] - K_SUMROWS[i] - K_SUMROWS[j] + K_SUM
		
	return K_
	
def kpca(A, kernel):
	n = A.shape[0]
	d = A.shape[1]
	# calculate the kernelized matrix of data
	K = k_matrix(A, kernel)
	# check if the matrix is centralized around zero
	#__DEBUG("sum K = " + str(K.sum()) ) # the sum should be zero!
	
	# eigendecoposition of kernelized covariance matrix
	eig_values, eig_vectors = eig(K)
	idx = eig_values.argsort()[::-1]   
	eig_values = eig_values[idx]
	eig_vectors = eig_vectors[:,idx]
	
	# project data (only the first d component) d: the number of features in the original space
	sub_eig_vectors = eig_vectors[0:d,:]
	#__DEBUG("sub_eig_vectors = \n" +  str(sub_eig_vectors))
	
	A_new = ones((n,d))
	for i in range(n):
		for j in range(d):
			temp = 0
			for z in range(n):
				temp += eig_vectors[j,z]*kernel(A[i],A[z])
			A_new[i,j] = temp
			
	#__DEBUG("A_new = \n" + str(A_new))
	return A_new


from numpy import array
#VERBOSE = True

data = loadtxt("data.data")
#__DEBUG("data : \n" + str(data))
d = data.shape[1]
A = data[:,0:(d-1)]
Y = data[:,(d-1):d].T[0]
	
plt.figure(fig, figsize=(8, 6))
plt.clf()
plt.scatter(A[:, 0], A[:, 1], c=Y,s=25, edgecolor='k')
plt.show()
fig += 1

A_new = kpca(A, polynomial_kernel)
plt.figure(fig, figsize=(8, 6))
plt.clf()
plt.scatter(A_new[:, 0], A_new[:, 1], c=Y,s=25, edgecolor='k')
plt.show()
fig += 1

sigma_array = [0.09,0.01,0.1,0.3,0.5,1,2]

for sg in sigma_array:
	K2_SIGMA = sg
	A_new = kpca(A, gaussian_kernel)
	plt.figure(fig, figsize=(8, 6))
	plt.clf()
	plt.scatter(A_new[:, 0], A_new[:, 1], c=Y,s=25, edgecolor='k')
	plt.show()
	fig += 1



