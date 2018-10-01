#
# kpca based on numpy
#

import matplotlib.pyplot as plt

from numpy import ones, exp, loadtxt
from numpy.linalg import eig, norm

VERBOSE = False
def __DEBUG(msg):
	if VERBOSE: print(msg)

K2_SIGMA = 0.007

def polynomial_kernel(X, Y):
	return (X.T.dot(Y) + 1) ** 2

def gaussian_kernel(X, Y):
	return exp((-1 * (norm(X - Y) ** 2)) / (2 * (K2_SIGMA ** 2)))

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
	
	#ONE_OVER_N = ones((d, d)) / n	
	#C_COV = COV - 2 * ONE_OVER_N.dot(COV) + ONE_OVER_N.dot(COV.dot(ONE_OVER_N)) 	
	
	return K_
	
def kpca(A, kernel, Y):

	

	n = A.shape[0]
	d = A.shape[1]
	#__DEBUG("A = " + str(A))	
	# calculate the kernelized matrix of data
	K = k_matrix(A, kernel)
	#__DEBUG("K = " + str(K))
	# check if the matrix is centralized around zero
	#__DEBUG("sum K = " + str(K.sum()) ) # the sum should be zero!
	
	# eigendecoposition of kernelized covariance matrix
	eig_values, eig_vectors = eig(K)
	idx = eig_values.argsort()[::-1]   
	eig_values = eig_values[idx]
	eig_vectors = eig_vectors[:,idx]
	#__DEBUG("eig_values  = \n" +  str(eig_values))
	#__DEBUG("eig_vectors = \n" +  str(eig_vectors))
	
	# project data (only the first d component) d: the number of features in the original space
	sub_eig_vectors = eig_vectors[0:d,:]
	#__DEBUG("sub_eig_vectors = \n" +  str(sub_eig_vectors))
	
	A_new = ones((n,d))
	for i in range(n):
		for j in range(d):
			coco = 0
			for z in range(n):
				coco = coco + eig_vectors[j,z]*kernel(A[i],A[z])
			A_new[i,j] = coco
			
	#__DEBUG("A_new = \n" + str(A_new))

	plt.figure(1, figsize=(8, 6))
	plt.clf()
	plt.scatter(A[:, 0], A[:, 1], c=Y,s=25, edgecolor='k')
    
	plt.figure(2, figsize=(8, 6))
	plt.clf()
	plt.scatter(A_new[:, 0], A_new[:, 1], c=Y,s=25, edgecolor='k')

	return A_new

	
if __name__ == "__main__":
	from numpy import array
	VERBOSE = True

	data = loadtxt("C:\\Users\\Muaz\\Desktop\\Advanced_ML_practical_M2_MLDM\\data.data")
	#__DEBUG("data : \n" + str(data))
	d = data.shape[1]
	A = data[:,0:(d-1)]
	Y = data[:,(d-1):d].T[0]
	#__DEBUG("A : \n" + str(A))
	#__DEBUG("Y : \n" + str(Y))
	kpca(A, polynomial_kernel,Y)
	
