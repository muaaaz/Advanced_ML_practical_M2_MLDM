#
# kpca based on numpy
#

from numpy import ones, exp
from numpy.linalg import eig, norm

VERBOSE = False
def __DEBUG(msg):
	if VERBOSE: print(msg)

K2_SIGMA = 0.007

def polynomial_kernel(X, Y):
	return (X.T.dot(Y) + 1) ** 2

def gaussian_kernel(X, Y):
	return exp((-1 * (norm(X - Y) ** 2)) / (2 * (K2_SIGMA ** 2)))

def k_cov(A, kernel):
	AT = A.T
	n = A.shape[0]
	d = A.shape[1]
	COV = ones((d, d))
	for i in range(d):
		for j in range(d):
			COV[i, j] = kernel(AT[i], AT[j])

	#ONE_OVER_N = ones((d, d)) / n	
	#C_COV = COV - 2 * ONE_OVER_N.dot(COV) + ONE_OVER_N.dot(COV.dot(ONE_OVER_N)) 	

	COV_SUM = COV.sum() / (n ** 2)
	COV_SUMROWS = COV.sum(axis=1) / n

	C_COV = ones((d, d))
	for i in range(d):
		for j in range(d):
			C_COV[i, j] = COV[i, j] - COV_SUMROWS[i] - COV_SUMROWS[j] + COV_SUM
	
	return C_COV
	
def kpca(A, kernel):
	__DEBUG("A = " + str(A))	
	# calculate the kernelized covariance matarix of data
	KCOV = k_cov(A, kernel)
	__DEBUG("KCOV = " + str(KCOV))
	# eigendecoposition of kernelized covariance matrix
	eig_values, eig_vectors = eig(KCOV)
	__DEBUG("eig_values  = " +  str(eig_values))
	__DEBUG("eig_vectors = " +  str(eig_vectors))
	# project data
	P = eig_vectors.T.dot(A.T)
	__DEBUG("P.T = " + str(P.T))
	return P.T


if __name__ == "__main__":
	from numpy import array
	VERBOSE = True

	A = array([[1, 2], [3, 4], [5, 6]])
	kpca(A, polynomial_kernel)
	

