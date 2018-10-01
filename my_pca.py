#
# pca based on numpy
#

from numpy import mean, cov
from numpy.linalg import eig

VERBOSE = False
def __DEBUG(msg):
	if VERBOSE: print(msg)

def pca(A):
	__DEBUG("A = " + str(A))	
	# calculate the mean of each column
	M = mean(A, axis=0)
	__DEBUG("M = " + str(M))
	# center columns by subtracting columns means 
	C = A - M
	__DEBUG("C = " + str(C))
	# calculate covariance matarix of centered matrix
	COV = cov(C.T) # TODO: calculate manually
	__DEBUG("COV = " + str(COV))
	# eigendecoposition of covariance matrix
	eig_values, eig_vectors = eig(COV)
	__DEBUG("eig_values  = " +  str(eig_values))
	__DEBUG("eig_vectors = " +  str(eig_vectors))
	# project data
	P = eig_vectors.T.dot(C.T)
	__DEBUG("P.T = " + str(P.T))
	return P.T


if __name__ == "__main__":
	from numpy import array
	VERBOSE = True

	A = array([[1, 2], [3, 4], [5, 6]])
	pca(A)
	

