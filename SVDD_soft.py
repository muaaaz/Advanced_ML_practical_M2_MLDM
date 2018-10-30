import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import math
from numpy import ones, exp, loadtxt, tanh
from numpy.linalg import eig, norm
SIGMA = 0.01
# kerenl function
def linear(X,Y):
    #no kernel 
    return np.dot(X,Y)
cc = 0
# gaussian_kernel(X,Y):
kernel_arr = [[1,2],[3,4]]

def polynomial(X,Y):
    return (X.T.dot(Y) + 1) ** 2

def run(X,Y,kernel):
    
    
    n = X.shape[0] # number of entries
    m = X.shape[1] # number of features

    # function to optimize
    def L(a,*args):
        ret = 0
        for i in range(len(a)):
            ret += a[i]*kernel_arr[i,i]
        for i in range(len(a)):
            for j in range(len(a)):
                ret -= a[i]*a[j]*kernel_arr[i,j]
        return -1 * ret # we add -1 because we want to mazimize the function instead of minimizing

    # partial derivative of L regarding the alphas
    def dL(a,*args):
        da = np.zeros(len(a))
        for i in range(len(a)):
            da[i] = kernel_arr[i,i]
            for j in range(len(a)):
                da[i] -= a[j]*kernel_arr[i,j]
        return -1 * np.array( da ,float) # we add -1 because we want to mazimize the function instead of minimizing

    import scipy.optimize as optimize
    
    # values of the constant C
    best_accuracy = 0
    best_C = 0
    best_colors = []

    C0 = [i/10000 for i in range(1,10)]
    C1 = [i/1000 for i in range(1,10)]
    C2 = [i/100 for i in range(1,10)]
    C3 = [i/10 for i in range(1,5)]

    C_arr = C0 + C1 + C2 + C3

    for C in C_arr:
        #print('.',end='')
        cc += 1
        sys.stdout.flush()
        x0 = [0 for _ in range(n)] # initial solution
        # solve for alphas
        sys.stdout = open(os.devnull, "w") # this line will prevent the function from printing in the batch console
        alpha = optimize.fmin_slsqp(L,x0, fprime=dL, eqcons=[lambda x: sum(x) - 1 ], bounds = [(0,C) for _ in range(n)] , full_output =False )
        sys.stdout = sys.__stdout__ # this line will restor default setting of printing
        if math.isnan(sum(alpha)) or sum(alpha) < 0.9 or sum(alpha) > 1.1:
            continue
        #print(sum(alpha))
        sys.stdout.flush()
        eps = 1e-12
        # plotting
        def calc_colore(i):
            #not SV alpha[i] == 0
            if alpha[i] < eps:
                return 0
            #outlyer if alpha[i] >= C
            if alpha[i] > C - eps:
                return 1
            #support vector if alpha[i] < C
            return 3

        colores = [calc_colore(i) for i in range(n)]
        cur_accuracy = np.mean([(Y[i] == 0 and colores[i] != 1) or (Y[i] == 1 and colores[i] == 1) for i in range(n)])
        # save the best results
        if cur_accuracy > best_accuracy:
            best_accuracy = cur_accuracy
            best_C = C
            best_colors = colores
    
    return best_accuracy,best_colors,best_C

from sklearn.datasets import make_moons

X,Y = make_moons(n_samples=50, shuffle=False, noise=.05, random_state=0)
X = X[0:30,:]
Y = Y[:30]
print(Y)

plt.figure(0, figsize=(8,6))
plt.clf()
plt.scatter(X[:,0], X[:,1], c=Y , s=25, edgecolor='k')
plt.show()

total_accuracy = -5
total_colors = [] 
total_C = 0
total_sigma = 0
from scipy.spatial.distance import pdist, squareform

S0 = [i/10000 for i in range(1,10)]
S1 = [i/1000 for i in range(1,10)]
S2 = [i/100 for i in range(1,10)]
S3 = [i/10 for i in range(1,10)]
S4 = [i for i in range(1,10)]
SIGMA_arr = S0 + S1 + S2 + S3 + S4

ii = 0
for sg in SIGMA_arr:
    if ii % 10 == 0:
        print(int(ii/10),end = ' ')
        sys.stdout.flush()
    ii += 1
    pairwise_sq_dists = squareform(pdist(X, 'sqeuclidean'))
    kernel_arr = np.exp(-pairwise_sq_dists / sg**2)
    #print()
    #SIGMA = sg
    accuracy,colors,C = run(X,Y,linear)
    #print(accuracy,colors,C)
    #print(kernel_arr)
    if accuracy > total_accuracy:
        total_accuracy = accuracy
        total_colors = colors
        total_C = C
        total_sigma = sg

print("best acc: ",total_accuracy)
print("colores:",total_colors)
print("total_C",total_C)
print("total_sigma",total_sigma)
print("cc",cc)
#plt.figure(0, figsize=(8,6))
#plt.clf()
#plt.scatter(X[:,0], X[:,1], c=total_colors , s=25, edgecolor='k')
#plt.show()