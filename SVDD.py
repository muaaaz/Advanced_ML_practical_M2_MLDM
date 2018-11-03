import numpy as np
import matplotlib.pyplot as plt

X_i = data = np.loadtxt('SVDD_positive.data') 
X_l = data = np.loadtxt('SVDD_negetive.data') 

n_i = X_i.shape[0] # number of positive entries
n_l = X_l.shape[0] # number of negetive entries

m = X_i.shape[1] # number of features
fig = 0

def kernel(x,y):
    return return (x.T.dot(y) + 1) ** 2
    #return np.dot(x,y)
    
# function to optimize
def L(a,*args):
    a_i = a[:n_i]
    a_l = a[n_i:]
    
    ret = 0
    for i in range(n_i):
        ret += a_i[i]*kernel(X_i[i],X_i[i])
    for i in range(n_l):
        ret += a_l[i]*kernel(X_l[i],X_l[i])
    
    for i in range(len(a)):
        for j in range(len(a)):
            ret -= a[i]*a[j]*kernel(X[i],X[j]) 
    return -1 * ret # we add -1 because we want to mazimize the function instead of minimizing

# partial derivative of L regarding the alphas
def dL(a,*args):
    da = np.zeros(len(a))
    for i in range(len(a)):
        da[i] = kernel(X[i],X[i])
        for j in range(len(a)):
            da[i] -= a[j]*kernel(X[i],X[j])
    return -1 * np.array( da ,float) # we add -1 because we want to mazimize the function instead of minimizing

import scipy.optimize as optimize

# values of the constant C
C_arr = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
for C in C_arr:
    
    X = X_
    x0 = [0 for _ in range(n)] # initial solution
    # solve for alphas
    alpha = optimize.fmin_slsqp(L,x0, fprime=dL, eqcons=[lambda x: sum(x) - 1 ], bounds = [(0,C) for _ in range(n)] , full_output =False )

    print("alphas:", np.round(alpha,4))
    print("sum: %.4f" % sum(alpha))
    print("ming: %.4f" % min(alpha))

    center = np.array([0 for _ in range(m)],float)
    for i in range(n):
        center += alpha[i]*X[i]
    print( np.round(center,4))

    # plotting
    eps = 1e-6
    def calc_colore(i):
        #not SV
        if alpha[i] < eps:
            return 0
        #outlyer if alpha[i] >= C
        if alpha[i] > C - eps:
            return 1
        #support vector if alpha[i] < C
        return 3

    colores = [calc_colore(i) for i in range(n)]
    # add the center to the data
    X = np.vstack([X,np.array(center)])
    # add colore of the center
    colores.append(2)

    print("colores:",colores)
    plt.figure(fig, figsize=(8,6))
    plt.clf()
    plt.scatter(X[:,0], X[:,1], c=colores , s=25, edgecolor='k')
    plt.show()
    fig += 1

