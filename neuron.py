import random
import numpy as np
class Neuron:
    err=0
    Errors=()
    n_iter=0
    weights=()
    alpha=0
    
    def __init__(self, n=1, n_iter=10, alpha=0.1):
        self.Errors*n
        self.err=0
        self.weights*n 6
        self.n_iter=n_iter
        self.alpha=alpha
    
    def tip(self, X, Y, start_w=[]):
        	#Y1=w00+w01*X0+...+w0(n-1)*Xn
		#Y2=w10+w11*X0+...+w1(n-1)*Xn
		#............................
		#Ym=wm0+wm1*X0+...+wm(n-1)*Xn
		for i in weights:
			if(start_w):
				weights[i]=start_w[i]
			else:
				weights[i]=random.randint(1000)/100
				
		for i in range(n_iter):
			for w in weights:
				diff=np.diff(weights)
				Y=0
				for b in X:
					Y+=weight[b]*X[b]
				weight[w]-=alpha*diff/Y
				for b in X:
					Errors[w]+=0.5*(Y-X[0]*weights[1]-weights[0])**2
        
        
    def do(self, X):
        return X*w[1]
