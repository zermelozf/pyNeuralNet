'''
Created on August 26, 2011

@author: Arnaud Rachez
'''

from numpy import ones, zeros, random, sqrt, dot, tanh, exp, c_

class MultilayerPerceptron:
    """
    A Multilayer Perceptron using the numpy library. The network can have multiple hidden layers.
    """    
    def __init__(self,T,bias=True, linOutput=True):       
        if bias == True:
            T[0] = T[0]+1   #bias
        self.T = list(T)
        self.H = [ones((n,1)) for n in self.T]
        self.S = [zeros((n,1)) for n in self.T]
        self.M = [(-ones((self.T[d],self.T[d+1])).T + 2*random.rand(self.T[d],self.T[d+1]).T)/sqrt(max(T[d],T[d+1])) for d in range(0,len(self.T)-1)]
        self.E = list(self.H)
        self.DM = list(self.M)
        self.pDM = list(self.DM)
        self.beta = 1.
        self.linOutput =linOutput
        
    def process(self,X):
        self.H[0][0:len(X)] = X     #bias
        self.S[0] = self.H[0]
        for l in range(0,len(self.T)-1):
            self.H[l+1] = dot(self.M[l],self.S[l])
            self.S[l+1] = self.sig(self.H[l+1],self.beta)
        if self.linOutput == True:
            self.S[len(self.T)-1] = self.H[len(self.T)-1]
        return self.S[len(self.T)-1]
            
    def learn(self,X,Y, alpha = 0.01, momentum = 0.):
        self.process(X)
        m = len(self.T)-1
        self.E[m] = Y-self.S[m]
        for l in range(m,0,-1):
            self.E[l-1] = self.dersig(self.H[l-1],self.beta)*dot(self.M[l-1].T,self.E[l])
            self.DM[l-1] = alpha*dot(self.E[l],self.S[l-1].T)
            self.M[l-1] = self.M[l-1] + self.DM[l-1] + momentum*self.pDM[l-1]
        self.pDM = list(self.DM)
        
        
    def sig(self,x,beta):
        return tanh(beta*x)
        return 1./(1.+exp(-beta*x))
    
    
    def dersig(self,x,beta):
        return beta*(1.-self.sig(x,beta)**2)
        return beta*(1.-self.sig(x,beta))*self.sig(x,beta)
            
if __name__ == "__main__":
    print "Learn the XOR function."
    X = [[0,0],[0,1],[1,0],[1,1]]
    Y = [0,1,1,0]
    
    mlp = MultilayerPerceptron([2,5,1], bias=True, linOutput=True)
    
    for t in range(200):
        for i in range(0, len(X)):
            mlp.learn(c_[X[i]],c_[Y[i]], alpha=0.1, momentum=0.9)              
    for x in X:
        print x, mlp.process(c_[x])[0]
       
    print "MSE:", 1./len(X)*sum([(Y[i]-mlp.process(c_[X[i]])[0][0]**2) for i in range(0, len(X))])
    