'''
Created on August 27, 2011

@author: Arnaud Rachez
'''
from numpy import *
#from pylab import *

class SelfOrganizingMap:
    def __init__(self,nbNeurons,ndim,type='line',tf=1000.):
        self.type = type
        self.n = nbNeurons
        self.W = zeros((nbNeurons,1))
        self.ref = 0.01*random.rand(nbNeurons,ndim)
        self.t = 1
        self.tf = float(tf)
        
    def getWinner(self,y):
        d = [linalg.norm(c_[x]-y) for x in self.ref]
        self.winner = min(d)
        self.wInd = d.index(self.winner)
    
    def updateWeights(self):
        if self.type == 'line':
            sigma = 0.4*self.n*(1./(0.6*self.n))**(float(self.t)/self.tf)+1.
            for j in range(0,self.n):
                self.W[j] = exp(-linalg.norm(self.wInd-j)**2/(2*sigma**2))
        if self.type == 'grid':
            sigma = 0.4*sqrt(self.n)*(1./(0.6*sqrt(self.n)))**(2*float(self.t)/self.tf)
            for j in range(0,self.n):
                ri = [self.wInd/sqrt(self.n), self.wInd%sqrt(self.n)]
                rj = [j/sqrt(self.n), j%sqrt(self.n)]
                self.W[j] = exp(-linalg.norm(c_[ri]-c_[rj])**2/(2*sigma**2))
        self.t += 1             
    
    def organize(self,y):
        alpha = 0.5*(0.01/0.5)**(float(self.t)/self.tf)
        self.getWinner(y)
        self.updateWeights()
        self.ref += alpha*self.W*(dot(y,ones((self.ref.shape[0],1)).T).T - self.ref)
        
    def display(self):
        if self.type == 'line':
            n = self.n
            [rx,ry] = [[r[0] for r in self.ref],[r[1] for r in self.ref]]
            plot(rx,ry,'b')
        if self.type == 'grid':
            n = int(sqrt(self.n))
            for i in range(0,n):
                [rx,ry] = [[self.ref[i*n+j][0] for j in range(0,n)],[self.ref[i*n+j][1] for j in range(0,n)]]
                plot(rx,ry,'b')
                [rx,ry] = [[self.ref[i+j*n][0] for j in range(0,n)],[self.ref[i+j*n][1] for j in range(0,n)]]
                plot(rx,ry,'b')
        
if __name__ == '__main__':
    print "plop"
    distrib = [1*(1-2*random.rand(2,1)) for i in range(0,5000)]
    som = SelfOrganizingMap(20*20,2,type='grid',tf=len(distrib))
    
    ion()
    for y in distrib:
        som.organize(y)
        clf()
        som.display()
        draw()
