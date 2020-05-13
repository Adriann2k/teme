'''
Created on May 12, 2020

@author: Adrian
'''

class BatchedGD:
    def __init__(self):
        self.intercept_ = 0.0 #w0
        self.coef_ = [] #pantele
                
  
    def fit(self, x, y, learningRate = 0.005, nrdataSets = 1000):
        self.coef_ = [0.0 for _ in range(len(x[0]) + 1)]
        errorSum = 0
        howManyErrors = 0
        for dataSet in range(nrdataSets):
            for i in range(len(x)):
                guess = self.eval(x[i])
                crtError = guess - y[i]
                errorSum += crtError
                howManyErrors += 1
                
            for j in range(0, len(x[0])): #update pante
                self.coef_[j] = self.coef_[j] - learningRate * errorSum/howManyErrors * x[i][j] #update pante
            self.coef_[len(x[0])] = self.coef_[len(x[0])] - learningRate * errorSum/howManyErrors #update w0
     
        self.intercept_ = self.coef_[-1] #new w0
        self.coef_ = self.coef_[:-1] #new pante

    def eval(self, xi):
        yi = self.coef_[-1]
        for j in range(len(xi)):
            yi += self.coef_[j] * xi[j]
        return yi 

    def predict(self, x):
        yComputed = [self.eval(xi) for xi in x]
        return yComputed