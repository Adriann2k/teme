import warnings; warnings.simplefilter('ignore')
import matplotlib.pyplot as plt
import os
import csv
import numpy as np
from regressor import BatchedGD
from sklearn import linear_model

class Data:
    
    trainInputs = []
    trainOutputs = []
    testInputs = []
    testOutputs = []
    def loadData(self,fileName, inputVariabName, outputVariabName):
        data = []
        dataNames = []
        with open(fileName) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    dataNames = row
                else:
                    data.append(row)
                line_count += 1
        selectedVariable = dataNames.index(inputVariabName)
        inputs = [float(data[i][selectedVariable]) for i in range(len(data))]
        selectedOutput = dataNames.index(outputVariabName)
        outputs = [float(data[i][selectedOutput]) for i in range(len(data))]
        return inputs, outputs
    
    def extractReadData(self):
        crtDir =  os.getcwd()
        filePath = os.path.join(crtDir, '2017.csv')
        inputs, outputs = self.loadData(filePath, 'Economy..GDP.per.Capita.', 'Happiness.Score')
        return inputs, outputs
    
    def splitData(self):
        np.random.seed(5)
        inputs, outputs = self.extractReadData()
        
        indexes = [i for i in range(len(inputs))]
        trainSample = np.random.choice(indexes, int(0.8 * len(inputs)), replace = False)
        testSample = [i for i in indexes  if not i in trainSample]
        self.trainInputs = [inputs[i] for i in trainSample]
        self.trainOutputs = [outputs[i] for i in trainSample]
        self.testInputs = [inputs[i] for i in testSample]
        self.testOutputs = [outputs[i] for i in testSample]
        
    def getTrainData(self):
        self.splitData()
        return self.trainInputs,self.trainOutputs
    
    def getTestData(self):
        self.splitData()
        return self.testInputs,self.testOutputs
    
    
class GD:
    
    data = Data()
    regressor = BatchedGD() #manual
    #regressor = linear_model.SGDRegressor(max_iter =  10000) #tool
    def plotData(self, x1, y1, x2 = None, y2 = None, x3 = None, y3 = None, title = None):
        plt.plot(x1, y1, 'ro', label = 'train data')
        if (x2):
            plt.plot(x2, y2, 'b-', label = 'learnt model')
        if (x3):
            plt.plot(x3, y3, 'g^', label = 'test data')
        plt.title(title)
        plt.legend()
        plt.show()
    
    def learnModel(self):
        trainInputs,trainOutputs = self.data.getTrainData()
        xx = [[el] for el in trainInputs]
        #manual
        self.regressor.fit(xx, trainOutputs)
        w0, w1 = self.regressor.intercept_, self.regressor.coef_[0]
        
        #tool
#         self.regressor.fit(xx, trainOutputs)
#         #save the model parameters
#         w0, w1 = self.regressor.intercept_[0], self.regressor.coef_[0]
        return w0,w1
    
    def predictBasedOnTestData(self):
        testInputs,testOutputs = self.data.getTestData()
        w0,w1=self.learnModel()
        computedTestOutputs = [w0 + w1 * el for el in testInputs] #manual
        #computedTestOutputs = self.regressor.predict([[x] for x in testInputs]) #tool
        plt.plot(testInputs, computedTestOutputs, 'yo', label = 'computed test data')
        plt.plot(testInputs, testOutputs, 'g^', label = 'real test data') 
        plt.title('computed test and real test data')
        plt.xlabel('GDP capital')
        plt.ylabel('happiness')
        plt.legend()
        plt.show()
        return computedTestOutputs
        
        
    def univariateRegression(self):
        data = Data()
        trainInputs,trainOutputs = data.getTrainData()
        testInputs,testOutputs = data.getTestData()
        b0,b1 = self.learnModel()
        print('model: f(x)=', b0, '+', b1, '*x')    
        error = 0.0
        computedTestOutputs = self.predictBasedOnTestData()
        for t1, t2 in zip(computedTestOutputs, testOutputs):
            error += (t1-t2) * (t1-t2)
        error = error / len(testOutputs)
        print("prediction error (manual): ", error)
        self.plotData([], [], testInputs, computedTestOutputs, testInputs, testOutputs, "predictions vs real test data")
        
        from sklearn.metrics import mean_squared_error
        error = mean_squared_error(testOutputs, computedTestOutputs)
        print("prediction error (tool): ", error)
    
    
    
    
    #######################       TESTS           ##################################
class Tests:

    gd = GD()
    def testSplit(self):
        trainInputs,trainOutputs = self.gd.data.getTrainData()
        testInputs,testOutputs = self.gd.data.getTestData() 
        plt.plot(trainInputs, trainOutputs, 'ro', label = 'training data')
        plt.plot(testInputs, testOutputs, 'g^', label = 'testing data')
        plt.title('train and test data')
        plt.xlabel('GDP capita')
        plt.ylabel('happiness')
        plt.legend()
        plt.show()
        
    def testCitire(self):
        inputs,outputs = self.gd.data.extractReadData()
        print('in:  ', inputs[:15])
        print('out: ', outputs[:15])
        plt.plot(inputs, outputs, 'yo') 
        plt.xlabel('PIB')
        plt.ylabel('Fericire')
        plt.title('Fericirea in functie de PIB')
        plt.show()
        
    def testLearnModel(self):
        w0,w1=self.gd.learnModel()
        trainInputs,trainOutputs= self.gd.data.getTrainData()
        print('model: f(x)=', w0, '+', w1, '*x')
        noOfPoints = 1000
        xref = []
        val = min(trainInputs)
        step = (max(trainInputs) - min(trainInputs)) / noOfPoints
        for i in range(1, noOfPoints):
            xref.append(val)
            val += step
        yref = [w0 + w1 * el for el in xref] 
        plt.plot(trainInputs, trainOutputs, 'yo', label = 'training data')
        plt.plot(xref, yref, 'k-', label = 'Model invatat')
        plt.title('train data and the learnt model')
        plt.xlabel('GDP capita')
        plt.ylabel('happiness')
        plt.legend()
        plt.show()
    
    def testGD(self):
        self.gd.univariateRegression()
    

test = Tests()
# test.testCitire()
# test.testSplit()
# test.testLearnModel()
test.testGD()