import warnings; warnings.simplefilter('ignore')
import matplotlib.pyplot as plt
import os
import csv
import numpy as np
from regressor import BatchedGD
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from math import sqrt
from mpl_toolkits import mplot3d

class Data:

    def plot3Ddata(self, x1Train, x2Train, yTrain, x1Model = None, x2Model = None, yModel = None, x1Test = None, x2Test = None, yTest = None, title = None):

        ax = plt.axes(projection = '3d')
        if (x1Train):
            plt.scatter(x1Train, x2Train, yTrain, c = 'r', marker = 'o', label = 'train data') 
        if (x1Model):
            plt.scatter(x1Model, x2Model, yModel, c = 'b', marker = '_', label = 'learnt model') 
        if (x1Test):
            plt.scatter(x1Test, x2Test, yTest, c = 'g', marker = '^', label = 'test data')  
        plt.title(title)
        ax.set_xlabel("capital")
        ax.set_ylabel("freedom")    
        ax.set_zlabel("happiness")
        plt.legend()
        plt.show()
        
    def normalisation(self, trainData, testData):
        scaler = StandardScaler()
        if not isinstance(trainData[0], list):
            trainData = [[d] for d in trainData]
            testData = [[d] for d in testData]
            
            scaler.fit(trainData)
            normalisedTrainData = scaler.transform(trainData)
            normalisedTestData = scaler.transform(testData)
            
            normalisedTrainData = [el[0] for el in normalisedTrainData]
            normalisedTestData = [el[0] for el in normalisedTestData]
        else:
            scaler.fit(trainData)
            normalisedTrainData = scaler.transform(trainData)
            normalisedTestData = scaler.transform(testData)
        return normalisedTrainData, normalisedTestData
    
    trainInputs = []
    trainOutputs = []
    testInputs = []
    testOutputs = []
    feature1 = []
    feature2 = []
    
    def getFeature1(self):
        self.extractReadData()
        return self.feature1
    
    def getFeature2(self):
        self.extractReadData(   )
        return self.feature2
        
    def loadDataMoreInputs(self, fileName, inputVariabNames, outputVariabName):
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
        selectedVariable1 = dataNames.index(inputVariabNames[0])
        selectedVariable2 = dataNames.index(inputVariabNames[1])
        inputs = [[float(data[i][selectedVariable1]), float(data[i][selectedVariable2])] for i in range(len(data))]
        selectedOutput = dataNames.index(outputVariabName)
        outputs = [float(data[i][selectedOutput]) for i in range(len(data))]
        
        return inputs, outputs
    
    def extractReadData(self):
        crtDir =  os.getcwd()
        filePath = os.path.join(crtDir, '2017.csv')
        inputs, outputs = self.loadDataMoreInputs(filePath, ['Economy..GDP.per.Capita.', 'Freedom'], 'Happiness.Score')
        self.feature1 = [ex[0] for ex in inputs]
        self.feature2 = [ex[1] for ex in inputs]
        return inputs, outputs
    
    
    
    def splitData(self):
        inputs,outputs = self.extractReadData()
        np.random.seed(5)
        indexes = [i for i in range(len(inputs))]
        trainSample = np.random.choice(indexes, int(0.8 * len(inputs)), replace = False)
        testSample = [i for i in indexes  if not i in trainSample]
        
        self.trainInputs = [inputs[i] for i in trainSample]
        self.trainOutputs = [outputs[i] for i in trainSample]
        self.testInputs = [inputs[i] for i in testSample]
        self.testOutputs = [outputs[i] for i in testSample]
        
        
        self.trainInputs, self.testInputs = self.normalisation(self.trainInputs, self.testInputs)
        self.trainOutputs, self.testOutputs = self.normalisation(self.trainOutputs, self.testOutputs)
        
        feature1train = [ex[0] for ex in self.trainInputs]
        feature2train = [ex[1] for ex in self.trainInputs]
        
        feature1test = [ex[0] for ex in self.testInputs]
        feature2test = [ex[1] for ex in self.testInputs]
        
        return feature1test,feature2test,feature1train,feature2train
        
    def getTrainData(self):
        self.splitData()
        return self.trainInputs,self.trainOutputs
    
    def getTestData(self):
        self.splitData()
        return self.testInputs,self.testOutputs
    
    
class GD:
    
    data = Data()
    regressor = BatchedGD() #manual
    #regressor = linear_model.SGDRegressor() #tool
    
    def learnModel(self):
        trainInputs,trainOutputs = self.data.getTrainData()
        #manual
        self.regressor.fit(trainInputs, trainOutputs)
        w0, w1, w2 = self.regressor.intercept_, self.regressor.coef_[0], self.regressor.coef_[1]
        
        #tool
#         self.regressor.fit(xx, trainOutputs)
#         #save the model parameters
#         w0, w1 = self.regressor.intercept_[0], self.regressor.coef_[0]
        return w0,w1,w2
    
    def predictBasedOnTestData(self):
        testInputs,testOutputs = self.data.getTestData()
        w0,w1,w2=self.learnModel()
        computedTestOutputs = [w0 + w1 * el[0] + w2 * el[1] for el in testInputs] #manual
        #computedTestOutputs = self. regressor.predict(testInputs) #tool
        feature1test,feature2test,feature1train,feature2train = self.data.splitData()
        self.data.plot3Ddata([], [], [], feature1test, feature2test, computedTestOutputs, feature1test, feature2test, testOutputs, 'predictions vs real test data')
        return computedTestOutputs
        
        
    def multivariateRegression(self):
        data = Data()
        trainInputs,trainOutputs = data.getTrainData()
        testInputs,testOutputs = data.getTestData()
        b0,b1,b2 = self.learnModel()
        print('model: f(x)=', b0, '+', b1, '*x')    
        error = 0.0
        computedTestOutputs=self.predictBasedOnTestData()
        for t1, t2 in zip(computedTestOutputs, testOutputs):
            error += (t1 - t2) * (t1-t2)
        error = error / len(testOutputs)
        print('prediction error (manual): ', error)
        
        from sklearn.metrics import mean_squared_error
        
        error = mean_squared_error(testOutputs, computedTestOutputs)
        print('prediction error (tool):   ', error)
    
    
    
    
    #######################       TESTS           ##################################
class Tests:

    gd = GD()
    def testSplit(self):
        trainInputs,trainOutputs = self.gd.data.getTrainData()
        testInputs,testOutputs = self.gd.data.getTestData() 
        feature1test,feature2test,feature1train,feature2train = self.gd.data.splitData()
        self.gd.data.plot3Ddata(feature1train, feature2train, trainOutputs, [], [], [], feature1test, feature2test, testOutputs, "train and test data (after normalisation)")

        
    def testCitire(self):
        inputs,outputs = self.gd.data.extractReadData()
        feature1 = [ex[0] for ex in inputs]
        feature2 = [ex[1] for ex in inputs]
        print('in:  ', inputs[:15])
        print('out: ', outputs[:15])
        self.gd.data.plot3Ddata(feature1, feature2, outputs, [], [], [], [], [], [], 'capital vs freedom vs happiness')
        
    def testLearnModel(self):
        w0,w1,w2=self.gd.learnModel()
        inputs,outputs = self.gd.data.extractReadData() #raw data
        trainInputs, trainOutputs = self.gd.data.getTrainData() #datele de antrenament normalizate
        feature1 = [ex[0] for ex in trainInputs]
        feature2 = [ex[1] for ex in trainInputs]
        print('model: f(x)=', w0, '+', w1, '*x')
        noOfPoints = 50
        xref1 = []
        val = min(feature1)
        step1 = (max(feature1) - min(feature1)) / noOfPoints
        for _ in range(1, noOfPoints):
            for _ in range(1, noOfPoints):
                xref1.append(val)
            val += step1
        
        xref2 = []
        val = min(feature2)
        step2 = (max(feature2) - min(feature2)) / noOfPoints
        for _ in range(1, noOfPoints):
            aux = val
            for _ in range(1, noOfPoints):
                xref2.append(aux)
                aux += step2
        yref = [w0 + w1 * el1 + w2 * el2 for el1, el2 in zip(xref1, xref2)]
        feature1test,feature2test,feature1train,feature2train = self.gd.data.splitData()
        self.gd.data.plot3Ddata(feature1train, feature2train, trainOutputs, xref1, xref2, yref, [], [], [], 'train data and the learnt model')
    
    def testGD(self):
        self.gd.multivariateRegression()
    

test = Tests()
#test.testCitire()
#test.testSplit()
test.testLearnModel()
test.testGD()