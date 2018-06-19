#By: Alexander Polus
#For: Brent Harrison
#Purpose: Identify pictures of zeros and ones correctly
#Preconditions: training and testing excel files - each row is an instance with first cell as a label and the rest of the cells being pixels of an image value 0-255 for black value (No RGB)
#Postconditions: % accuracy of the model predicting 0 or 1 for each test case
#------------------------------------------------------------------------------------------------

#--------Libraries-------------------------------------------------------------------------------
import csv
import sys
import math
import numpy as np
#------------------------------------------------------------------------------------------------
#--------Validate Usage--------------------------------------------------------------------------
if (len(sys.argv) != 3):
    print("Usage: Python3 NeuralNetwork1.py <Training Filename> <Testing Filename>")
    sys.exit(0)
#------------------------------------------------------------------------------------------------
#--------Useful Global Variables-----------------------------------------------------------------

e = 2.71828
alpha = 0.05

#------------------------------------------------------------------------------------------------
#--------Activation Function and Derivative------------------------------------------------------

def sigmoid(x):
    result = 1 / (1 + e**(-1 * x))
    return result

def sigmoidDeriv(x):
    retult = sigmoid(x) * (1 - sigmoid(x))
    return result

#------------------------------------------------------------------------------------------------
#--------Main------------------------------------------------------------------------------------
#make 2D array from training CSV
trainfilename = sys.argv[-2]
trainfile = open(trainfilename)
trainreader = csv.reader(trainfile, delimiter=',')
TempTrainingData = list(trainreader) #each item in this is a row in the csv file
#make everything a float
for i in range(0,len(TempTrainingData)):
    for j in range(0,len(TempTrainingData[i])):
        TempTrainingData[i][j] = float(TempTrainingData[i][j])

#prepare input for training
TrainingData = []
for row in TempTrainingData:
    #print(row)
    TrainingPixels = row[1:]
    TrainingPixels.append(1)                #HERE IS THE BIAS
    TrainingData.append([row[0], np.array(TrainingPixels)]) #Now each entry is [ label, [pixels] ]
#print(TrainingData[0])

#create the skeleton of my weight matrices

#TempRow = np.array([0.1] * 785) #CHANGED FROM 784 TO 785 TO HANDLE BIAS NODE
TempMatrix = np.random.uniform(low=-1, high=1, size = (785,5))
#TempWeights = []
#for i in range(0,5):
#TempWeights.append(TempRow)
InputWeights = np.array(TempMatrix) # now I have a 5x784 matrix for input weights
#InputWeights = np.transpose(InputWeights) # transpose into 784x5 matrix
InputWeights.shape = (785,5)
#HiddenWeights = np.array([0.1, -0.1, 0.2, -0.2, 0.3, -0.3]) # ADDED A SIXTH WEIGHT FOR BIAS NODE
HiddenWeights = np.random.uniform(low=-1, high=1, size = (6,1))
#define my "TrainWalk" function to get a result from one example for training
def TrainWalk(example):
    #print(1)
    global InputWeights
    global HiddenWeights
    example = np.array(example, dtype=float)
    

    #print(example)
    
    #example = np.transpose(example) # transpose example into (1,784)
    
    #print(example.shape)
    #print(InputWeights.shape)
    
    
    RawMiddle = example.dot(InputWeights)
    TempMiddle = []
    for node in RawMiddle:
        TempMiddle.append(sigmoid(node)) #now all hidden nodes should be activated
    TempMiddle.append(1)            #APPENDED THE BIAS NODE FOR MIDDLE LAYER
    ActivatedMiddle = np.array(TempMiddle)

    #print(ActivatedMiddle.shape)
    #print(HiddenWeights.shape)  #ignoring lack of transposition

    RawOutput = ActivatedMiddle.dot(HiddenWeights)
    ActivatedOutput = sigmoid(RawOutput) # raw output is not a 1 element array, just a number
    return [ActivatedMiddle, ActivatedOutput]

#define my "TestWalk" function to get a result, 0 or 1, for testing
def TestWalk(example):
    global InputWeights
    global HiddenWeights
    example = np.array(example, dtype=float)
    
    
    RawMiddle = example.dot(InputWeights)
    TempMiddle = []
    for node in RawMiddle:
        TempMiddle.append(sigmoid(node)) #now all hidden nodes should be activated
    TempMiddle.append(1)        #APPENDED THE BIAS NODE FOR MIDDLE LAYER
    ActivatedMiddle = np.array(TempMiddle)
    RawOutput = ActivatedMiddle.dot(HiddenWeights)
    ActivatedOutput = sigmoid(RawOutput)
    if ActivatedOutput > 0.5:
        ActivatedOutput = 1
    else: ActivatedOutput = 0
    return ActivatedOutput

#"train" my model by walking each example and back propagating error, updating weights

for epoch in range(0,10):   #setup for 10 Epochs
    for row in TrainingData:
        #print("one run")
        actual = row[0]
        example = row[1]
        
        #  1) walk the example to get a decimal-value output
        MiddleAndPredicted = TrainWalk(example)
        ActivatedMiddle = MiddleAndPredicted[0]
        predicted = MiddleAndPredicted[1]

        #  2) Calculate Delta of Output
        DeltaOutput = (actual - predicted) * (predicted * (1 - predicted))

        #  3) Propagate back and find delta at the hidden layers
        DeltaHidden = [0,0,0,0,0,0]
        for i in range(0,5): # only update first 5, not the bias delta
            DeltaHidden[i] = DeltaOutput * HiddenWeights[i] * (ActivatedMiddle[i] * (1 - ActivatedMiddle[i]))
    
    
    
        DeltaHidden[5] = DeltaOutput * HiddenWeights[5] # Update bias Delta
        
        
        
        
        #  4) Update weights between normal hidden layer and output layer
        for i in range(0, len(HiddenWeights) - 1):
            HiddenWeights[i] = HiddenWeights[i] + alpha * ActivatedMiddle[i] * DeltaOutput
        
        #  4.1) Update Bias Weight
        HiddenWeights[5] = HiddenWeights[5] + alpha * DeltaOutput

        #  5) Update weights between normal input layer and hidden layer
        for i in range(0,5):
            for j in range(0, len(InputWeights[i] - 1)):
                InputWeights[i][j] = InputWeights[i][j] + alpha * example[j] * DeltaHidden[i]
            #5.1) Update Bias Weight
            InputWeights[i][-1] = InputWeights[i][-1] + alpha * DeltaHidden[i]
        print(HiddenWeights)


#make 2D array from testing CSV
testfilename = sys.argv[-1]
testfile = open(testfilename)
testreader = csv.reader(testfile, delimiter=',')
TempTestingData = list(testreader) #each item in this is a row in the csv file
#make everything a float
for i in range(0,len(TempTestingData)):
    for j in range(0,len(TempTestingData[i])):
        TempTestingData[i][j] = float(TempTestingData[i][j])


#prepare input for testing
TestingData = []
for row in TempTestingData:
    TestingPixels = row[1:]
    TestingPixels.append(1)
    TestingData.append([row[0], np.array(TestingPixels)]) #Now each entry is [ label, [pixels] ]


#"test" my model over the new data and output the error
num_correct = 0
total = len(TestingData)
for row in TestingData:
    print(TestWalk(row[1]))
    if row[0] == TestWalk(row[1]):
        num_correct = num_correct + 1

percent_correct = num_correct / total * 100
print("This model predicted", percent_correct, "% of test examples correctly")






