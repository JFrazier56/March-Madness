import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as mp

SIZE_TRAINING = 0.8
LAMBDA_FOUND = True
STEP_SIZE = 0.00001
LAMBDA = 16

def processData():
    detailed_results = pd.read_csv('RegularizedSeasonDetailed.csv')
    data = detailed_results.values
    np.random.shuffle(data)
    row = int(math.floor(0.8 * data.shape[0]))
    training = data[0:row - 1, :]
    test = data[row:data.shape[0] - 1, :]

    return training, test

def logisticRegressionSGD(dataset, weights, stepSize, lambdaValue, iterations):
    datasetSize = dataset.shape[1]
    order = []
    for i in range(1, dataset.shape[0]):
        order.append(i)
    for k in range(0, iterations):
        random.shuffle(order)
        for i in order:
            for j in range(1, dataset.shape[1]):
                if j == 0 or j == 1 or j == 2:
                    continue
                weightTranspose = np.transpose(weights)
                dotProduct = np.dot(weightTranspose, dataset[i])
                lastTerm = sigmoid(dotProduct)
                partial = dataset[i, j] * (dataset[i, 2] - lastTerm)
                weights[j] -= stepSize * (partial - ((2 / datasetSize) * lambdaValue * weightTranspose[j]))

    return weights

def testingNewWeight(dataset, weights):
    order = []
    for i in range(1, dataset.shape[0]):
        order.append(i)
    random.shuffle(order)
    LogLoss = 0
    for i in order:
        weightTranspose = np.transpose(weights)
        dotProduct = np.dot(weightTranspose, dataset[i])
        lastTerm = sigmoid(dotProduct)
        if (lastTerm >= 0.5):
            yValue = 1.0
        else:
            yValue = 0.0

        if dataset[i][2] == yValue:
            LogLoss += 1.0

    return LogLoss / dataset.shape[0]



def sigmoid(x):
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        return 1

def graphLoss(iterations, all_testloss, title):
    mp.figure()
    mp.plot(iterations, all_testloss, label=title)
    mp.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
              ncol=2, mode="expand", borderaxespad=0.)
    mp.axhline(0, color='black')
    mp.axvline(0, color='black')
    mp.show()

def main():
    if(LAMBDA_FOUND):
        iterations = [1, 3, 5, 10, 15]
        all_testloss = []
        training, test = processData()
        for i in range(0, len(iterations)):
            total_loss = 0
            for j in range(0, 3):
                trainingWeights = createWeights(training)
                resultWeights = logisticRegressionSGD(training, trainingWeights, STEP_SIZE, LAMBDA, iterations[i])
                test_loss = testingNewWeight(test, resultWeights)
                total_loss += test_loss
                print resultWeights
            all_testloss += [total_loss / 3]
            print total_loss / 3

        print all_testloss
        title = 'Logisitic Regression Correctly Predicted versus Iterations on Dataset'
        graphLoss(iterations, all_testloss, title)

    else :
        training, test = processData()
        min_loss = float("-inf")
        best_lambda = -1
        lambdas = [1, 2, 4, 8, 16, 32, 64, 128]
        all_loss = []

        for i in range(0, len(lambdas)):
            total_loss = 0
            for j in range(0, 1):
                trainingWeights = createWeights(training)
                resultWeights = logisticRegressionSGD(training, trainingWeights, STEP_SIZE, lambdas[i], 1)
                test_loss = testingNewWeight(test, resultWeights)
                total_loss += test_loss
            total_loss = total_loss / 1
            if(total_loss > min_loss):
                best_lambda = lambdas[i]
                min_loss = total_loss

            all_loss += [total_loss]
        print all_loss
        print "Best Lambda:", best_lambda
        print "Min Loss:", min_loss
        title = 'Logisitic Regression Correctly Predicted versus Lambda on Dataset'
        graphLoss(lambdas, all_loss, title)


def createWeights(dataset):
    weights = []

    for i in range(0, dataset.shape[1]):
        weights += [0]

    return weights

if __name__ == "__main__":
    main()
