import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as mp

SIZE_TRAINING = 0.8
LAMBDA_FOUND = True
LAMBDA = 1 * (10 ** -5)

def processData():
    detailed_results = pd.read_csv('data\odds_seeds.csv')
    data = detailed_results.values
    np.random.shuffle(data)
    row = int(math.floor(0.8 * data.shape[0]))
    training = data[0:row - 1, 1:]
    trainingy = data[0:row - 1, 0]
    test = data[row:data.shape[0] - 1, 1:]
    testy = data[row:data.shape[0] - 1, 0]
    return training, trainingy, test, testy

def LinearRegression(x, y, w, lambdaValue):
    Ht_H = np.dot(np.transpose(x), x)
    lambda_i = np.multiply(lambdaValue * np.identity(x.shape[1]), x.shape[1])
    Ht_y = np.dot(np.transpose(x), y)
    w = np.dot(np.linalg.inv(np.add(Ht_H, lambda_i)), Ht_y)
    return w

def testingNewWeight(x, y, w):

    SSE = 0
    for i in range(0, x.shape[0]):
        weightTranspose = np.transpose(w)
        dotProduct = np.dot(weightTranspose, x[i])
        SSE += (dotProduct - y[i]) ** 2

    return SSE


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
        training, trainingy, test, testy = processData()

        for i in range(0, len(iterations)):
            total_loss = 0
            for j in range(0, 3):
                trainingWeights = createWeights(training)
                resultWeights = LinearRegression(training, trainingy, trainingWeights, LAMBDA)
                test_loss = testingNewWeight(test, testy, resultWeights)
                total_loss += test_loss
                print resultWeights
            all_testloss += [total_loss / 3]
            print total_loss / 3

        print all_testloss
        title = 'Logisitic Regression Correctly Predicted versus Iterations on Dataset'
        graphLoss(iterations, all_testloss, title)

    else :
        training, trainingy, test, testy = processData()
        min_loss = float("inf")
        best_lambda = -1
        lambdas = [0.001, 0.00001, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256]
        all_loss = []

        for i in range(0, len(lambdas)):
            total_loss = 0
            for j in range(0, 1):
                trainingWeights = createWeights(training)
                resultWeights = LinearRegression(training, trainingy, trainingWeights, lambdas[i])
                print resultWeights
                test_loss = testingNewWeight(test, testy, resultWeights)
                total_loss += test_loss
            total_loss = total_loss / 1
            if(total_loss < min_loss):
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