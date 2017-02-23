import numpy as np
import pandas as pd
import random
import math

SIZE_TRAINING = 0.8

STEP_SIZE = 0.00001

LAMBDA = 1

def processData():
    detailed_results = pd.read_csv('RegularizedSeasonDetailed.csv')
    data = detailed_results.values
    np.random.shuffle(data)
    row = int(math.floor(0.8 * data.shape[0]))
    training = data[0:row - 1, :]
    test = data[row:data.shape[0] - 1, :]

    return training, test

def logisticRegressionSGD(dataset, weights, stepSize, lambdaValue):

    order = []

    for i in range(1, dataset.shape[0]):
        order.append(i)

    random.shuffle(order)

    for k in range(0, 2):
        for i in order:
            if i == 0 or i == 1 or i == 2:
                continue
            for j in range(1, dataset.shape[1]):
                weightTranspose = np.transpose(weights)
                dotProduct = np.dot(weightTranspose, dataset[i])
                lastTerm = sigmoid(dotProduct)
                partial = dataset[i, j] * (dataset[i, 2] - lastTerm)
                weights[j] -= stepSize * (partial - (2 * lambdaValue * weightTranspose[j]))


    return weights

def sigmoid(x):
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        return 1



def main():
    training, test = processData()

    trainingWeights = createWeights(training)

    resultWeights = logisticRegressionSGD(training, trainingWeights, STEP_SIZE, LAMBDA)

    largestCoef = -100000
    smallestCoef = 1000000
    largestIndex = 0
    smallestIndex = 0

    index = 0

    print(resultWeights)

    for i in resultWeights:
        if i > largestCoef:
            largestCoef = i
            largestIndex = index
        if i < smallestCoef:
            smallestCoef = i
            smallestIndex = index
        index += 1

    print(largestCoef)
    print(largestIndex)
    print(smallestCoef)
    print(smallestIndex)


def createWeights(dataset):
    weights = []

    for i in range(0, dataset.shape[1]):
        weights += [0]

    return weights

if __name__ == "__main__":
    main()
