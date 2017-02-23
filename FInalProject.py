import numpy as np
import pandas as pd
import random
import math

SIZE_TRAINING = 0.8

def processData():
    detailed_results = pd.read_csv('RegularSeasonDetailedResults.csv')
    data = detailed_results.values
    np.random.shuffle(data)
    row = int(math.floor(0.8 * data.shape[0]))
    training = data[0:row - 1, :]
    test = data[row:data.shape[0] - 1, :]

    return training, test

def logisticRegressionSGD(dataset, weights, stepSize):

    order = []
    weights = []

    for i in range(0, dataset.shape[0]):
        weights += [0]
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
                partial = dataset[i, j] * (dataset[i, 0] - lastTerm)
                weights[j] -= partial * stepSize


    return weights

def sigmoid(x):
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        return 1



def main():
    training, test = processData()

    print(training)
    print(test)

if __name__ == "__main__":
    main()
