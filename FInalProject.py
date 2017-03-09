import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as mp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RandomizedLogisticRegression
from sklearn.neural_network import MLPClassifier

SIZE_TRAINING = 0.8
LAMBDA_FOUND = True
STEP_SIZE = 0.00001
LAMBDA = 16
num_neightbors = 285

# Preprocesses the given data and returns training and testing sets on the data
def processData():
    print "Pre-Processing Data..."
    detailed_results = pd.read_csv('RegularizedSeasonDetailed.csv')
    data = detailed_results.values
    np.random.shuffle(data)
    row = int(math.floor(0.8 * data.shape[0]))
    training = data[0:row - 1, :]
    test = data[row:data.shape[0] - 1, :]

    return training, test

# Run Logistic regression with L2 normalization on the dataset
def logisticRegressionSGD(dataset, results, weights, stepSize, lambdaValue, iterations):
    datasetSize = dataset.shape[1]
    shuffle_in_unison(dataset, results)
    for k in range(0, iterations):
        for i in range(0, dataset.shape[0]):
            for j in range(0, dataset.shape[1]):
                weightTranspose = np.transpose(weights)
                dotProduct = np.dot(weightTranspose, dataset[i])
                lastTerm = sigmoid(dotProduct)
                partial = dataset[i, j] * (results[i] - lastTerm)
                weights[j] -= stepSize * (partial - ((2 / datasetSize) * lambdaValue * weightTranspose[j]))

    return weights

# Compute the perecentage of corrent guesses we get with the newly learned weights from SGD
def testingNewWeight(dataset, results, weights):
    LogLoss = 0
    for i in range(0, dataset.shape[0]):
        weightTranspose = np.transpose(weights)
        dotProduct = np.dot(weightTranspose, dataset[i])
        lastTerm = sigmoid(dotProduct)
        if (lastTerm >= 0.5):
            yValue = 1.0
        else:
            yValue = 0.0

        if results[i] == yValue:
            LogLoss += 1.0

    return LogLoss / dataset.shape[0]

# Function to compute the sigmoid
def sigmoid(x):
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        return 1

# Shuffles both A and B in the same way, done by resetting the random state before shuffling again
def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

# Function for graphing iterations vs all_testloss
def graphLoss(iterations, all_testloss, title):
    mp.figure()
    mp.plot(iterations, all_testloss, label=title)
    mp.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
              ncol=2, mode="expand", borderaxespad=0.)
    mp.axhline(0, color='black')
    mp.axvline(0, color='black')
    mp.show()

# Run Logistic regression with various iterations over the given datasets
def runLogisticRegression(X_training, y_training, X_testing, y_testing):
    iterations = [1, 3, 5, 10, 15]
    all_testloss = []
    for i in range(0, len(iterations)):
        print "Starting logistic regression on data with %d iterations..." % iterations[i]
        total_loss = 0
        for j in range(0, 3):
            trainingWeights = createWeights(X_training)
            resultWeights = logisticRegressionSGD(X_training, y_training, trainingWeights, STEP_SIZE, LAMBDA,
                                                  iterations[i])
            test_loss = testingNewWeight(X_testing, y_testing, resultWeights)
            total_loss += test_loss
        all_testloss += [total_loss / 3]

    return iterations, all_testloss

# Run K Nearest Neighbors on the dataset for various numbers of neighbors to fit to
def runKNearestNeighbors(X_training, y_training, X_testing, y_testing):
    print "Starting K Nearest Neighbors..."
    neighbors = [i * 10 for i in range(1, 60)]
    KNN_all_testloss = []
    for neighbor in neighbors:
        clf = KNeighborsClassifier(neighbor, weights="uniform")

        clf.fit(X_training, y_training)

        KNN_all_testloss += [clf.score(X_testing, y_testing)]
    return neighbors, KNN_all_testloss

# Run Random Forest on the dataset for various numbers of estimators
def runRandomForest(X_training, y_training, X_testing, y_testing):
    print "Starting Random Forest..."
    num_estimators = [i for i in range(1, 40)]
    RF_all_testloss = []
    for estimatos in num_estimators:
        clf = RandomForestClassifier(max_depth=5, n_estimators=estimatos, max_features='log2')

        clf.fit(X_training, y_training)

        RF_all_testloss += [clf.score(X_testing, y_testing)]
    return num_estimators, RF_all_testloss

# Run a Neural Network on the dataset for various numbers of lambda values
def runNeuralNetwork(X_training, y_training, X_testing, y_testing):
    print "Starting Neural Network..."
    lambda_values = [.000001, .0001, .1, 1, 2, 4, 8, 16, 32, 64, 128]
    NN_all_testloss = []
    for lambdaVal in lambda_values:
        print lambdaVal
        mlp = MLPClassifier(activation='logistic', solver='sgd', alpha=lambdaVal)

        mlp.fit(X_training, y_training)

        NN_all_testloss += [mlp.score(X_testing, y_testing)]
    return lambda_values, NN_all_testloss

# Feature selection on the given datasets
def featureSelect(trainingDataset, testingDataset):
    print "Starting Feature Selection..."
    X_training = trainingDataset[:, 3:]
    y_training = trainingDataset[:, 2]

    X_testing = testingDataset[:, 3:]
    y_testing = testingDataset[:, 2]

    randomlr = RandomizedLogisticRegression()
    X_training = randomlr.fit_transform(X_training, y_training)
    X_testing = randomlr.transform(X_testing)

    return X_training, y_training, X_testing, y_testing

def main():
    # Pre-Process the data
    training, test = processData()

    # Feature Selection
    X_training, y_training, X_testing, y_testing = featureSelect(training, test)

    # Logistic Regression
    (iterations, all_testloss) = runLogisticRegression(X_training, y_training, X_testing, y_testing)

    # KNN
    (neighbors, KNN_all_testloss) = runKNearestNeighbors(X_training, y_training, X_testing, y_testing)

    # Random Forest
    (num_estimators, RF_all_testloss) = runRandomForest(X_training, y_training, X_testing, y_testing)

    # Neural Network
    (lambda_values, NN_all_testloss) = runNeuralNetwork(X_training, y_training, X_testing, y_testing)

    # Plot iteration vs. Percent right
    print "Max percentage right with Logisitc Regression was %.10f" % max(all_testloss)
    title = 'Logisitic Regression Correctly Predicted versus Iterations on Dataset'
    graphLoss(iterations, all_testloss, title)

    # Graph the number of neighbors to the percentage of correct guesses
    print "Max percentage right with K Nearest Neighbors was %.10f" % max(KNN_all_testloss)
    title = 'K Nearest Neighbors Correctly Predicted verses Num_Neighbors'
    graphLoss(neighbors, KNN_all_testloss, title)

    print "Max percentage right with Random Forest was %.10f" % max(RF_all_testloss)
    title = 'Random Forest Correctly Predicted versus Num_estimators on Dataset'
    graphLoss(num_estimators, RF_all_testloss, title)

    print "Max percentage right with Neural Network was %.10f" % max(NN_all_testloss)
    title = 'Neural Network Correctly Predicted versus Lambda Value on Dataset'
    graphLoss(lambda_values, NN_all_testloss, title)
    print(NN_all_testloss)


def createWeights(dataset):
    weights = []

    for i in range(0, dataset.shape[1]):
        weights += [0]

    return weights

if __name__ == "__main__":
    main()



"""

        ** THIS WAS THE CODE THAT WAS USED TO DETERMINE THE BEST WEIGHT TYPE AND NUMBER OF NEIGHBORS***

        for ki in range(0, 10):
            training, test = processData()
            maxPercentRight = 0.0
            maxWeightType = ""
            maxNeighborNum = 0
            for weight in ["distance", "uniform"]:
                neighbors = [i * 10 for i in range(10, 40)]
                for neighbor in neighbors:
                    clf = KNeighborsClassifier(neighbor, weights=weight)
                    y_values = training[:, 2]
                    x_values = training[:, 3:]
                    clf.fit(x_values, y_values)

                    test_x_values = test[:, 3:]
                    z = clf.predict(test_x_values)

                    totalCount = 0
                    correctCount = 0

                    for i in range(0, len(z)):
                        predicted = z[i]
                        actual = test[i, 2]

                        if predicted == actual:
                            correctCount += 1

                        totalCount += 1

                    percentageRight = float(correctCount) / float(totalCount)

                    if percentageRight > maxPercentRight:
                        maxPercentRight = percentageRight
                        maxWeightType = weight
                        maxNeighborNum = neighbor


            print "Best performance in round %d was weight type %s with num_neighbors = %d with a percentage of %.10f" % (ki, maxWeightType, maxNeighborNum, maxPercentRight)


"""

"""

        *** THIS WAS THE CODE USED TO FIND THE BEST LAMBDA VALUE FOR LOGISTIC REGRESSION ***

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

"""