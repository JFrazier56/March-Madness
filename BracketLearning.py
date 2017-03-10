import numpy as np
import pandas as pd
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RandomizedLogisticRegression
from sklearn.neural_network import MLPClassifier

SIZE_TRAINING = 0.8
STEP_SIZE = 0.00001
LAMBDA = 16
num_neightbors = 285

SEEDING_LAMBDA = 1 * (10 ** -5)


ITERATIONS = 25

all_teams_average_stats = pd.read_csv('Data/averaged_stats.csv')
all_teams_average_stats_data = all_teams_average_stats.values

team_id_to_team_name = pd.read_csv('Data/Teams.csv')
team_id_to_team_name_data = team_id_to_team_name.values

team_id_to_seed = pd.read_csv('Data/2017Seeds.csv')
team_id_to_seed_data = team_id_to_seed.values


# Preprocesses the given data and returns training and testing sets on the data
def processData():
    print "Pre-Processing Data..."
    detailed_results = pd.read_csv('Data/ALL_CompleteTourneyDetailed.csv')
    data = detailed_results.values
    np.random.shuffle(data)
    row = int(math.floor(0.8 * data.shape[0]))
    training = data[0:row - 1, :]
    test = data[row:data.shape[0] - 1, :]

    bracket_results = pd.read_csv("Data/bracket_games.csv")
    bracket_data = bracket_results.values

    return training, test, bracket_data

def processSeedData():
    detailed_results = pd.read_csv('Data/odds_seeds.csv')
    data = detailed_results.values
    np.random.shuffle(data)
    training = data[:, 1:]
    trainingy = data[:, 0]

    return training, trainingy

def runLinearRegression(x, y):
    Ht_H = np.dot(np.transpose(x), x)
    lambda_i = np.multiply(SEEDING_LAMBDA * np.identity(x.shape[1]), x.shape[1])
    Ht_y = np.dot(np.transpose(x), y)
    w = np.dot(np.linalg.inv(np.add(Ht_H, lambda_i)), Ht_y)
    return w


def predictVegasOdds(x, w):
    weightTranspose = np.transpose(w)
    dotProduct = np.dot(weightTranspose, x)
    return dotProduct


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
def logisticRegressionPredict(dataset, weights):
    predictedValues = []
    for i in range(0, dataset.shape[0]):
        weightTranspose = np.transpose(weights)
        dotProduct = np.dot(weightTranspose, dataset[i])
        lastTerm = sigmoid(dotProduct)
        if (lastTerm >= 0.5):
            yValue = 1.0
        else:
            yValue = 0.0

        predictedValues += [yValue]

    return predictedValues

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

# Run Logistic regression with various iterations over the given datasets
# TODO: Pick best num iterations
def runLogisticRegression(X_training, y_training):

    trainingWeights = createWeights(X_training)

    resultWeights = logisticRegressionSGD(X_training, y_training, trainingWeights, STEP_SIZE, LAMBDA, 15)

    return resultWeights

# Run K Nearest Neighbors on the dataset for various numbers of neighbors to fit to
# TODO: Pick best num_neighbors
def runKNearestNeighbors(X_training, y_training):
    print "Starting K Nearest Neighbors..."

    clf = KNeighborsClassifier(330, weights="uniform")

    clf.fit(X_training, y_training)

    return clf

# Run Random Forest on the dataset for various numbers of estimators
# TODO: Pick best num_estimators
def runRandomForest(X_training, y_training):
    print "Starting Random Forest..."

    clf = RandomForestClassifier(max_depth=5, n_estimators=22, max_features='log2')

    clf.fit(X_training, y_training)

    return clf

# Run a Neural Network on the dataset for various numbers of lambda values
# TODO: Pick best lambda value
def runNeuralNetwork(X_training, y_training):
    print "Starting Neural Network..."

    mlp = MLPClassifier(activation='logistic', solver='sgd', alpha=0.1)

    mlp.fit(X_training, y_training)

    return mlp

# Feature selection on the given datasets
def featureSelect(trainingDataset, testingDataset):
    print "Starting Feature Selection..."
    X_training = trainingDataset[:, 4:]
    y_training = trainingDataset[:, 3]

    X_testing = testingDataset[:, 4:]
    y_testing = testingDataset[:, 3]

    randomlr = RandomizedLogisticRegression()
    X_training = randomlr.fit_transform(X_training, y_training)
    X_testing = randomlr.transform(X_testing)

    return X_training, y_training, X_testing, y_testing, randomlr

def main():

    # for i in range(0, ITERATIONS):
    #     print "Starting iteration %d..." % i
        # Pre-Process the data
    trainingDataset, testingDataset, bracketDataset = processData()

    X_seeding_training, y_seeding_training = processSeedData()

    # Feature Selection
    X_training, y_training, X_testing, y_testing, feature_selector = featureSelect(trainingDataset, testingDataset)

    # Run Linear Regression for seeing data
    seedingWeights = runLinearRegression(X_seeding_training, y_seeding_training)

    # Logistic Regression
    resultWeights = runLogisticRegression(X_training, y_training)

    # KNN
    KNN_clf = runKNearestNeighbors(X_training, y_training)

    # Random Forest
    RF_clf = runRandomForest(X_training, y_training)

    # Neural Network
    NN_mlp = runNeuralNetwork(X_training, y_training)

    print "Training classifiers complete"


    logisitcRegressionBrackerPrediction(resultWeights, bracketDataset, feature_selector, seedingWeights)
    sklearnBracketPredictions(KNN_clf, bracketDataset, feature_selector, seedingWeights, "K Nearest Neighbors")
    sklearnBracketPredictions(RF_clf, bracketDataset, feature_selector, seedingWeights, "Random Forest")
    sklearnBracketPredictions(NN_mlp, bracketDataset, feature_selector, seedingWeights, "Neural Network")


def sklearnBracketPredictions(clf, bracket_dataset, feature_selector, seeding_weights, classif_name):
    outputFile = open(classif_name + ".txt", 'w')
    print "Starting predictions with classifier %s" % classif_name
    full_dataset = generateBracketDataset(bracket_dataset, seeding_weights)
    winning_teams = []
    round = 1
    while len(winning_teams) != 1:
        winning_teams = []

        teams = full_dataset[:, :2]
        teams_data = full_dataset[:, 2:]

        teams_data = feature_selector.transform(teams_data)

        game_results = clf.predict(teams_data)

        for i in range(0, len(game_results)):
            result = game_results[i]
            if result == 1:
                winner = teams[i, 0]
            else:
                winner = teams[i, 1]

            winning_teams += [winner]

        outputString = "Round " + str(round) + ": " + convertTeamIdToTeamName(winning_teams[0])

        for i in range(1, len(winning_teams)):
            outputString += ", " + convertTeamIdToTeamName(winning_teams[i])

        outputFile.write(outputString)

        if len(winning_teams) != 1:
            print winning_teams
            full_dataset = buildBracketDataset(winning_teams, len(winning_teams) / 2)
            full_dataset = generateBracketDataset(full_dataset, seeding_weights)

        round += 1
    outputFile.close()
    print "Classifier %s predicted that the overall winner would be %d" % (classif_name, winning_teams[0])


def logisitcRegressionBrackerPrediction(weights, bracket_dataset, feature_selector, seeding_weights):
    print "Starting predictions with logistic regression"
    outputFile = open("LogisticRegression.txt", 'w')
    full_dataset = generateBracketDataset(bracket_dataset, seeding_weights)
    winning_teams = []
    while len(winning_teams) != 1:
        winning_teams = []

        teams = full_dataset[:, :2]
        teams_data = full_dataset[:, 2:]
        teams_data = feature_selector.transform(teams_data)

        game_results = logisticRegressionPredict(teams_data, weights)

        for i in range(0, len(game_results)):
            result = game_results[i]
            if result == 1:
                winner = teams[i, 0]
            else:
                winner = teams[i, 1]

            winning_teams += [winner]

        outputString = "Round " + str(round) + ": " + convertTeamIdToTeamName(winning_teams[0])

        for i in range(1, len(winning_teams)):
            outputString += ", " + convertTeamIdToTeamName(winning_teams[i])

        outputFile.write(outputString)

        if len(winning_teams) != 1:
            print winning_teams
            full_dataset = buildBracketDataset(winning_teams, len(winning_teams) / 2)
            full_dataset = generateBracketDataset(full_dataset, seeding_weights)

    print "Logistic Regression predicted that the overall winner would be %d" % winning_teams[0]
    outputFile.close()


def buildBracketDataset(winning_teams, num_matches):
    dataset = np.zeros(shape=(num_matches, 2))
    for i in range(0, num_matches):
        team1 = winning_teams[2 * i]
        team2 = winning_teams[(2 * i) + 1]
        dataset[i] = np.array([[team1, team2]])

    return dataset


def generateBracketDataset(bracket_dataset, seeding_weights):
    num_rows = bracket_dataset.shape[0]
    full_dataset = np.zeros(shape=[num_rows, 37])
    for i in range(0, bracket_dataset.shape[0]):
        team1_id = bracket_dataset[i, 0]
        team2_id = bracket_dataset[i, 1]

        teams_array = all_teams_average_stats['team']
        seeding_array = team_id_to_seed['team']

        team1_index = np.where(teams_array == team1_id)
        team2_index = np.where(teams_array == team2_id)

        team1_stats = all_teams_average_stats_data[team1_index, :]
        team2_stats = all_teams_average_stats_data[team2_index, :]
        team1_stats = team1_stats.flatten()
        team2_stats = team2_stats.flatten()

        team1_stats = np.delete(team1_stats, 0)
        team2_stats = np.delete(team2_stats, 0)

        team1_seed_index = np.where(seeding_array == team1_id)
        team2_seed_index = np.where(seeding_array == team2_id)
        team1_seed = team_id_to_seed_data[team1_seed_index, 1]
        team2_seed = team_id_to_seed_data[team2_seed_index, 1]

        team1_seed = team1_seed.flatten()
        team2_seed = team2_seed.flatten()

        seeds_array = np.array([team1_seed, team2_seed])

        vegasOdd = predictVegasOdds(seeds_array, seeding_weights)

        new_row = np.hstack((team1_id, team2_id, vegasOdd, 0, 0, team1_stats, team1_seed, team2_stats, team2_seed))
        full_dataset[i] = new_row

    return full_dataset

def createWeights(dataset):
    weights = []

    for i in range(0, dataset.shape[1]):
        weights += [0]

    return weights

def convertTeamIdToTeamName(team_id):
    team_id_array = team_id_to_team_name['Team_Id']
    team_index = np.where(team_id_array == team_id)
    team_name = team_id_to_team_name_data[team_index, 1]
    return team_name

if __name__ == "__main__":
    main()