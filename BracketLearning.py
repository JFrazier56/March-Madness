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

ITERATIONS = 1

all_teams_average_stats = pd.read_csv('Data/Regular Season Data/averaged_stats.csv')
all_teams_average_stats_data = all_teams_average_stats.values

team_id_to_team_name = pd.read_csv('Data/Regular Season Data/Teams.csv')
team_id_to_team_name_data = team_id_to_team_name.values

team_id_to_seed = pd.read_csv('Data/Tournament Data/2017Seeds.csv')
team_id_to_seed_data = team_id_to_seed.values


# Preprocesses the given data and returns training and testing sets on the data
def processData():
    print "Pre-Processing Data..."
    detailed_results = pd.read_csv('Data/Multiple Seasons Data/ALL_CompleteTourneyDetailed.csv')
    data = detailed_results.values
    np.random.shuffle(data)
    row = int(math.floor(0.8 * data.shape[0]))
    training = data[0:row - 1, :]
    test = data[row:data.shape[0] - 1, :]

    return training, test

def processSeedData():
    detailed_results = pd.read_csv('Data/Tournament Data/odds_seeds.csv')
    data = detailed_results.values
    np.random.shuffle(data)
    training = data[:, 1:]
    trainingy = data[:, 0]

    bracket_results = pd.read_csv("Data/Bracket Games/start_bracket_games.csv")
    bracket_data = bracket_results.values

    return training, trainingy, bracket_data

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
def runLogisticRegression(X_training, y_training):

    trainingWeights = createWeights(X_training)

    resultWeights = logisticRegressionSGD(X_training, y_training, trainingWeights, STEP_SIZE, LAMBDA, 15)

    return resultWeights

# Run K Nearest Neighbors on the dataset for various numbers of neighbors to fit to
def runKNearestNeighbors(X_training, y_training):
    clf = KNeighborsClassifier(330, weights="uniform")

    clf.fit(X_training, y_training)

    return clf

# Run Random Forest on the dataset for various numbers of estimators
def runRandomForest(X_training, y_training):
    clf = RandomForestClassifier(max_depth=5, n_estimators=22, max_features='log2')

    clf.fit(X_training, y_training)

    return clf

# Run a Neural Network on the dataset for various numbers of lambda values
def runNeuralNetwork(X_training, y_training):
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

    lr_outputFile = open("Data/Bracket Predictions/LogisticRegr.txt", 'w')
    kn_outputFile = open("Data/Bracket Predictions/KNN.txt", 'w')
    rf_outputFile = open("Data/Bracket Predictions/RandomForest.txt", 'w')
    nn_outputFile = open("Data/Bracket Predictions/NeuralNetwork.txt", 'w')
    all_outputFile = open("Data/Bracket Predictions/AllResults.txt", 'w')

    num_games = 32
    round = 1

    LR_prob = [0] * num_games
    KN_prob = [0] * num_games
    RF_prob = [0] * num_games
    NN_prob = [0] * num_games
    total_prob = [0] * num_games

    winning_teams_lr = []
    winning_teams_kn = []
    winning_teams_rf = []
    winning_teams_nn = []
    winning_teams_all = []

    LR_res = []
    KN_res = []
    RF_res = []
    NN_res = []
    total_res = []

    # Run Linear Regression for seeing data
    X_seeding_training, y_seeding_training, bracketDataset = processSeedData()

    seedingWeights = runLinearRegression(X_seeding_training, y_seeding_training)

    lr_bracket_dataset = generateBracketDataset(bracketDataset, seedingWeights)
    kn_bracket_dataset = generateBracketDataset(bracketDataset, seedingWeights)
    rf_bracket_dataset = generateBracketDataset(bracketDataset, seedingWeights)
    nn_bracket_dataset = generateBracketDataset(bracketDataset, seedingWeights)
    all_bracket_dataset = generateBracketDataset(bracketDataset, seedingWeights)

    while(len(LR_res) != 1):
        for i in range(0, ITERATIONS):
            print "Starting iteration %d..." % i
            # Pre-Process the data
            trainingDataset, testingDataset = processData()

            # Feature Selection
            X_training, y_training, X_testing, y_testing, feature_selector = featureSelect(trainingDataset, testingDataset)

            # Logistic Regression
            resultWeights = runLogisticRegression(X_training, y_training)

            # KNN
            KNN_clf = runKNearestNeighbors(X_training, y_training)

            # Random Forest
            RF_clf = runRandomForest(X_training, y_training)

            # Neural Network
            NN_mlp = runNeuralNetwork(X_training, y_training)

            LR_res = logisitcRegressionBrackerPrediction(resultWeights, lr_bracket_dataset, feature_selector)
            KN_res = sklearnBracketPredictions(KNN_clf, kn_bracket_dataset, feature_selector, "K Nearest Neighbors")
            RF_res = sklearnBracketPredictions(RF_clf, rf_bracket_dataset, feature_selector, "Random Forest")
            NN_res = sklearnBracketPredictions(NN_mlp, nn_bracket_dataset, feature_selector, "Neural Network")

            LR_res_tot = logisitcRegressionBrackerPrediction(resultWeights, all_bracket_dataset, feature_selector)
            KN_res_tot = sklearnBracketPredictions(KNN_clf, all_bracket_dataset, feature_selector, "K Nearest Neighbors")
            RF_res_tot = sklearnBracketPredictions(RF_clf, all_bracket_dataset, feature_selector, "Random Forest")
            NN_res_tot = sklearnBracketPredictions(NN_mlp, all_bracket_dataset, feature_selector, "Neural Network")

            for i in range(0, len(LR_res)):
                LR_prob[i] += LR_res[i]
                KN_prob[i] += KN_res[i]
                RF_prob[i] += RF_res[i]
                NN_prob[i] += NN_res[i]
                total_prob[i] += (KN_res_tot[i] + RF_res_tot[i] + NN_res_tot[i] + LR_res_tot[i])

        for i in range(0, len(LR_res)):
            LR_prob[i] /= ITERATIONS
            KN_prob[i] /= ITERATIONS
            RF_prob[i] /= ITERATIONS
            NN_prob[i] /= ITERATIONS
            total_prob[i] /= ITERATIONS * 4

        teams_lr = lr_bracket_dataset[:, :2]
        teams_kn = kn_bracket_dataset[:, :2]
        teams_rf = rf_bracket_dataset[:, :2]
        teams_nn = nn_bracket_dataset[:, :2]
        teams_all = all_bracket_dataset[:, :2]

        for i in range(0, len(LR_res)):
            result = LR_prob[i]
            if result >= 0.5:
                winner = teams_lr[i, 0]
            else:
                winner = teams_lr[i, 1]

            winning_teams_lr += [winner]

        for i in range(0, len(KN_res)):
            result = KN_prob[i]
            if result >= 0.5:
                winner = teams_kn[i, 0]
            else:
                winner = teams_kn[i, 1]

            winning_teams_kn += [winner]

        for i in range(0, len(RF_res)):
            result = RF_prob[i]
            if result >= 0.5:
                winner = teams_rf[i, 0]
            else:
                winner = teams_rf[i, 1]

            winning_teams_rf += [winner]

        for i in range(0, len(NN_res)):
            result = NN_prob[i]
            if result >= 0.5:
                winner = teams_nn[i, 0]
            else:
                winner = teams_nn[i, 1]

            winning_teams_nn += [winner]

        for i in range(0, len(LR_res_tot)):
            result = total_prob[i]
            if result >= 0.5:
                winner = teams_all[i, 0]
            else:
                winner = teams_all[i, 1]

            winning_teams_all += [winner]

        printWinnersToFile(winning_teams_lr, lr_outputFile, round)
        printWinnersToFile(winning_teams_kn, kn_outputFile, round)
        printWinnersToFile(winning_teams_rf, rf_outputFile, round)
        printWinnersToFile(winning_teams_nn, nn_outputFile, round)
        printWinnersToFile(winning_teams_all, all_outputFile, round)

        if len(winning_teams_lr) != 1:
            lr_bracket_dataset = buildBracketDataset(winning_teams_lr, len(winning_teams_lr) / 2)
            lr_bracket_dataset = generateBracketDataset(lr_bracket_dataset, seedingWeights)

            kn_bracket_dataset = buildBracketDataset(winning_teams_kn, len(winning_teams_kn) / 2)
            kn_bracket_dataset = generateBracketDataset(kn_bracket_dataset, seedingWeights)

            rf_bracket_dataset = buildBracketDataset(winning_teams_rf, len(winning_teams_rf) / 2)
            rf_bracket_dataset = generateBracketDataset(rf_bracket_dataset, seedingWeights)

            nn_bracket_dataset = buildBracketDataset(winning_teams_nn, len(winning_teams_nn) / 2)
            nn_bracket_dataset = generateBracketDataset(nn_bracket_dataset, seedingWeights)

            all_bracket_dataset = buildBracketDataset(winning_teams_all, len(winning_teams_all) / 2)
            all_bracket_dataset = generateBracketDataset(all_bracket_dataset, seedingWeights)

        num_games /= 2

        LR_prob = [0] * num_games
        KN_prob = [0] * num_games
        RF_prob = [0] * num_games
        NN_prob = [0] * num_games
        total_prob = [0] * num_games

        winning_teams_lr = []
        winning_teams_kn = []
        winning_teams_rf = []
        winning_teams_nn = []
        winning_teams_all = []

        round += 1


    lr_outputFile.close()
    kn_outputFile.close()
    rf_outputFile.close()
    nn_outputFile.close()
    all_outputFile.close()

def printWinnersToFile(winning_teams, outputFile, round):
    outputString = "Round " + str(round) + ": " + convertTeamIdToTeamName(winning_teams[0])[0]

    for i in range(1, len(winning_teams)):
        outputString += ", " + convertTeamIdToTeamName(winning_teams[i])[0]

    outputString += " \n"

    outputFile.write(outputString)

def sklearnBracketPredictions(clf, bracket_dataset, feature_selector, classif_name):

    teams_data = bracket_dataset[:, 2:]

    teams_data = feature_selector.transform(teams_data)

    game_results = clf.predict(teams_data)

    return game_results


def logisitcRegressionBrackerPrediction(weights, bracket_dataset, feature_selector):

    teams_data = bracket_dataset[:, 2:]

    teams_data = feature_selector.transform(teams_data)

    game_results = logisticRegressionPredict(teams_data, weights)

    return game_results


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
    return team_name.flatten()

if __name__ == "__main__":
    main()