import os.path
import collections
import time
import pickle
import numpy
import cvxpy
from scipy import spatial
from sklearn import preprocessing, metrics, model_selection, linear_model

import data_loader
import cvx_wheels


def concatenate(model):
    return cvxpy.atoms.affine.hstack.hstack((model.coefficients, model.intercept))

def grid_search(lamb = 1, number_of_neighbours = 5):
    print("starting to construct the network lasso model with a lambda of", lamb)
    start_time = time.time()

    # fit to get the preliminary local models
    cvx_models = collections.defaultdict(cvx_wheels.linear_regression)
    for suburb in suburbs_to_model:
        cvx_models[suburb].fit(X_train[suburb], Y_train[suburb])

    costs_fi, costs_gi = 0, 0

    # individual model costs
    for suburb in suburbs_to_model:
        for i in range(cvx_models[suburb].dimensionality):
            costs_fi += (X_train[suburb][i] * cvx_models[suburb].coefficients + cvx_models[suburb].intercept - Y_train[suburb][i]) ** 2

    # lasso terms
    for i in range(n):
        nearest_suburbs = suburb_tree.query([coordinates[i]], number_of_neighbours + 1)[1][0][1:]
        for j in nearest_suburbs:
            distance = numpy.sqrt((coordinates[i][0] - coordinates[j][0]) ** 2 + (coordinates[i][1] - coordinates[j][1]) ** 2)
            costs_gi += lamb / distance * cvxpy.norm(concatenate(cvx_models[suburbs_to_model[i]]) - concatenate(cvx_models[suburbs_to_model[j]]), 2)

    # solve the problem
    problem = cvxpy.Problem(cvxpy.Minimize(costs_fi + costs_gi))
    problem.solve()
    print(costs_fi.value, costs_gi.value)
    print("problem solved at", round(time.time() - start_time), "seconds" )

    predictions, target = [], []
    for suburb in suburbs_to_model:
        raw_predictions = cvx_models[suburb].predict(X_test[suburb])
        predictions = numpy.concatenate((predictions, Y_scalers[suburb].inverse_transform(numpy.reshape(raw_predictions, (-1, 1))).flatten()))
        target = numpy.concatenate((target, Y_test[suburb]))
    print("RMSE:", numpy.sqrt(metrics.mean_squared_error(predictions, target)))



# configs
number_of_transactions_needed = 60

# load data
if os.path.isfile("selected_data"):
    with open("selected_data", "rb") as f:
        data = pickle.load(f)
else:
    data = data_loader.load_data(2729, 2789, verbose = True)
    with open("selected_data", "wb") as f:
        pickle.dump(data, f)

# find suburbs to model
with open("suburbs_geolocations", "rb") as f:
    suburbs = pickle.load(f)

transaction_count = collections.defaultdict(int)
for item in data:
    if item[10] in suburbs and item[11] == suburbs[item[10]][0]:
        transaction_count[item[10]] += 1

suburbs_to_model = []
for item in transaction_count:
    if transaction_count[item] > number_of_transactions_needed:
        suburbs_to_model.append(item)
n = len(suburbs_to_model)


# data preprocessing
X = numpy.array([item[:-3] for item in data if item[10] in suburbs_to_model and item[11] == suburbs[item[10]][0]])
Y = numpy.array([item[-1] for item in data if item[10] in suburbs_to_model and item[11] == suburbs[item[10]][0]])
imputer = preprocessing.Imputer(missing_values="NaN", strategy="mean")
scaler = preprocessing.MinMaxScaler()

# per suburb linear regression
X_seg, Y_seg = collections.defaultdict(list), collections.defaultdict(list)
Y_scalers = collections.defaultdict(preprocessing.MinMaxScaler)
X_train, X_test, Y_train, Y_test = dict(), dict(), dict(), dict()

for item in data:
    if item[10] in suburbs_to_model and item[11] == suburbs[item[10]][0]:
        X_seg[item[10]].append(item[:-3])
        Y_seg[item[10]].append(item[-1])

for suburb in X_seg:
    # train test split
    X_train[suburb], X_test[suburb], Y_train[suburb], Y_test[suburb] = model_selection.train_test_split(X_seg[suburb], Y_seg[suburb], test_size = 0.2, random_state = 0)

    # impute missing values
    X_train[suburb] = imputer.fit_transform(X_train[suburb])
    X_test[suburb] = imputer.transform(X_test[suburb])

    # scale
    X_train[suburb] = scaler.fit_transform(X_train[suburb])
    X_test[suburb] = scaler.transform(X_test[suburb])
    Y_train[suburb] = Y_scalers[suburb].fit_transform(numpy.reshape(Y_train[suburb], (-1, 1))).flatten()

# build kdtree to lookup nearest neighbours of suburbs
coordinates = [suburbs[suburb][1:] for suburb in suburbs_to_model]
suburb_tree = spatial.KDTree(coordinates)


# network lasso
grid_search(lamb = 0.000001)