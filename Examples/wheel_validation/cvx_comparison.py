import os.path
import collections
import time
import pickle
import numpy
import cvxpy
import xgboost
from scipy import spatial
from sklearn import preprocessing, metrics, model_selection, linear_model

import data_loader
import cvx_wheels


def network_lasso(lamb = 1, number_of_neighbours = 5):
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
            costs_gi += lamb / distance * cvxpy.norm(cvx_models[suburbs_to_model[i]].model - cvx_models[suburbs_to_model[j]].model, 2)

    # solve the problem
    problem = cvxpy.Problem(cvxpy.Minimize(costs_fi + costs_gi))
    problem.solve()

    print("total cost at nodes =", costs_fi.value, "total cost on edges =", costs_gi.value)
    return cvx_models


# configs
number_of_transactions_needed = 60

# load data
if os.path.isfile("data"):
    with open("data", "rb") as f:
        data = pickle.load(f)
else:
    data = data_loader.load_data(2729, 2789, verbose = True)
    with open("data", "wb") as f:
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

X = numpy.array([item[:-3] for item in data if item[10] in suburbs_to_model and item[11] == suburbs[item[10]][0]])
Y = numpy.array([item[-1] for item in data if item[10] in suburbs_to_model and item[11] == suburbs[item[10]][0]])


# global model setting
imputer = preprocessing.Imputer(missing_values="NaN", strategy="mean")
scaler = preprocessing.MinMaxScaler()
Y_scaler = preprocessing.MinMaxScaler()

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = 0.2, random_state = 0)
X_train = imputer.fit_transform(X_train)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(imputer.transform(X_test))
Y_train = Y_scaler.fit_transform(numpy.reshape(Y_train, (-1, 1))).flatten()


# global level linear regression
print("global linear regression")
start_time = time.time()

sklearn_model = linear_model.LinearRegression(n_jobs = -1)
sklearn_model.fit(X_train, Y_train)
print("sklearn finished at", round(time.time() - start_time), "seconds" )
print(sklearn_model.coef_, sklearn_model.intercept_)
raw_predictions = sklearn_model.predict(X_test)
print("global linear regression RMSE:", numpy.sqrt(metrics.mean_squared_error(Y_scaler.inverse_transform(numpy.reshape(raw_predictions, (-1, 1))).flatten(), Y_test)))


# global xgboost
num_boost_round = 50
print("\nglobal xgboost with", num_boost_round, "rounds")
start_time = time.time()

parameters = {"eta": 0.1, "seed": 0, "subsample": 0.8, "colsample_bytree": 0.8, "objective": "reg:linear", "max_depth": 15, "min_child_weight": 1, "silent": True}
train_matrix = xgboost.DMatrix(X_train, Y_train)
xgboost_model = xgboost.train(parameters, train_matrix, num_boost_round = num_boost_round)
raw_predictions = xgboost_model.predict(xgboost.DMatrix(X_test))
print("xgboost RMSE:", numpy.sqrt(metrics.mean_squared_error(Y_scaler.inverse_transform(numpy.reshape(raw_predictions, (-1, 1))).flatten(), Y_test)))


# per suburb model setting
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


# per suburb linear regression
print("\nper suburb linear regression")
start_time = time.time()

sklearn_model = collections.defaultdict(lambda: linear_model.LinearRegression(n_jobs = -1))
for suburb in suburbs_to_model:
    sklearn_model[suburb].fit(X_train[suburb], Y_train[suburb])
print("sklearn finished at", round(time.time() - start_time), "seconds" )
print(sklearn_model["Parramatta"].coef_, sklearn_model["Parramatta"].intercept_)
predictions, target = [], []
for suburb in suburbs_to_model:
    raw_predictions = sklearn_model[suburb].predict(X_test[suburb])
    predictions = numpy.concatenate((predictions, (Y_scalers[suburb].inverse_transform(numpy.reshape(raw_predictions, (-1, 1))).flatten())))
    target = numpy.concatenate((target, Y_test[suburb]))
print("per suburb linear regression RMSE:", numpy.sqrt(metrics.mean_squared_error(predictions, target)))


# network lasso
lamb = 0.0001
number_of_neighbours = 3
print("\nnetwork lasso with lambda =", lamb, "number of neighbours =", number_of_neighbours)
start_time = time.time()

lasso_model = network_lasso(lamb, number_of_neighbours)
print("global problem solved at", round(time.time() - start_time), "seconds" )
print(lasso_model["Parramatta"].model.value)
predictions, target = [], []
for suburb in suburbs_to_model:
    raw_predictions = lasso_model[suburb].predict(X_test[suburb])
    predictions = numpy.concatenate((predictions, (Y_scalers[suburb].inverse_transform(numpy.reshape(raw_predictions, (-1, 1))).flatten())))
    target = numpy.concatenate((target, Y_test[suburb]))
print("network lasso RMSE:", numpy.sqrt(metrics.mean_squared_error(predictions, target)))