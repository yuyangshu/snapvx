# tests to ensure the behavior of the cvx library is as expected

import os.path
import collections
import time
import pickle
import numpy
import cvxpy
from sklearn import metrics, model_selection, linear_model

import preprocessing
import cvx_wheels

# configs
number_of_transactions_needed = 60

# load data
if os.path.isfile("selected_data"):
    with open("selected_data", "rb") as f:
        data = pickle.load(f)
else:
    data = preprocessing.load_data(2729, 2789, verbose = True)
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


# global linear regression
X = numpy.array([item[:-3] for item in data if item[10] in suburbs_to_model and item[11] == suburbs[item[10]][0]])
Y = numpy.array([item[-1] for item in data if item[10] in suburbs_to_model and item[11] == suburbs[item[10]][0]])

# replace missing values and scale
replacer = preprocessing.missing_value_replacer()
replacer.set_means(X)
X = replacer.replace_missing(numpy.array(X))

scaler = preprocessing.data_scaler()
scaler.set_scale(X)
X = scaler.scale(X)
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = 0.2, random_state = 0)

# compare sklearn and cvx for equivalence
print("global level linear regression")
start_time = time.time()

sklearn_model = linear_model.LinearRegression(n_jobs = -1)
sklearn_model.fit(X_train, Y_train)
print("sklearn finished at", round(time.time() - start_time), "seconds" )
print(sklearn_model.coef_, sklearn_model.intercept_)
print("sklearn RMSE:", numpy.sqrt(metrics.mean_squared_error(sklearn_model.predict(X_test), Y_test)))

cvx_model = cvx_wheels.linear_regression()
cvx_model.fit(X_train, Y_train)
print("cvx finished at", round(time.time() - start_time), "seconds" )
print(numpy.array([item.value for item in cvx_model.coefficients]), cvx_model.intercept.value[0])
print("cvx RMSE:", cvx_wheels.root_mean_squared_error(cvx_model.predict(X_test), Y_test))


# per suburb linear regression
X_seg = collections.defaultdict(list)
Y_seg = collections.defaultdict(list)
for item in data:
    if item[10] in suburbs and item[11] == suburbs[item[10]][0]:
        X_seg[item[10]].append(item[:-3])
        Y_seg[item[10]].append(item[-1])

for suburb in X_seg:
    X_seg[suburb] = scaler.scale(replacer.replace_missing(numpy.array(X_seg[suburb])))
    Y_seg[suburb] = numpy.array(Y_seg[suburb])

X_train, X_test, Y_train, Y_test = dict(), dict(), dict(), dict()
for suburb in suburbs_to_model:
    X_train[suburb], X_test[suburb], Y_train[suburb], Y_test[suburb] = model_selection.train_test_split(X_seg[suburb], Y_seg[suburb], test_size = 0.2, random_state = 0)

print("\nper suburb linear regression")
start_time = time.time()

sklearn_model = collections.defaultdict(lambda: linear_model.LinearRegression(n_jobs = -1))
for suburb in suburbs_to_model:
    sklearn_model[suburb].fit(X_train[suburb], Y_train[suburb])
print("sklearn finished at", round(time.time() - start_time), "seconds" )
print(sklearn_model["Parramatta"].coef_, sklearn_model["Parramatta"].intercept_)
predictions, target = [], []
for suburb in suburbs_to_model:
    predictions = numpy.concatenate((predictions, sklearn_model[suburb].predict(X_test[suburb])))
    target = numpy.concatenate((target, Y_test[suburb]))
print("sklearn RMSE:", numpy.sqrt(metrics.mean_squared_error(predictions, target)))

cvx_model = collections.defaultdict(cvx_wheels.linear_regression)
for suburb in suburbs_to_model:
    cvx_model[suburb].fit(X_train[suburb], Y_train[suburb])
print("cvx finished at", round(time.time() - start_time), "seconds" )
print(numpy.array([item.value for item in cvx_model["Parramatta"].coefficients]), cvx_model["Parramatta"].intercept.value[0])
predictions, target = [], []
for suburb in suburbs_to_model:
    predictions = numpy.concatenate((predictions, cvx_model[suburb].predict(X_test[suburb])))
    target = numpy.concatenate((target, Y_test[suburb]))
print("cvx RMSE:", numpy.sqrt(metrics.mean_squared_error(predictions, target)))


# preliminary to network lasso
print("\nper suburb model, optimize total cost directly")
start_time = time.time()

# fit to get the preliminary local models
# repetitive to the second case, but kept for the sake of completeness
cvx_models = collections.defaultdict(cvx_wheels.linear_regression)
for suburb in suburbs_to_model:
    cvx_models[suburb].fit(X_train[suburb], Y_train[suburb])

overall_costs = 0
for suburb in suburbs_to_model:
    for i in range(cvx_models[suburb].dimensionality):
        overall_costs += (X_train[suburb][i] * cvx_models[suburb].coefficients + cvx_models[suburb].intercept - Y_train[suburb][i]) ** 2
problem = cvxpy.Problem(cvxpy.Minimize(overall_costs))
problem.solve()
print("global problem solved at", round(time.time() - start_time), "seconds" )

print(numpy.array([item.value for item in cvx_models["Parramatta"].coefficients]), cvx_models["Parramatta"].intercept.value[0])
predictions, target = [], []
for suburb in suburbs_to_model:
    predictions = numpy.concatenate((predictions, cvx_models[suburb].predict(X_test[suburb])))
    target = numpy.concatenate((target, Y_test[suburb]))
print("RMSE:", numpy.sqrt(metrics.mean_squared_error(predictions, target)))