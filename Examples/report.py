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



# configs
number_of_transactions_needed = 60

# load data
all_data = data_loader.load_data(days_in_reporting_period = 60, house_only = True, verbose = True)
with open("stratified_data", "wb") as f:
    pickle.dump(all_data, f)

for period in sorted(all_data.keys()):
    data = all_data[period]

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


    X = numpy.array([item[:-1] for item in data if item[10] in suburbs_to_model and item[11] == suburbs[item[10]][0]])
    Y = numpy.array([item[-1] for item in data if item[10] in suburbs_to_model and item[11] == suburbs[item[10]][0]])

    price_calculator = collections.defaultdict(list)
    for item in data:
        price_calculator[item[10]].append(item[12])


    # reporting
    print("{} transactions for {} suburbs between {} and {} days".format(sum([len(item) for item in transaction_count]),len(transaction_count), i, i + 60))
    ordered_suburbs = sorted(transaction_count, key = lambda key: transaction_count[key], reverse = True)
    # for key in ordered_suburbs:
    #     print(key, transaction_count[key], numpy.mean(price_calculator[key]), numpy.median(price_calculator[key]), sep = ", ")


    # global model setting
    imputer = preprocessing.Imputer(missing_values="NaN", strategy="mean")
    scaler = preprocessing.MinMaxScaler()
    Y_scaler = preprocessing.MinMaxScaler()

    # preserve suburb labels for reporting purpases
    X_train_with_suburb, X_test_with_suburb, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = 0.2, random_state = 0)

    X_train = imputer.fit_transform([item[:-2] for item in X_train_with_suburb])
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(imputer.transform([item[:-2] for item in X_test_with_suburb]))
    Y_train = Y_scaler.fit_transform(numpy.reshape(Y_train, (-1, 1))).flatten()


    # global level linear regression
    print("global linear regression")
    start_time = time.time()

    sklearn_model = linear_model.LinearRegression(n_jobs = -1)
    sklearn_model.fit(X_train, Y_train)
    # print("sklearn finished at", round(time.time() - start_time), "seconds" )
    # print(sklearn_model.coef_, sklearn_model.intercept_)
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

    # per period reporting, too specific for all periods
    # per_suburb_result = collections.defaultdict(list)
    # truth = collections.defaultdict(list)
    # for i in range(len(X_test)):
    # per_suburb_result[X_test_with_suburb[i][-2]].append(abs(raw_predictions[i]))
    # truth[X_test_with_suburb[i][-2]].append(Y_test[i])
    # for key in per_suburb_result:
    # per_suburb_result[key] = Y_scaler.inverse_transform(numpy.reshape(per_suburb_result[key], (-1, 1))).flatten()
    # for key in ordered_suburbs:
    # diff = numpy.abs(numpy.array(per_suburb_result[key]) - numpy.array(truth[key]))
    # print(key, numpy.mean(diff), numpy.median(diff), sep = ", ")


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


    # for suburb in ordered_suburbs:
    #     raw_predictions = numpy.abs(Y_scalers[suburb].inverse_transform(numpy.reshape(sklearn_model[suburb].predict(X_test[suburb]), (-1, 1))).flatten() - Y_test[suburb])
    #     print(suburb, numpy.mean(raw_predictions), numpy.median(raw_predictions), sep = ", ")