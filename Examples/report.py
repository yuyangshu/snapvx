import os.path
import collections
import time
import queue
import pickle
import threading
import numpy
import cvxpy
import xgboost
from scipy import spatial
from sklearn import preprocessing, metrics, model_selection, linear_model

import data_loader
import cvx_wheels


def process_period(period_queue, result_queue, log_file, log_lock):
    while not period_queue.empty():
        period, data = period_queue.get()
        print("working on period", period)

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


        # reporting
        # starting date, # of transactions, # of suburbs, average price
        result = str(period) + ", " + str(len(X)) + ", " + str(len(suburbs_to_model)) + ", " + str(numpy.mean(Y))

        # global linear regression
        sklearn_model = linear_model.LinearRegression(n_jobs = -1)
        sklearn_model.fit(X_train, Y_train)
        predictions = Y_scaler.inverse_transform(numpy.reshape(sklearn_model.predict(X_test), (-1, 1))).flatten()
        percentage_errors = numpy.abs((predictions - Y_test) / Y_test)
        result += ", " + str(numpy.sqrt(sum(numpy.power(percentage_errors, 2)) / len(percentage_errors)))

        # global xgboost
        num_boost_round = 50
        parameters = {"eta": 0.1, "seed": 0, "subsample": 0.8, "colsample_bytree": 0.8, "objective": "reg:linear", "max_depth": 15, "min_child_weight": 1, "silent": True}
        train_matrix = xgboost.DMatrix(X_train, Y_train)
        xgboost_model = xgboost.train(parameters, train_matrix, num_boost_round = num_boost_round)
        predictions = Y_scaler.inverse_transform(numpy.reshape(xgboost_model.predict(xgboost.DMatrix(X_test)), (-1, 1))).flatten()
        percentage_errors = numpy.abs((predictions - Y_test) / Y_test)
        result += ", " + str(numpy.sqrt(sum(numpy.power(percentage_errors, 2)) / len(percentage_errors)))
        
        # importance logging
        scores = xgboost_model.get_score(importance_type = "gain")
        log_lock.acquire()
        log_file.write(str(period))
        for item in sorted(scores, key = lambda item: scores[item]):
            log_file.write(", " + item[1:])
        log_file.write("\n")
        log_lock.release()


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
        sklearn_model = collections.defaultdict(lambda: linear_model.LinearRegression(n_jobs = -1))
        for suburb in suburbs_to_model:
            sklearn_model[suburb].fit(X_train[suburb], Y_train[suburb])
        predictions, target = [], []
        for suburb in suburbs_to_model:
            predictions = numpy.concatenate((predictions, Y_scalers[suburb].inverse_transform(numpy.reshape(sklearn_model[suburb].predict(X_test[suburb]), (-1, 1))).flatten()))
            target = numpy.concatenate((target, Y_test[suburb]))
        percentage_errors = numpy.abs((predictions - target) / target)
        result += ", " + str(numpy.sqrt(sum(numpy.power(percentage_errors, 2)) / len(percentage_errors)))

        result_queue.put(result)


if __name__ == "__main__":
    # configs
    number_of_transactions_needed = 60
    days_in_reporting_period = 60
    house_only = True
    file_name = "{}_day_{}".format(days_in_reporting_period, "houses" if house_only else "all") 

    # load data
    if os.path.isfile(file_name):
        with open(file_name, "rb") as f:
            all_data = pickle.load(f)
    else:
        all_data = data_loader.load_data(days_in_reporting_period, house_only = house_only, verbose = True)
        with open(file_name, "wb") as f:
            pickle.dump(all_data, f)

    period_queue = queue.Queue()
    for period in sorted(all_data.keys()):
        period_queue.put((period, all_data[period]))
    result_queue = queue.Queue()

    with open("importance_log.csv", "w") as log_file:
        log_lock = threading.Lock()

        threads = []
        for _ in range(12):
            new_thread = threading.Thread(target = process_period, args = (period_queue, result_queue, log_file, log_lock))
            new_thread.start()
            threads.append(new_thread)

        for thread in threads:
            thread.join()

        with open(file_name + ".csv", 'w') as f:
            while not result_queue.empty():
                f.write(result_queue.get() + '\n')