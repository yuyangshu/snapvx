import os.path
import csv
import math
import heapq
import pickle
import collections
from datetime import datetime

import numpy
import snap
import cvxpy
import snapvx
from sklearn import metrics, model_selection, linear_model

def load_data(years = "all", dump = False, load_dump = True):

    def read_csv_file(fname):
        with open(fname + ".csv", encoding = "utf-8-sig", errors = 'ignore') as csvfile:
            reader = csv.reader(csvfile, delimiter = ',') # dialect = "excel"
            return list(reader)

    assert(years in {"all", "2017"})

    if years == "all":

        if load_dump and os.path.isfile("data"):
            print("restoring dumped data (all)")
            with open("data", "rb") as f:
                return pickle.load(f)

        data = read_csv_file("1")

        year_data = read_csv_file("2")
        data += year_data[1:]
        dedup = set()
        for item in year_data[1:]:
            dedup.add(item[0])

        cache = []
        for item in read_csv_file("2016")[1:]:
            if item[0] not in dedup:
                cache.append(item)
        data += cache

        data += read_csv_file("2017")[1:]
        
        if dump:
            with open("data", 'wb') as f:
                pickle.dump(data, f)

    else:
        if load_dump and os.path.isfile("small_data"):
            print("restoring dumped data (2017)")
            with open("small_data", "rb") as f:
                return pickle.load(f)

        data = read_csv_file("2017")

        if dump:
            with open("small_data", 'wb') as f:
                pickle.dump(data, f)

    return data[1:]

def extract_fields(data, fields = "full"):

    def get_date(date_str):
        try:
            return datetime.strptime(date_str, "%d/%m/%Y")
        except:
            print(date_str, datetime.strptime(date_str, "%d/%m/%y"))
            return datetime.strptime(date_str, "%d/%m/%y")

    # minimal for parramatta local model
    assert(fields in {"full", "reduced", "minimal"})

    base_date = get_date("1/1/2001")
    entries = []

    if fields == "minimal":
        for item in data[1:]:
            # skip if no geo info or price recorded
            # need to be precise in local models so skip as long as property geoinfo is missing
            if item[73] == "" or item[17] == "" or float(item[17]) == 0:
                continue

            # parramatta station: -33.817303, 151.004823
            if item[13] == "Parramatta":
                # area size, # of bedrooms, # of baths, # of parkings, 
                # # of days since 1/1/2001, latitude, longitude, price
                entries.append(list(map(lambda x: int(x) if x != "" else 0, item[32:36])) + \
                               [(get_date(item[16].split()[0]) - base_date).days,
                                float(item[73]),
                                float(item[74]),
                                float(item[17])])

    return entries

def lr_solve(X_train, X_test, Y_train, Y_test):

    model = linear_model.LinearRegression(n_jobs = -1)
    model.fit(X_train, Y_train)
    print(model.coef_)
    Y_pred = model.predict(X_test)
    print("RMSE:", math.sqrt(metrics.mean_squared_error(Y_pred, Y_test)))
    print("r2_score:", metrics.r2_score(Y_pred, Y_test))

def vx_solve(X_train, X_test, Y_train, Y_test, lamb = 0, mu = 0.1):

    def distance(i, j):
        return cvxpy.norm(locations[i] - locations[j], 2).value

    # setup
    n = len(X_train)
    gvx = snapvx.TGraphVX()
    S = [cvxpy.Variable(shape = 4) for _ in range(n)]

    # record geolocations of models, also record vector form to compute distance
    geoinfo = dict()
    locations = dict()
    for i in range(n):
        geoinfo[i] = (X_train[i][5], X_train[i][6])
        locations[i] = numpy.array(geoinfo[i])

    # record the nodes' connections
    connections = collections.defaultdict(lambda: [])

    # turn training data into numpy arrays
    X_train = numpy.array([item[:3] + [1] for item in X_train])
    Y_train = numpy.array(Y_train)

    # add nodes, node cost = square error
    # the original paper also added regularization terms
    for i in range(n):
        # could add regularization, + mu * cvxpy.norm(S[i][:-1], 2) ** 2
        gvx.AddNode(NId = i, Objective = cvxpy.sum_squares(S[i] * X_train[i] - Y_train[i]))

    # add 3 edges for each node, to nearest neighbours
    for i in range(n):
        edge_pool = []
        for j in range(n):
            if geoinfo[i] != geoinfo[j]:
                heapq.heappush(edge_pool, (distance(i, j), i, j))
        for _ in range(3):
            edge_to_add = heapq.heappop(edge_pool)
            j = edge_to_add[2]
            if j not in connections[i]:
                gvx.AddEdge(SrcNId = i,
                            DstNId = j,
                            Objective = lamb / edge_to_add[0] * cvxpy.norm(S[i] - S[j], 2))
                connections[i] += [j]
                connections[j] += [i]

    gvx.Solve(Verbose=True, Rho=0.1)
    gvx.PrintSolution()
    
    # test the model


#__main__

## load data into selected_data ##
# data = extract_fields(load_data("all", False, False), "minimal")
# selected_data = [item for item in data if int(item[4]) >= 2729 and int(item[4]) <= 2789]
# with open("tiny_data", "wb") as f:
#     pickle.dump(selected_data, f)
with open("tiny_data", "rb") as f:
    selected_data = pickle.load(f)

## prepare data ##
X = [item[:-1] + [1] for item in selected_data]
Y = [item[-1] for item in selected_data]
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = 0.2, random_state = 0)

## feed into model ##
# lr_solve(X_train, X_test, Y_train, Y_test)

# if run regression with 7 features
# RMSE: 82025.51851690875
# r2_score: -3.961004990007666
# [-1.30471340e+01 -3.22226437e+03 -1.47154757e+04  2.26577883e+04
#  -3.09251108e+01 -2.67852669e+06  5.54130897e+06  0.00000000e+00]

# run regression with 3 features
# RMSE: 91193.75444985367
# r2_score: -13.579062442242591
# [  -14.90423172 -4560.70742336  7427.1270868      0.        ]

# run snapvx
vx_solve(X_train, X_test, Y_train, Y_test)