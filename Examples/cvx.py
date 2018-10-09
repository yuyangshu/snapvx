import os.path
import csv
import heapq
import pickle
import collections
from datetime import datetime

import numpy
import snap
import cvxpy
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
        if load_dump and os.path.isfile("2017_data"):
            print("restoring dumped data (2017)")
            with open("2017_data", "rb") as f:
                return pickle.load(f)

        data = read_csv_file("2017")

        if dump:
            with open("2017_data", 'wb') as f:
                pickle.dump(data, f)

    return data[1:]

def extract_fields(data, fields = "full"):

    def get_date(date_str):
        try:
            return datetime.strptime(date_str, "%d/%m/%Y")
        except:
            return datetime.strptime(date_str, "%d/%m/%y")

    # minimal for snapvx validation
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

def set_scale(Y):
    global base, stretch
    base = float(min(Y))
    stretch = float(max(Y) - base)

def scaled_prices(Y):
    scaled = []
    for item in Y:
        scaled.append((item - base) / stretch)
    return scaled

def lr_solve(X_train, X_test, Y_train, Y_test):

    model = linear_model.LinearRegression(n_jobs = -1)
    model.fit(X_train, Y_train)
    print(model.coef_)

    # print training error
    Y_pred = model.predict(X_train)
    print("training RMSE:", numpy.sqrt(metrics.mean_squared_error(Y_pred, Y_train)))
    print("r2_score:", metrics.r2_score(Y_pred, Y_train))

    # print testing error
    Y_pred = model.predict(X_test)
    print("RMSE:", numpy.sqrt(metrics.mean_squared_error(Y_pred, Y_test)))
    print("r2_score:", metrics.r2_score(Y_pred, Y_test))

def cvx_solve(X_train, X_test, Y_train, Y_test, lamb = 1, num_neighbours = 3):

    def distance(a, b):
        return cvxpy.norm(a - b, 2).value
    
    def predict(item_location, item_X, *neighbours):
        total_weights = 0
        prediction_acc = 0
        for neighbour in neighbours:
            dist, j = neighbour
            if dist > 0:
                # property does not fall on any of the models
                total_weights += 1 / dist
                prediction_acc += (S[j] * item_X / dist).value
            else:
                # if the property falls right on one model
                return (S[j] * item_X).value

        return prediction_acc / total_weights

    # setup
    n = len(X_train)
    S = [cvxpy.Variable(shape = 4) for _ in range(n)]

    # record geolocations of models, also record vector form to compute distance
    locations = dict()
    for i in range(n):
        locations[i] = numpy.array((X_train[i][5], X_train[i][6]))

    # record the nodes' connections
    connections = collections.defaultdict(lambda: [])

    # turn training data into numpy arrays
    X_train = numpy.array([item[:3] + [1] for item in X_train])
    Y_train = numpy.array(Y_train)

    # collect costs
    target = 0
    constraints = []

    # add nodes, node cost <= square error
    # the original paper also added regularization terms like + mu * cvxpy.norm(S[i][:-1], 2) ** 2
    for i in range(n):
        target += (S[i] * X_train[i] - Y_train[i]) ** 2

    # add num_neighbours edges for each node, to nearest neighbours
    for i in range(n):
        edge_pool = []
        for j in range(n):
            if i != j:
                heapq.heappush(edge_pool, (distance(locations[i], locations[j]), i, j))
        for _ in range(num_neighbours):
            edge_to_add = heapq.heappop(edge_pool)
            dst, j = edge_to_add[0], edge_to_add[2]
            if j not in connections[i]:
                connections[i] += [(j, dst)]
                connections[j] += [(i, dst)]

    for i in range(n):
        for j, dst in connections[i]:
            if dst > 0:
                # lasso term
                target += lamb / dst * cvxpy.norm(S[i] - S[j], 2)
            else:
                # same location => constrain to be the same model
                constraints.append(S[i] == S[j])

    # learn the models
    problem = cvxpy.Problem(cvxpy.Minimize(target), constraints)
    try:
        result = problem.solve()
    except:
        print("resorting to SCS")
        result = problem.solve(solver= "SCS")

    # print total training error and model parameters
    print("approximate training RMSE:", numpy.sqrt(result / n))
    # for item in S:
        # print(item.value)

    # test on dev set
    predictions = []
    for item in X_test:
        item_location = numpy.array((item[5], item[6]))
        item_X = numpy.array((item[:3] + [1]))

        # find the nearest models and predict based on them
        neighbour_heap = []
        for i in range(n):
            heapq.heappush(neighbour_heap, (distance(locations[i], item_location), i))
        neighbours = []
        for i in range(num_neighbours):
            neighbours.append(heapq.heappop(neighbour_heap))
        predictions.append(predict(item_location, item_X, *neighbours))

    # sum of squared error
    # print(numpy.array(predictions) - numpy.array(Y_test))
    print("RMSE:", numpy.sqrt(metrics.mean_squared_error(predictions, Y_test)))


#__main__

## load data into selected_data ##
# data = extract_fields(load_data("all", False, False), "minimal")
# 2729-2789 has the highest # of transactions per day
# selected_data = [item for item in data if int(item[4]) >= 2500 and int(item[4]) <= 3000]
# with open("tiny_data", "wb") as f:
#     pickle.dump(selected_data, f)
with open("tiny_data", "rb") as f:
    selected_data = pickle.load(f)
print("using {} data points".format(len(selected_data)))

## prepare data ##
# train/test split
X = [item[:-1] + [1] for item in selected_data]
Y = [item[-1] for item in selected_data]
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = 0.2, random_state = 0)

# scale price
set_scale(Y_train)
Y_train = scaled_prices(Y_train)
Y_test = scaled_prices(Y_test)

## feed into model ##
# linear regression
lr_solve(X_train, X_test, Y_train, Y_test)

# convex optimization
cvx_solve(X_train, X_test, Y_train, Y_test, lamb = 2)