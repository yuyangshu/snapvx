import collections
import pickle
import numpy
import scipy
import cvxpy
from sklearn import metrics, model_selection, linear_model

import cvx

# global
number_of_transactions_needed = 60

def grid_search(lamb = 1, number_of_neighbours = 5):

    def test():
        try:
            errors = 0
            for i in range(n):
                for j in range(len(X_test[i])):
                    errors += ((models[i] * X_test[i][j]).value - Y_test[i][j]) ** 2
            print("RMSE:", numpy.sqrt(result / sum([len(row) for row in X_test])))
        except:
            print("unable to test, some parameters are None")

    # build individual models and collect costs
    models = []
    target = 0

    for i in range(n):
        # linear regression model
        lr_model = linear_model.LinearRegression(n_jobs = -1).fit(X_train[i], Y_train[i])
        lr_coefficients = numpy.concatenate((lr_model.coef_[:-1], lr_model.intercept_.reshape(1,)))

        # add to overall cost
        models.append(cvxpy.Variable(shape = 8, value = lr_coefficients)) # 5 for benchmark, 8 for full model
        for j in range(len(X_train[i])):
            target += (models[i] * X_train[i][j] - Y_train[i][j]) ** 2

    # lasso terms
    for i in range(n):
        nearest_suburbs = kdtree.query([coordinates[i]], number_of_neighbours + 1)[1][0][1:]
        for j in nearest_suburbs:
            distance = numpy.sqrt((coordinates[i][0] - coordinates[j][0]) ** 2 + (coordinates[i][1] - coordinates[j][1]) ** 2)
            target += lamb / distance * cvxpy.norm(models[i] - models[j], 2)

    # solve
    print("\n------\n", "starting to solve with ECOS, lambda = {}".format(lamb), sep = '')
    problem = cvxpy.Problem(cvxpy.Minimize(target))

    # ECOS solver
    try:
        SCS_flag = False
        result = problem.solve()
        print("approximate training RMSE:", numpy.sqrt(result / sum([len(row) for row in X_train])))
    except:
        SCS_flag = True
        print("not solved with ECOS; nonetheless, test the partially trained parameters")
    test()

    if SCS_flag:
        print("resorting to SCS")
        try:
            result = problem.solve(solver = "SCS")
            print("approximate training RMSE:", numpy.sqrt(result / sum([len(row) for row in X_train])))
        except:
            print("unable to solve with SCS; still, test the partially trained parameters")
        test()
    


# __main__
with open("suburbs_geolocations", 'rb') as f:
    suburbs = pickle.load(f)

# data = cvx.extract_fields(cvx.load_data("all", False, False), "reduced")
# selected_data = [item for item in data if int(item[4]) >= 2729 and int(item[4]) <= 2789]

# print("total number of transactions:", len(selected_data)) # 37830
# with open("selected_data", 'wb') as f:
#     pickle.dump(selected_data, f)

with open("selected_data", 'rb') as f:
    selected_data = pickle.load(f)

# pick suburbs to model
seg_data = collections.defaultdict(list)
for item in selected_data:
    if item[7] in suburbs and int(item[8]) == suburbs[item[7]][0]:
        seg_data[item[7]].append(item)

suburbs_to_model = []
for item in seg_data:
    if len(seg_data[item]) > number_of_transactions_needed:
        suburbs_to_model.append(item)

# number of models
n = len(suburbs_to_model)
suburbs_to_model = sorted(suburbs_to_model)
suburb_pool = set(suburbs_to_model)

# build kdtree to lookup nearest neighbours
coordinates, lookup_table = [], []
for i in range(n):
    coordinates.append(suburbs[suburbs_to_model[i]][1:])
    lookup_table.append(suburbs_to_model[i])
kdtree = scipy.spatial.KDTree(coordinates)


# prepare data
X = [item[:-3] for item in selected_data if item[-3] in suburb_pool] # 4 for benchmark, -3 for full model
Y = [item[-1] for item in selected_data if item[-3] in suburb_pool]
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = 0.2, random_state = 0)

# scale price
cvx.set_scale(Y_train)
Y_train = cvx.scaled_prices(Y_train)
Y_test = cvx.scaled_prices(Y_test)


# global linear model
print("global linear model:")
cvx.lr_solve(X_train, X_test, Y_train, Y_test)


# local models

# prepare data
for to_init in X_train, X_test, Y_train, Y_test:
    to_init = [[] for _ in range(n)]

for i in range(n):
    suburb = suburbs_to_model[i]
    X = [item[:-3] + [1] for item in seg_data[suburb]] # 4 for benchmark, -3 for full model
    Y = [item[-1] for item in seg_data[suburb]]
    X_train[i], X_test[i], Y_train[i], Y_test[i] = model_selection.train_test_split(X, Y, test_size = 0.2, random_state = 0)
    Y_train[i] = cvx.scaled_prices(Y_train[i])
    Y_test[i] = cvx.scaled_prices(Y_test[i])

# local linear models
print("\nlocal linear model:")
Y_train_pred, Y_test_pred = numpy.array([]), numpy.array([])
Y_train_truth, Y_test_truth = [], []

for i in range(n):
    model = linear_model.LinearRegression(n_jobs = -1)
    model.fit(X_train[i], Y_train[i])
    Y_train_pred = numpy.concatenate((Y_train_pred, model.predict(X_train[i])))
    Y_test_pred = numpy.concatenate((Y_test_pred, model.predict(X_test[i])))
    Y_train_truth += Y_train[i]
    Y_test_truth += Y_test[i]

print("training RMSE:", numpy.sqrt(metrics.mean_squared_error(Y_train_pred, Y_train_truth)))
print("r2_score:", metrics.r2_score(Y_train_pred, Y_train_truth))

print("RMSE:", numpy.sqrt(metrics.mean_squared_error(Y_test_pred, Y_test_truth)))
print("r2_score:", metrics.r2_score(Y_test_pred, Y_test_truth))


# convex optimization
for i in range(-10, 10):
    grid_search(numpy.e ** i)
