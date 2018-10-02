import os.path
import csv
import math
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
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
        return datetime.strptime(date_str, "%d/%m/%Y")

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

            # latitude = float(item[73])
            # longitude = float(item[74])
            # parramatta station: -33.817303, 151.004823
            if item[13] == "Parramatta":
                # area size, # of bedrooms, # of baths, # of parkings, # of days since 1/1/2001, price
                entries.append(list(map(lambda x: int(x) if x != "" else 0, item[32:36])) + \
                               [(get_date(item[16].split()[0]) - base_date).days,
                                float(item[73]),
                                float(item[74]),
                                float(item[17])])

    return entries


#__main__

data = extract_fields(load_data("all", False, False), "minimal")
# print(len(data))
# print(data[:10])

# dates = [item[4] for item in data]
# n, bins, patches = plt.hist(dates, 50, alpha=0.5)
# plt.show()
# 2729 - 2848, 120 days, 524 points
# n, bins, patches = plt.hist(dates, 100, alpha=0.5)
# plt.show()
# 2729 - 2789, 60 days, 292 points

selected_data = [item for item in data if int(item[4]) >= 2729 and int(item[4]) <= 2789]
# with open("tiny_data", "wb") as f:
#     pickle.dump(selected_data, f)
# with open("tiny_data", "rb") as f:
#     selected_data = pickle.load(f)

X = [item[:-1] + [1] for item in selected_data]
Y = [item[-1] for item in selected_data]
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = 0.2, random_state = 0)

model = linear_model.LinearRegression(n_jobs = -1)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
print("RMSE:", math.sqrt(metrics.mean_squared_error(Y_pred, Y_test)))
print("r2_score:", metrics.r2_score(Y_pred, Y_test))
# RMSE: 82025.51851690875
# r2_score: -3.961004990007666

# from snapvx import *

# global_lambda = 1
# gvx = TGraphVX()
# S = [Variable(shape = (6)) for _ in range(X_train)]

# for i in range(len(X_train)):
#     gvx.AddNode(NId = i, Objective = (S[i] - Y_train[i]) ** 2)

