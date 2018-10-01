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
                entries.append(list(map(lambda x: int(x) if x != "" else 0, item[32:36])) + \
                               [(get_date(item[16].split()[0]) - base_date).days, float(item[17])])

    return entries


#__main__

# data = extract_fields(load_data("all", False, False), "minimal")
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
