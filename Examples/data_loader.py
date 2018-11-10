import collections
import csv
import time
import pickle
from datetime import datetime
import numpy
from scipy import spatial


def load_data(start = 0, stop = numpy.Infinity, verbose = False):
    def read_csv_file(fname):
        with open(fname + ".csv", encoding = "utf-8-sig", errors = 'ignore') as csvfile:
            reader = csv.reader(csvfile, delimiter = ',') # dialect = "excel"
            return list(reader)

    def distance(coordinates, kdtree):
        return kdtree.query([coordinates], 1)[0][0]

    def get_date(date_str):
        try:
            return datetime.strptime(date_str, "%d/%m/%Y")
        except:
            return datetime.strptime(date_str, "%d/%m/%y")

    def get_postcode(postcode):
        try:
            return int(postcode)
        except:
            return 0

    def distance_to_entities(coordinates):
        with open("stations", "rb") as f:
            station_tree = spatial.KDTree(pickle.load(f))
        with open("hospitals", "rb") as f:
            hospital_tree = spatial.KDTree(pickle.load(f))
        with open("high_schools", "rb") as f:
            school_tree = spatial.KDTree(pickle.load(f))
        
        return distance(coordinates, station_tree), distance(coordinates, hospital_tree), distance(coordinates, school_tree)

    def extract_fields(item, date):
        def cast_and_mark_missing_values(x):
            return int(x) if x != "" else numpy.NaN

        coordinates = (float(item[73]), float(item[74]))
        return [cast_and_mark_missing_values(item[32]), # [0] area size
                cast_and_mark_missing_values(item[33]), # [1] number of bedrooms
                cast_and_mark_missing_values(item[34]), # [2] number of bathrooms
                cast_and_mark_missing_values(item[35]), # [3] number of parkings
                date, # [4] number of days since base date
                *distance_to_entities(coordinates), # [5, 6, 7] distance to nearest train station, hospital, and high school
                *coordinates, # [8, 9] latitude, longitude
                item[13], # [10] suburb
                get_postcode(item[14]), # [11] postcode
                float(item[17])] # [12] price

    if verbose:
        print("starting to load data")
        timer_base = time.time()
        count = 0

    data = read_csv_file("1") + read_csv_file("2")[1:]
    deduplication = set()
    for item in data[1:]:
        deduplication.add(item[0])

    for item in read_csv_file("2016")[1:]:
        if item[0] not in deduplication:
            data.append(item)
            deduplication.add(item[0])
    
    data += read_csv_file("2017")[1:]

    if verbose:
        print("finished reading csv files,", round(time.time() - timer_base), "seconds elapsed") # 42

    base_date = get_date("1/1/2001")
    selected_data = []

    for item in data[1:]:
        if verbose:
            count += 1
            if count % 100000 == 0:
                print(count, "entries processed, ", round(time.time() - timer_base), "seconds elapsed")

        if item[73] == "" or item[17] == "" or float(item[17]) == 0:
            continue

        date = (get_date(item[16].split()[0]) - base_date).days
        if date < start or date > stop:
            continue

        selected_data.append(extract_fields(item, date))

    if verbose:
        print("fields extracted from raw data,", round(time.time() - timer_base), "seconds elapsed")

    return selected_data