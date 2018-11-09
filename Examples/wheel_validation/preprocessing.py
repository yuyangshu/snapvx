import collections
import csv
import time
import pickle
from datetime import datetime
import numpy
from scipy import spatial

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

def load_data(start, stop, verbose = False):
    def read_csv_file(fname):
        with open(fname + ".csv", encoding = "utf-8-sig", errors = 'ignore') as csvfile:
            reader = csv.reader(csvfile, delimiter = ',') # dialect = "excel"
            return list(reader)

    def distance_to_entities(coordinates):
        with open("stations", "rb") as f:
            station_tree = spatial.KDTree(pickle.load(f))
        with open("hospitals", "rb") as f:
            hospital_tree = spatial.KDTree(pickle.load(f))
        with open("high_schools", "rb") as f:
            school_tree = spatial.KDTree(pickle.load(f))
        
        return distance(coordinates, station_tree), distance(coordinates, hospital_tree), distance(coordinates, school_tree)

    def extract_fields(item, date):
        def cast_and_mark_missing_values(x, row_number = 0):
            return int(x) if x != "" else -row_number

        coordinates = (float(item[73]), float(item[74]))
        return [cast_and_mark_missing_values(item[32], row_number = 1), # [0] area size
                cast_and_mark_missing_values(item[33], row_number = 2), # [1] number of bedrooms
                cast_and_mark_missing_values(item[34], row_number = 3), # [2] number of bathrooms
                cast_and_mark_missing_values(item[35], row_number = 4), # [3] number of parkings
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

class missing_value_replacer:
    def __init__(self):
        self.mapper = None
    
    def set_means(self, data):
        assert(len(data.shape) == 2)

        transposed = numpy.transpose(data)
        means = list(map(numpy.mean, transposed[0:4]))
        self.mapper = dict()
        for i in range(4):
            self.mapper[-i - 1] = means[i]

    def replace_missing(self, data):
        assert(len(data.shape) == 2)

        new_array = numpy.copy(data)
        for k, v in self.mapper.items():
            new_array[data == k] = v

        return new_array

class data_scaler:
    def __init__(self):
        self.base = None
        self.spread = None

    def set_scale(self, data):
        assert(len(data.shape) == 2)

        transposed = numpy.transpose(data)
        self.base = numpy.array(list(map(numpy.min, transposed[0:10])))
        maxes = numpy.array(list(map(numpy.max, transposed[0:10])))
        self.spread = maxes - self.base

    def scale(self, data):
        assert(len(data.shape) == 2)
        assert(data[0].shape[0] == 10)

        return (data - self.base) / self.spread