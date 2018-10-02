import csv

data = []

print("1st", end = "")

with open("1.csv", encoding = "utf-8-sig", errors = 'ignore') as csvfile:
    reader = csv.reader(csvfile, delimiter = ',') # dialect = "excel"
    year_data = list(reader)
    print(" {} rows".format(len(year_data)))
    data += year_data[1:]

print("total", len(data))

print("2nd", end = "")

with open("2.csv", encoding = "utf-8-sig", errors = 'ignore') as csvfile:
    reader = csv.reader(csvfile, delimiter = ',')
    year_data = list(reader)
    print(" {} rows".format(len(year_data)))
    data += year_data[1:]

print("total", len(data))

print("dedup 2016")
dedup = set()
print("example:", year_data[1][0])
for item in year_data[1:]:
    dedup.add(item[0])

with open("2016.csv", encoding = "utf-8-sig", errors = 'ignore') as csvfile:
    reader = csv.reader(csvfile, delimiter = ',')
    year_data = list(reader)
    cache = []
    for item in year_data[1:]:
        if item[0] not in dedup:
            cache += [item]
    data += cache
    print("{} more rows from 2016, total {} rows in 2016.csv".format(len(cache), len(year_data)))

print("2017", end = "")

with open("2017.csv", encoding = "utf-8-sig", errors = 'ignore') as csvfile:
    reader = csv.reader(csvfile, delimiter = ',')
    year_data = list(reader)
    print(" {} rows".format(len(year_data)))
    data += year_data[1:]

print("total", len(data))
