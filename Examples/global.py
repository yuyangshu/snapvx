import collections
import pickle
import numpy
import scipy
import cvxpy
from sklearn import metrics, model_selection, linear_model

import cvx

# globals
number_of_transactions_needed = 60
number_of_neighbours = 5
lamb = 100

with open("suburbs_geolocations", 'rb') as f:
    suburbs = pickle.load(f)

# data = cvx.extract_fields(cvx.load_data("all", False, False), "reduced")
# selected_data = [item for item in data if int(item[4]) >= 2729 and int(item[4]) <= 2789]

# print("total number of transactions:", len(selected_data)) # 37830
# with open("selected_data", 'wb') as f:
#     pickle.dump(selected_data, f)

with open("selected_data", 'rb') as f:
    selected_data = pickle.load(f)

# lr model
X = [item[:4] for item in selected_data] # :-3 for full model
Y = [item[-1] for item in selected_data]
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = 0.2, random_state = 0)
cvx.lr_solve(X_train, X_test, Y_train, Y_test)

# convex optimization

# for item in selected_data:
#     if (item[7], int(item[8])) in suburbs:
#         count[item[7], int(item[8])] += 1
# sorted_count = sorted(count.items(), key = lambda x: x[1], reverse = True)
# for item in sorted_count:
#     print(item)

seg_data = collections.defaultdict(list)
for item in selected_data:
    if item[7] in suburbs and int(item[8]) == suburbs[item[7]][0]:
        seg_data[item[7]].append(item)

# filter for suburbs that will be modelled
suburbs_to_model = []
for item in seg_data:
    if len(seg_data[item]) > number_of_transactions_needed:
        suburbs_to_model.append(item)

# number of models
n = len(suburbs_to_model)
suburbs_to_model = sorted(suburbs_to_model)

# build kdtree to lookup nearest neighbours
coordinates, lookup_table = [], []
for i in range(n):
    coordinates.append(suburbs[suburbs_to_model[i]][1:])
    lookup_table.append(suburbs_to_model[i])
kdtree = scipy.spatial.KDTree(coordinates)

# print(suburbs_to_model, '\n', coordinates, sep = '')
# ['Alexandria', 'Ashfield', 'Auburn', 'Avalon Beach', 'Balgowlah', 'Balmain', 'Bankstown', 'Baulkham Hills', 'Bellevue Hill', 'Belmore', 'Bexley', 'Blacktown', 'Bondi', 'Bondi Beach', 'Botany', 'Burwood', 'Cabramatta', 'Camperdown', 'Campsie', 'Caringbah', 'Carlingford', 'Carlton', 'Castle Hill', 'Chatswood', 'Cherrybrook', 'Coogee', 'Cranebrook', 'Cremorne', 'Cronulla', 'Crows Nest', 'Darlinghurst', 'Dee Why', 'Drummoyne', 'Dulwich Hill', 'Earlwood', 'Eastwood', 'Elizabeth Bay', 'Engadine', 'Epping', 'Fairfield', 'Freshwater', 'Gladesville', 'Glebe', 'Glenmore Park', 'Glenwood', 'Granville', 'Greenacre', 'Greystanes', 'Guildford', 'Gymea', 'Haymarket', 'Homebush West', 'Hornsby', 'Hurstville', 'Ingleburn', 'Kellyville', 'Kellyville Ridge', 'Killara', 'Kingsford', 'Kirrawee', 'Kogarah', 'Lakemba', 'Lane Cove', 'Lane Cove North', 'Leichhardt', 'Lidcombe', 'Liverpool', 'Macquarie Fields', 'Manly', 'Maroubra', 'Marrickville', 'Marsfield', 'Mascot', 'Merrylands', 'Miranda', 'Mona Vale', 'Mortdale', 'Mosman', 'Mount Annan', 'Mount Druitt', 'Narrabeen', 'Neutral Bay', 'Newtown', 'North Parramatta', 'North Sydney', 'Northmead', 'Paddington', 'Padstow', 'Panania', 'Parramatta', 'Penrith', 'Penshurst', 'Potts Point', 'Prestons', 'Punchbowl', 'Pyrmont', 'Quakers Hill', 'Randwick', 'Redfern', 'Revesby', 'Rhodes', 'Rockdale', 'Rooty Hill', 'Rose Bay', 'Rozelle', 'Ryde', 'Sans Souci', 'Seven Hills', 'South Hurstville', 'St Clair', 'St Helens Park', 'St Ives', 'St Leonards', 'St Marys', 'Stanhope Gardens', 'Strathfield', 'Surry Hills', 'Sutherland', 'Sydney', 'Toongabbie', 'Turramurra', 'Wahroonga', 'Waterloo', 'Wentworth Point', 'West Pennant Hills', 'West Ryde', 'Westmead', 'Wiley Park', 'Winston Hills', 'Woollahra', 'Yagoona']
# [(-33.90166, 151.20007), (-33.8909735, 151.125756), (-33.8485157, 151.0296696), (-33.6302836, 151.3297698), (-33.79645, 151.25854), (-33.85895, 151.17906), (-33.91817, 151.03497), (-33.76288, 150.99212), (-33.8826955, 151.2539627), (-33.9178204, 151.0908086), (-33.9568, 151.12621), (-33.771, 150.9063), (-33.89195, 151.26099), (-33.89102, 151.277726), (-33.94513, 151.19934), (-33.8799121, 151.1024569), (-33.894444, 150.9375), (-33.8899117, 151.1767348), (-33.91057840000001, 151.1024569), (-34.0452, 151.1218), (-33.7799114, 151.0413133), (-33.9685296, 151.1246063), (-33.7270691, 150.9947439), (-33.80077, 151.1796), (-33.72874669999999, 151.0413133), (-33.919, 151.2555), (-33.72088, 150.71302), (-33.82886939999999, 151.2255485), (-34.05744, 151.15219), (-33.82613, 151.20505), (-33.8780176, 151.2204441), (-33.7544, 151.2854), (-33.8523805, 151.1548847), (-33.9041707, 151.1374068), (-33.92018, 151.12682), (-33.790362, 151.081731), (-33.87143, 151.22841), (-34.065722, 151.012664), (-33.7746, 151.0788), (-33.87028, 150.95622), (-33.7749, 151.28783), (-33.8296098, 151.125756), (-33.87978, 151.18541), (-33.79, 150.676), (-33.734, 150.934), (-33.8404, 151.0079), (-33.9092, 151.0534), (-33.823244, 150.943633), (-33.8558528, 150.9948543), (-34.0333, 151.08556), (-33.88092, 151.20294), (-33.858623, 151.0791612), (-33.70489999999999, 151.09901), (-33.966667, 151.1), (-34.0043204, 150.8630244), (-33.71053, 150.95114), (-33.703, 150.919), (-33.76864, 151.16347), (-33.92457, 151.22765), (-34.034722, 151.071111), (-33.9674, 151.13648), (-33.9199504, 151.0791612), (-33.8148708, 151.1664351), (-33.807, 151.164), (-33.8847, 151.1573), (-33.86462, 151.04563), (-33.92092, 150.92314), (-33.99434, 150.88757), (-33.8060158, 151.2947775), (-33.9495, 151.2437), (-33.9086291, 151.1548847), (-33.77963, 151.10253), (-33.9383158, 151.1811052), (-33.83742, 150.99168), (-34.03562, 151.10276), (-33.6757024, 151.3064407), (-33.96932, 151.06969), (-33.8303776, 151.2393885), (-34.065, 150.76), (-33.7722, 150.8194), (-33.7231, 151.2952), (-33.83450000000001, 151.2184), (-33.897, 151.1793), (-33.8032, 151.0055), (-33.83965, 151.20541), (-33.7902573, 150.9918338), (-33.88477, 151.22621), (-33.95592, 151.03247), (-33.95793, 151.002), (-33.815, 151.001111), (-33.75, 150.7), (-33.9623045, 151.0849848), (-33.86795, 151.22411), (-33.94286, 150.87116), (-33.933333, 151.05), (-33.8687895, 151.1942171), (-33.7241591, 150.8889765), (-33.91643, 151.23653), (-33.892215, 151.205873), (-33.95001, 151.01016), (-33.82821, 151.08569), (-33.95329, 151.13996), (-33.771543, 150.843922), (-33.8754209, 151.265623), (-33.8661553, 151.1738213), (-33.815278, 151.101111), (-33.98898, 151.1393), (-33.7777, 150.9416), (-33.98, 151.1035), (-33.7987, 150.7824), (-34.1105, 150.8079), (-33.72136, 151.16844), (-33.8227402, 151.1942171), (-33.7589212, 150.7708274), (-33.723, 150.926), (-33.8808808, 151.0762495), (-33.886111, 151.211111), (-34.03314, 151.0583), (-33.8708464, 151.20733), (-33.78606, 150.95645), (-33.7338, 151.1301), (-33.7183, 151.1187), (-33.9004, 151.20664), (-33.828026, 151.0769775), (-33.7462388, 151.0296696), (-33.80818, 151.08352), (-33.80829, 150.98208), (-33.92594, 151.06347), (-33.77747, 150.97788), (-33.8865002, 151.2437606), (-33.9050378, 151.0209375)]

# build individual models and collect costs
models, X_train, X_test, Y_train, Y_test = [], [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)]
target = 0

for i in range(n):
    suburb = suburbs_to_model[i]

    # build the local lr model
    X = [item[:4] + [1] for item in seg_data[suburb]] # :-3 for full model
    Y = [item[-1] for item in seg_data[suburb]]
    X_train[i], X_test[i], Y_train[i], Y_test[i] = model_selection.train_test_split(X, Y, test_size = 0.2, random_state = 0)

    # linear regression model
    lr_model = linear_model.LinearRegression(n_jobs = -1).fit(X_train[i], Y_train[i])
    lr_coefficients = numpy.concatenate((lr_model.coef_[:-1], lr_model.intercept_.reshape(1,)))

    # add to overall cost
    models.append(cvxpy.Variable(shape = 5, value = lr_coefficients))
    for j in range(len(X_train[i])):
        target += (models[i] * X_train[i][j] - Y_train[i][j]) ** 2

# lasso terms
for i in range(n):
    nearest_suburbs = kdtree.query([coordinates[i]], number_of_neighbours + 1)[1][0][1:]
    for j in nearest_suburbs:
        distance = numpy.sqrt((coordinates[i][0] - coordinates[j][0]) ** 2 + (coordinates[i][1] - coordinates[j][1]) ** 2)
        target += lamb / distance * cvxpy.norm(models[i] - models[j], 2)

# solve
print("starting to solve")
problem = cvxpy.Problem(cvxpy.Minimize(target))
try:
    result = problem.solve(verbose = True)
except:
    print("resorting to SCS")
    result = problem.solve(solver = "SCS", verbose = True)

print("approximate training RMSE:", numpy.sqrt(result / sum([len(row) for row in X_train])))

# test
errors = 0
for i in range(n):
    for j in range(len(X_test[i])):
        errors += ((models[i] * X_test[i][j]).value - Y_test[i][j]) ** 2
print("RMSE:", numpy.sqrt(result / sum([len(row) for row in X_test])))