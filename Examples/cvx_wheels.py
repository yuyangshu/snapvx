import cvxpy
import numpy

class linear_regression:
    def __init__(self):
        self.dimensionality = 0
        self.coefficients = None
        self.intercept = None

    def fit(self, X, Y):
        assert(isinstance(X, numpy.ndarray))
        assert(isinstance(Y, numpy.ndarray))
        assert(len(X.shape) == 2 and len(Y.shape) == 1)
        assert(X.shape[0] == Y.shape[0])

        self.dimensionality = Y.shape[0]
        coefficients = cvxpy.Variable(shape = len(X[0]))
        intercept = cvxpy.Variable(shape = 1)
        
        costs = 0
        for i in range(self.dimensionality):
            prediction = X[i] * coefficients + intercept
            costs += (prediction - Y[i]) ** 2

        problem = cvxpy.Problem(cvxpy.Minimize(costs))
        problem.solve()

        self.coefficients = coefficients
        self.intercept = intercept

    def predict(self, data):
        predictions = []
        for item in data:
            predictions.append((item * self.coefficients + self.intercept).value[0])
        
        return numpy.array(predictions)

def root_mean_squared_error(A, B):
    assert(isinstance(A, numpy.ndarray))
    assert(isinstance(B, numpy.ndarray))
    assert(A.shape[0] == B.shape[0])

    n = B.shape[0]
    sum_of_squared_errors = 0
    for i in range(n):
        sum_of_squared_errors += (A[i] - B[i]) ** 2
    
    return numpy.sqrt(sum_of_squared_errors / n)