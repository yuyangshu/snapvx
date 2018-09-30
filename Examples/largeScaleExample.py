from snapvx import *
import sys
import builtins
import numpy
import time

def laplace_reg(src, dst, data):
    return (norm(src['x'] - dst['x']), [])

numpy.random.seed(0)
sys.setrecursionlimit(10000)

num_nodes = 100
size_prob = 300

temp = GenRndDegK(num_nodes, 3)
gvx = TGraphVX(temp)

for i in range(num_nodes):
    x = Variable(size_prob,name='x')
    a = numpy.random.randn(size_prob)
    gvx.SetNodeObjective(i, builtins.sum(huber(x-a)))


gvx.AddEdgeObjectives(laplace_reg)

start = time.time()
gvx.Solve(Verbose=True, Rho=0.1)
ellapsed = time.time() - start
print(ellapsed, "seconds; with ADMM")
gvx.PrintSolution()

start = time.time()
gvx.Solve(UseADMM=False, Verbose=True)                                                                                             
ellapsed = time.time() - start
print(ellapsed, "seconds; no ADMM")