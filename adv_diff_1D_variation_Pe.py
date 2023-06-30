from __future__ import print_function
from fenics import *
from dolfin import *
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

# Constants
mu = 0.01
b = 1
fig = plt.figure(figsize=(15,8))
###########
# N = 20
###########
n = 20
mesh = UnitIntervalMesh(n)
x = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, 'P', 1)

u_D = Expression('0',degree=2)
def boundary(x, on_boundary):
    return on_boundary
bc = DirichletBC(V, u_D, boundary)

u = TrialFunction(V)
v = TestFunction(V)
f = Expression('1', degree=2)
a = (mu*inner(grad(u),grad(v))*dx + b*u.dx(0)*v*dx)
L = f*v*dx

u = Function(V)
solve(a == L, u, bc)
plot(u,color='b', label='Pe = 2.5')

###########
# N = 40
###########
n = 40
mesh = UnitIntervalMesh(n)
x = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, 'P', 1)

u_D = Expression('0',degree=2)
def boundary(x, on_boundary):
    return on_boundary
bc = DirichletBC(V, u_D, boundary)

u = TrialFunction(V)
v = TestFunction(V)
f = Expression('1', degree=2)
a = (mu*inner(grad(u),grad(v))*dx + b*u.dx(0)*v*dx)
L = f*v*dx

u = Function(V)
solve(a == L, u, bc)
plot(u,color='k', label='Pe = 1.25')

###########
# N = 80
###########
n = 80
mesh = UnitIntervalMesh(n)
x = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, 'P', 1)

u_D = Expression('0',degree=2)
def boundary(x, on_boundary):
    return on_boundary
bc = DirichletBC(V, u_D, boundary)

u = TrialFunction(V)
v = TestFunction(V)
f = Expression('1', degree=2)
a = (mu*inner(grad(u),grad(v))*dx + b*u.dx(0)*v*dx)
L = f*v*dx

u = Function(V)
solve(a == L, u, bc)
plot(u,color='g', label='Pe = 0.625')

#################
#Exact solution
#################
u_exact = Expression('-1/(1*(exp(b/mu)-1))*(exp(b/mu*x[0])-1)+x[0]/b', degree=4, b=b, mu=mu)
V_ex = FunctionSpace(UnitIntervalMesh(1000), "CG", 2)
u_ex = interpolate(u_exact,V_ex)
plot(u_ex, color='r', label='Exact', linestyle='-.')

###########
# Plot
###########
plt.legend()
plt.savefig("variationPe.pdf")
plt.show()