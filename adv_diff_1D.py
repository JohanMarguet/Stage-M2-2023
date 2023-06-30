from __future__ import print_function
import time
import os
import math
from fenics import *
from dolfin import *
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt


#Constants
mu = 10E-3
b = 1.0
gamma = Constant(0.25)
N = 20
f = Expression('sin(x[0])', degree=2, mu=mu, b=b)
alpha = Constant(2.50)
kappa = 1.0
gamma_cip = 0.1
xl = 0
xr = 1

#Mesh
mesh = UnitIntervalMesh(N)
n = FacetNormal(mesh)
h = mesh.hmin()
x = SpatialCoordinate(mesh)
X =np.linspace(xl, xr, N+1)


Pe = abs(b)*h/(2*mu)
tho = h/(2*abs(b))*(1/np.tanh(Pe)-1/Pe)

bv = as_vector((b,))
bn = (dot(bv,n) + abs(dot(bv,n)))/2.0
bnorm = max(bv)


V = FunctionSpace(mesh, 'P', 1)
Vd = FunctionSpace(mesh, 'DG', 1)

#Boundaries conditions
u_D = Expression('0',degree=2)
def boundary (x, on_boundary):
    return on_boundary
bc = DirichletBC (V, u_D, boundary)


#Numerical solution SUPG
muh = mu+tho*abs(b)*abs(b)
u1 = TrialFunction(V)
v1 = TestFunction(V)
r = -mu*div(grad(u1))+b*u1.dx(0)
a1 = mu*dot(grad(u1),grad(v1))*dx + b*u1.dx(0)*v1*dx + tho*r*b*v1.dx(0)*dx
L1 = f*v1*dx + tho*f*b*v1.dx(0)*dx
u1 = Function(V)
solve(a1 == L1, u1, bc)

#Numerical solution classic Galerkin
u2 = TrialFunction(V)
v2 = TestFunction(V)
a2 = mu*inner(grad(u2),grad(v2))*dx + b*u2.dx(0)*v2*dx
L2 = f*v2*dx
u2 = Function(V)
solve(a2 == L2, u2, bc)

#Numerical solution Discontinous Galerkin - N
u3 = TrialFunction(Vd)
v3 = TestFunction(Vd)
a_int = dot((mu*u3.dx(0)-b*u3),v3.dx(0))*dx
a_fac = mu*(alpha/h)*dot(jump(v3,n),jump(u3,n))*dS \
        + mu*(alpha/h)*v3*u3*ds - mu*dot(avg(grad(v3)),jump(u3,n))*dS \
        - mu*dot(grad(v3),n)*u3*ds - mu*dot(jump(v3,n),avg(grad(u3)))*dS \
        - mu*v3*dot(grad(u3),n)*ds
a_vel = dot(jump(v3),bn('+')*u3('+')-bn('-')*u3('-'))*dS \
    + dot(v3,bn*u3)*ds
a3 = a_int + a_fac + a_vel
L3 = v3*f*dx - mu*v3.dx(0)*u_D*ds + mu*(alpha/h)*v3*u_D*ds \
      + dot(v3,bn*u_D)*ds
u3 = Function(Vd)
solve(a3 == L3, u3)

#Numerical solution CIP
u4 = TrialFunction(V)
v4 = TestFunction(V)
a4 = mu*dot(grad(u4),grad(v4))*dx + b*v4*u4.dx(0)*dx + gamma_cip*h**2*bnorm*dot(jump(grad(u4),n),jump(grad(v4),n))*dS
L4 = f*v4*dx
u4 = Function(V)
solve(a4 == L4, u4, bc)

#Exact solution
if mu>=10E-3 :
    c1 = b/(mu*mu+b*b)*(1-1/(exp(b/mu)-1)*(cos(1)-mu/b*sin(1)-1))
    c2 = b/(mu*mu+b*b)*(1/(exp(b/mu)-1)*(cos(1)-mu/b*sin(1)-1))
    u_exact = Expression(' c1 + c2*exp(b/mu*x[0]) - b/(mu*mu+b*b)*cos(x[0]) + 1/(mu+b*b/mu)*sin(x[0]) ', degree=4, mu=mu, b=b, c1=c1, c2=c2)
else :
    c1 = b/(mu*mu+b*b)*(1-exp(-b/mu)*(cos(1)-mu/b*sin(1)-1))
    u_exact = Expression(' c1 + b/(mu*mu+b*b)*(exp(-b/mu*(1-x[0]))*(cos(1)-mu/b*sin(1)-1)) - b/(mu*mu+b*b)*cos(x[0]) + 1/(mu+b*b/mu)*sin(x[0]) ', degree=4, mu=mu, b=b, c1=c1)   
V_ex = FunctionSpace(UnitIntervalMesh(1000), "CG", 1)
u_ex = interpolate(u_exact,V_ex)

#Plot
plt.plot(X,u2.compute_vertex_values(),label='Classic Galerkin', linestyle='--')
plt.plot(X,u4.compute_vertex_values(),label='CIP',linestyle=':')
plt.plot(X,u3.compute_vertex_values(),label='DG-N')
plt.plot(X,u1.compute_vertex_values(),label='SUPG')
plt.plot(np.linspace(xl, xr, 1001),u_ex.compute_vertex_values(),label='Exact solution', color='r', linestyle=':')
plt.legend()
plt.show()