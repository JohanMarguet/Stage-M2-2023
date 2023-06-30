from __future__ import print_function
from fenics import *
from dolfin import *
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
from ufl import nabla_div

# Constants
mu = 10E-3
b = 1.0
dt = 0.1
f = Constant(1)
n_elements = 20
n_time_steps = 5
kappa = Constant(1.0)
alpha = Constant(2.5)

# Mesh
mesh = UnitIntervalMesh(n_elements)
n = FacetNormal(mesh)
h = mesh.hmin()
x = np.linspace(0, 1, n_elements+1)
gamma = Constant(0.5)
Pe = abs(b)*h/(2*mu)
tho = h/(2*abs(b))*(1/np.tanh(Pe)-1/Pe)
bv = as_vector((b,))
bn = (dot(bv,n) + abs(dot(bv,n)))/2.0
bnorm = max(bv)

# Function space
V_DG = FunctionSpace(mesh,"DG",1)
V = FunctionSpace(mesh, 'P', 1)

x_l = 0
x_r = 1


u_D = 0.0
def boundary (x, on_boundary):
    return on_boundary
bc = DirichletBC (V, u_D, boundary)
initial_condition = Expression(' sin(pi*x[0]) ',degree=1)

#CG
plt.figure(1)
plt.subplot(1, 4, 1)
u_old1 = interpolate(initial_condition,V)
plt.plot(x, u_old1.compute_vertex_values(), label="t=0.0")
u1 = TrialFunction(V)
v1 = TestFunction(V)
a1 = u1*v1*dx + dt*mu*dot(grad(u1),grad(v1))*dx + dt*b*u1.dx(0)*v1*dx
L1 = u_old1*v1*dx + dt*f*v1*dx
u_solution1 = Function(V)
time_current = 0.0
for i in range(n_time_steps):
    time_current += dt
    solve(a1 == L1,u_solution1,bc)
    u_old1.assign(u_solution1)
    plt.plot(x, u_solution1.compute_vertex_values(), label='t = %f'%(time_current))
plt.legend()
plt.ylim((0,1.6))
plt.title('CG')

#CIP
plt.figure(1)
plt.subplot(1, 4, 3)
u_old2 = interpolate(initial_condition,V)
plt.plot(x, u_old2.compute_vertex_values(), label="t=0.0")
u2 = TrialFunction(V)
v2 = TestFunction(V)
a2 = u2*v2*dx + dt*mu*u2.dx(0)*v2.dx(0)*dx + dt*b*v2*u2.dx(0)*dx + dt*gamma*avg(h)*avg(h)*bnorm('+')*dot((u2.dx(0)('-')-u2.dx(0)('+'))*n('+'),(v2.dx(0)('-')-v2.dx(0)('+'))*n('+'))*dS + dt*gamma*h*h*bnorm*u2.dx(0)*v2.dx(0)*ds 
L2 = u_old2*v2*dx + dt*f*v2*dx
u_solution2 = Function(V)
time_current = 0.0
for i in range(n_time_steps):
    time_current += dt
    solve(a2 == L2,u_solution2,bc)
    u_old2.assign(u_solution2)
    plt.plot(x, u_solution2.compute_vertex_values(), label='t = %f'%(time_current))
plt.legend()
plt.ylim((0,1.6))
plt.title('CIP')

#SUPG
plt.figure(1)
plt.subplot(1, 4, 2)
u_old3 = interpolate(initial_condition,V)
plt.plot(x, u_old3.compute_vertex_values(), label="t=0.0")
muh = mu*(1+tho*abs(b)*abs(b)/mu)
u3 = TrialFunction(V)
v3 = TestFunction(V)
a3 = u3*v3*dx + dt*muh*dot(grad(u3),grad(v3))*dx + dt*b*u3.dx(0)*v3*dx
L3 = u_old3*v3*dx + dt*f*v3*dx
u_solution3 = Function(V)
time_current = 0.0
for i in range(n_time_steps):
    time_current += dt
    solve(a3 == L3,u_solution3,bc)
    u_old3.assign(u_solution3)
    plt.plot(x, u_solution3.compute_vertex_values(), label='t = %f'%(time_current))
plt.legend()
plt.ylim((0,1.6))
plt.title('SUPG')


#DG-N
plt.figure(1)
plt.subplot(1, 4, 4)
u_old4 = interpolate(initial_condition,V_DG)
plt.plot(x, u_old4.compute_vertex_values(), label="t=0.0")
u4 = TrialFunction(V_DG)
v4 = TestFunction(V_DG)
a_int = mu*u4.dx(0)*v4.dx(0)*dx - b*u4*v4.dx(0)*dx
a_fac = mu*(alpha/h)*dot(jump(v4,n),jump(u4,n))*dS + mu*(alpha/h)*v4*u4*ds - mu*dot(avg(grad(v4)),jump(u4,n))*dS - mu*dot(grad(v4),n)*u4*ds - mu*dot(jump(v4,n),avg(grad(u4)))*dS - mu*v4*dot(grad(u4),n)*ds
a_vel = dot(jump(v4),bn('+')*u4('+')-bn('-')*u4('-'))*dS + dot(v4,bn*u4)*ds
a4 = u4*v4*dx + dt*a_int + dt*a_fac + dt*a_vel
L4 = u_old4*v4*dx + dt*f*v4*dx #- dt*mu*v4.dx(0)*u_D*ds + dt*mu*(alpha/h)*v4*u_D*ds + dt*dot(v4,bn*u_D)*ds
u_solution4 = Function(V_DG)
time_current = 0.0
for i in range(n_time_steps):
    time_current += dt
    solve(a4 == L4,u_solution4)
    u_old4.assign(u_solution4)
    plt.plot(x, u_solution4.compute_vertex_values(), label='t = %f'%(time_current))
plt.legend()
plt.ylim((0,1.6))
plt.title('DG-N')

plt.show()