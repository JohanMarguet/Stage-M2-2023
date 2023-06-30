from __future__ import print_function
from http.client import MULTIPLE_CHOICES
from math import inf
from pyclbr import Function
from fenics import *
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
from matplotlib import cm
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
#import warnings
#import logging

# Time
t = 0
num_steps = 10 # number of time steps
dt = 0.01 # time step size
T = dt*num_steps # final time

# Create mesh and define function space
L = 10
xl = -L
xr = L
N = 200
mesh = IntervalMesh(N, xl, xr)
h = mesh.hmax()
x = np.linspace(xl,xr,N+1)

# Diffusivity contant
D = 1

# Define boundaries conditions
class PeriodicBoundary(SubDomain):
	def inside(self, x, on_boundary):
		return bool(near(x[0],xl) and on_boundary)
	def map(self, x, y):
		y[0] = x[0] - (xr - xl)

# Define function space for system of concentrations
V = FunctionSpace(mesh, 'P', 1, constrained_domain=PeriodicBoundary())
V0 = FunctionSpace(mesh, 'DG', 0)

# Define function
v = TestFunction(V)
u = TrialFunction(V)

# Define initial condition
u0 = Expression('1+cos(2*pi*x[0]/L)', L=L, degree=2)
u_n = interpolate(u0,V)

# plot the iniital conditions
#graph = plt.figure(figsize=(15,8))
plt.figure()
plt.subplot(231)
plt.plot(x,u_n.compute_vertex_values())
plt.title('t=0.0')
plt.ylim((-10,10))


# Bilinear form
a_b = u*v*dx + dt*dot(D*grad(u),grad(v))*dx
# Linear form
L_b = u_n*v*dx

alpha = 10

# Subdmomain integration
coordinate = mesh.coordinates() # getting the coordinates for each nodes

r = 1 # range of interaction kernel
tol = 1E-13

materials = MeshFunction("size_t", mesh, mesh.topology().dim())#, 0)<-I don't what the last ",0" means

ds = Measure('ds', domain=mesh, subdomain_data=materials)
dx = Measure('dx', domain=mesh, subdomain_data=materials)

#define DG order 0 element, one value per cell
DG_Function = Function(V0)

#define cell function
cf = MeshFunction("size_t", mesh, 1)

#plot
pl = [232, 233, 234, 235, 236]
m = 0

#initialize the kernel (as if x=x-l)
k = np.ones(N)
for i in range(int(N/2)+1,N) : k[i]=-1
k[0] = -1


# time step
for n in range(num_steps):

	#iterate on the cells of the mesh
	for c in range(len(cf.array())):
		#define the cell
		cellule_test=MeshEntity(mesh, 1, c)
		#fine the midle
		mesh_point=Point(cellule_test.midpoint()[0])
		class Interaction_range(SubDomain):
			def inside(self, x, on_boundary):
				if mesh_point[0]-r<=x and x<=mesh_point[0] :
					return 1
				elif x <= mesh_point[0]+r-2*L or x >= mesh_point[0]-r+2*L :
					return 1
				else :
					return 0
		
		subdomain_1 = Interaction_range()
		subdomain_1.mark(materials, 1)

		# Define convolution
		K = Function(V)
		K.vector().set_local(np.array(k[0:N]))

		nonlocal_interaction = alpha*u_n(mesh_point[0])*K*u_n*dx(1)
		DG_Function.vector()[c] = assemble(nonlocal_interaction)
		subdomain_1.mark(materials, 0)
		
		#shift the kernel
		k = np.roll(k,1)

	Nonlocalinteraction = Function(V)
	Nonlocalinteraction = interpolate(DG_Function, V)

	# Adding the functions together 
	F = L_b + dt*Nonlocalinteraction*v.dx(0)*dx

	u = Function(V)
	solve(a_b==F, u)
    
    # Update current time
	t += dt
	
	#Plot solution
	if dt-tol <= t <= dt+tol or 2*dt-tol <= t <= 2*dt+tol or dt*5-tol <= t <= dt*5+tol or dt*7-tol <= t <= dt*7+tol or T-tol <= t <= T+tol:
		plt.subplot(pl[m])
		plt.plot(x, u.compute_vertex_values())
		plt.title("t = %f" %t)
		plt.ylim((-10,10))
		m = m+1
		
	
	# Update previous solution
	u_n.assign(u)

plt.savefig("convolution_direct_1D.pdf")
plt.show()