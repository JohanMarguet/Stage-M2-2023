from __future__ import print_function
from fenics import *
import matplotlib.pyplot as plt
import numpy as np
from ufl import nabla_div
from matplotlib import cm
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable

nx = ny = 16
x = np.linspace(0.0, 1.0, nx + 1)
y = np.linspace(0.0, 1.0, ny + 1)
X, Y = np.meshgrid(x, y) 
mu = 1E-8
b = Expression(('1','1'), degree=2)
bv = as_vector((b,))
f = Expression(' 2*mu*sin(2*pi*x[0])*(2*pi*pi*(x[1]-x[1]*x[1])+1) + 2*pi*cos(2*pi*x[0])*(x[1]-x[1]*x[1]) + (1-2*x[1])*sin(2*pi*x[0])', mu=mu, degree=4)
bnorm = sqrt(1.0+1.0)
alpha = 10
gamma_cip = 0.1

# Mesh
mesh = UnitSquareMesh(nx, ny)
n = FacetNormal(mesh)
x = SpatialCoordinate(mesh)
h = mesh.hmin()
Pe = h*bnorm/(2*mu)
if Pe<=3 : gamma = Pe/3
else : gamma = 1.0
bn = (dot(b,n)+abs(dot(b,n)))/2.0
tho = h/(2*bnorm)*(1/np.tanh(Pe)-1/Pe)


#Function space
V = FunctionSpace(mesh, 'P', 1)
V_dg = FunctionSpace (mesh, 'DG', 1)
V_exact = FunctionSpace ( UnitSquareMesh(200,200) , 'P',1)

# Exact solution
u_exact = Expression(' sin(2*pi*x[0])*(x[1]-x[1]*x[1]) ', mu=mu, degree=4)
u_ex = interpolate(u_exact, V)

# Boundaries conditions
def boundary (x, on_boundary):
    return on_boundary
bcs = DirichletBC (V, u_exact, boundary)

# CG
u1 = TrialFunction(V)
v1 = TestFunction(V)
a1 = mu*dot(grad(u1), grad(v1))*dx + dot(b,grad(u1))*v1*dx
L1 = f*v1*dx
u1 = Function(V)
solve(a1 == L1, u1, bcs)

# CIP
u2 = TrialFunction(V)
v2 = TestFunction(V)
a2 = mu*inner(grad(u2),grad(v2))*dx + dot(b,grad(u2))*v2*dx + gamma_cip*avg(h)**2*bnorm*dot(jump(grad(u2),n),jump(grad(v2),n))*dS
L2 = f*v2*dx
u2 = Function(V)
solve(a2 == L2, u2, bcs)

# DG-N
u3 = TrialFunction(V_dg)
v3 = TestFunction(V_dg)
a_int = dot(grad(v3),mu*grad(u3)-b*u3)*dx
a_fac = mu*(alpha/h)*dot(jump(v3,n),jump(u3,n))*dS \
        + mu*alpha/h*dot(v3,u3)*ds \
        - mu*dot(avg(grad(v3)),jump(u3,n))*dS \
        - mu*dot(grad(v3),n)*u3*ds \
        - mu*dot(jump(v3,n),avg(grad(u3)))*dS \
        - mu*v3*dot(n,grad(u3))*ds
a_vel = dot(jump(v3),(dot(b,n('+'))+abs(dot(b,n('+'))))/2.0*u3('+')-(dot(b,n('-'))+abs(dot(b,n('-'))))/2.0*u3('-'))*dS \
        + dot(v3,bn*u3)*ds
a3 = a_int + a_fac + a_vel
L3 = v3*f*dx - mu*dot(grad(v3),n)*u_exact*ds + mu*(alpha/h)*dot(v3,u_exact)*ds + dot(v3,bn*u_exact)*ds
u3 = Function(V_dg)
solve(a3 == L3, u3)

# SUPG
u4 = TrialFunction(V)
v4 = TestFunction(V)
r = dot(b,grad(u4))-mu*nabla_div(grad(u4))
a4 = mu*dot(grad(u4), grad(v4))*dx + dot(b,grad(u4))*v4*dx + tho*(dot(b,grad(v4)))*r*dx
L4 = f*v4*dx + tho*f*dot(b,grad(v4))*dx
u4 = Function(V)
solve(a4 == L4, u4, bcs)


# Plot
fig = plt.figure(figsize=plt.figaspect(0.3))
vertex_values_u1 = u1.compute_vertex_values()
Z_vert1 = vertex_values_u1.reshape((nx + 1,ny + 1 ))
ax1 = fig.add_subplot(2, 3, 2, projection='3d',title='CG')
ax1.plot_surface(X, Y, Z_vert1, rstride=1, cstride=1, cmap='jet', edgecolor='none', vmin=u1.vector().min(), vmax=u1.vector().max())
ax1.view_init(15,-120)


vertex_values_u2 = u2.compute_vertex_values()
Z_vert2 = vertex_values_u2.reshape((nx + 1,ny + 1 ))
ax2 = fig.add_subplot(2, 3, 5, projection='3d', title='CIP')
ax2.plot_surface(X, Y, Z_vert2, rstride=1, cstride=1, cmap='jet', edgecolor='none', vmin=u2.vector().min(), vmax=u2.vector().max())
ax2.view_init(15,-120)

vertex_values_u3 = u3.compute_vertex_values()
Z_vert3 = vertex_values_u3.reshape((nx + 1,ny + 1 ))
ax3 = fig.add_subplot(2, 3, 3, projection='3d', title='DG-N')
ax3.plot_surface(X, Y, Z_vert3, rstride=1, cstride=1, cmap='jet', edgecolor='none', vmin=u3.vector().min(), vmax=u3.vector().max())
ax3.view_init(15,-120)

vertex_values_u4 = u4.compute_vertex_values()
Z_vert4 = vertex_values_u4.reshape((nx + 1,ny + 1 ))
ax4 = fig.add_subplot(2, 3, 4, projection='3d', title='SUPG')
ax4.plot_surface(X, Y, Z_vert4, rstride=1, cstride=1, cmap='jet', edgecolor='none', vmin=u4.vector().min(), vmax=u4.vector().max())
ax4.view_init(15,-120)

vertex_values_u_ex = u_ex.compute_vertex_values()
Z_vert6 = vertex_values_u_ex.reshape((nx + 1,ny + 1 ))
ax6 = fig.add_subplot(2, 3, 1, projection='3d', title='Exact solution')
ax6.plot_surface(X, Y, Z_vert6, rstride=1, cstride=1, cmap='jet', edgecolor='none', vmin=u_ex.vector().min(), vmax=u_ex.vector().max())
ax6.view_init(15,-120)

plt.show()