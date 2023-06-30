from __future__ import print_function
from fenics import *
import matplotlib.pyplot as plt
import numpy as np
from ufl import nabla_div
from matplotlib import cm
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable



nx = ny = 5
x = np.linspace(0.0, 1.0, nx + 1)
y = np.linspace(0.0, 1.0, ny + 1)
X, Y = np.meshgrid(x, y) 
mu = 1E-4
b = Expression(('1','1'), degree=2)
bv = as_vector((b,))
f = Expression(' 2*mu*sin(2*pi*x[0])*(2*pi*pi*(x[1]-x[1]*x[1])+1) + 2*pi*cos(2*pi*x[0])*(x[1]-x[1]*x[1]) + (1-2*x[1])*sin(2*pi*x[0])', mu=mu, degree=4)
bnorm = sqrt(1.0+1.0)
alpha = 10
gamma_cip = 0.1


# Exact solution
u_exact = Expression(' sin(2*pi*x[0])*(x[1]-x[1]*x[1]) ', mu=mu, degree=4)


H = []
HH = []
E1 = []
E2 = []
E3 = []
E4 = []
Err1 = []
Err2 = []
Err3 = []
Err4 = []


for i in range(5):

    nx = nx*2
    ny = ny*2

    # Mesh
    mesh = UnitSquareMesh(nx, ny)
    n = FacetNormal(mesh)
    x = SpatialCoordinate(mesh)
    h = mesh.hmin()
    H.append(h)
    HH.append(h*h)
    Pe = h*bnorm/(2*mu)
    if Pe<=3 : gamma = Pe/3
    else : gamma = 1.0
    bn = (dot(b,n)+abs(dot(b,n)))/2.0
    tho = h/(2*bnorm)*(1/np.tanh(Pe)-1/Pe)


    #Function space
    V = FunctionSpace(mesh, 'P', 1)
    V_dg = FunctionSpace (mesh, 'DG', 1)
    u_ex = interpolate(u_exact, V)

    # Boundaries conditions
    def boundary (x, on_boundary):
        return on_boundary
    bcs = DirichletBC (V, u_exact, boundary)
    bcd = DirichletBC ( V_dg, u_exact, boundary, 'geometric')

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



    E1.append(errornorm(u_exact,u1, 'L2'))
    E2.append(errornorm(u_exact,u2, 'L2'))
    E3.append(errornorm(u_exact,u3, 'L2'))
    E4.append(errornorm(u_exact,u4, 'L2'))

    Err1.append(errornorm(u_exact,u1, 'H1'))
    Err2.append(errornorm(u_exact,u2, 'H1'))
    Err3.append(errornorm(u_exact,u3, 'H1'))
    Err4.append(errornorm(u_exact,u4, 'H1'))


# Plot
plt.subplot(1,2,1)
plt.loglog(H,E1, label='CG', linestyle='--')
plt.loglog(H,E2, label='CIP', linestyle='--')
plt.loglog(H,E3, label='DG-N', linestyle='--')
plt.loglog(H,E4, label='SUPG', linestyle='--')
plt.loglog(H,HH, label='h²')
plt.title('Norme L2')
plt.legend()

plt.subplot(1,2,2)
plt.loglog(H,Err1, label='CG', linestyle='--')
plt.loglog(H,Err2, label='CIP', linestyle='--')
plt.loglog(H,Err3, label='DG-N', linestyle='--')
plt.loglog(H,Err4, label='SUPG', linestyle='--')
plt.loglog(H,H, label='h')
plt.title('Norme H1')

plt.legend()

plt.show()