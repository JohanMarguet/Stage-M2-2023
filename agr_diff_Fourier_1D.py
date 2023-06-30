from __future__ import print_function
from ast import Expression
from dolfin import *
import numpy as np
import math as mt
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Define the convolution 1D and 2D
def convolution_1D(u, K, V, h, N):
    K_hat = np.fft.fft(K)
    u_hat = np.fft.fft(u.compute_vertex_values()[0:N])
    u_conv_K_hat = h*u_hat*K_hat
    u_conv_K = np.fft.ifft(u_conv_K_hat)
    u_conv_K = np.fft.fftshift(u_conv_K)
    u_convolution_K = Function(V)
    u_convolution_K.vector().set_local(np.array(u_conv_K)[0:N])
    return u_convolution_K

# Define the diffusion coefficient
D = 1


# Interaction range ok the kernel
r = 1


# Define the time step and final time
Nt = 10
dt = 0.01 # time step size
T = dt*Nt # final time
t = 0


# Define the domain
N = 200
L = 10
x_l = -L
x_r = L
mesh = IntervalMesh(N, x_l, x_r)
h = mesh.hmin()
x = np.linspace(x_l, x_r, N+1)

# Boundaries conditions and function space
class PeriodicBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return bool(near(x[0],x_l) and on_boundary)
    def map(self, x, y):
        y[0] = x[0] - (x_r - x_l)
V = FunctionSpace(mesh, 'P', 1, constrained_domain=PeriodicBoundary())


# Define the initial condition
#u0 = np.random.random_sample(N)
#u_0 = Function(V)
#u_0.vector().set_local(u0)
u0 = Expression('1+cos(2*pi*x[0]/L)', L=L, degree=2)
u_0 = interpolate(u0,V)


# Plot the initial solution
#graph = plt.figure(figsize=(15,8))
plt.subplot(231)
plt.plot(x, u_0.compute_vertex_values())
plt.title('t = 0.0')
plt.ylim((-10,10))

# Define the source fuction
f = Expression('0.0', degree=4)


# Define the kernel for the convolution
K = np.zeros(N)
LJ = np.zeros(N)
E_0 = 1E-15
tol = 1E-10
i = 0
for i in range(N):
    """
    if 0 < x[i] <=r or -r <= x[i] < 0 :
        K[i] = 4*E_0*(1/pow(abs(x[i]),12)-1/pow(abs(x[i]),6))*x[i]/abs(x[i])          
    """
    if -r-tol <= x_l+i*h < 0 :
        K[i] = -1
    elif 0 < x_l+i*h <= r+tol :
        K[i] = 1

# Define the test and trial functions
v = TestFunction(V)
u = TrialFunction(V)

alpha = 10

# Define the variational problem
a = u*v*dx + dt*D*dot(grad(u),grad(v))*dx + dt*alpha*u*convolution_1D(u_0,K,V,h,N)*v.dx(0)*dx
L = u_0*v*dx + dt*f*v*dx
u = Function(V)

# Time-stepping loop
tol = 10E-4
k = 0
pl = [232, 233, 234, 235, 236]

while t < T:

    solve( a==L, u)

    # Update the time
    t += dt

    # Update the solution
    u_0.assign(u)

    #Plot solution at each time
    if dt-tol <= t <= dt+tol or dt*2-tol <= t <= dt*2+tol or dt*5-tol <= t <= dt*5+tol or dt*7-tol <= t <= dt*7+tol or T-tol <= t <= T+tol:
        plt.subplot(pl[k])
        plt.plot(x, u.compute_vertex_values())
        plt.title("t = %f" %t)
        plt.ylim((-10,10))
        k = k+1
plt.savefig("convolution_fourier_1D.pdf")
plt.show()