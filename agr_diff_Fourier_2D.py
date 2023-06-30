from __future__ import print_function
from ast import Expression
from dolfin import *
import numpy as np
import math as mt
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def convolution_2D(K, u_0, V, h, Nx, Ny):
    K_hat = np.fft.fft2(K, axes=(0,1))
    u_0_matrix = np.reshape(u_0.compute_vertex_values()[0:Nx*Ny],(Nx, Ny))
    u_hat = np.fft.fft2(u_0_matrix, axes=(0,1))
    u_conv_K_hat = h*K_hat*u_hat
    u_conv_K = np.fft.ifft2(u_conv_K_hat, axes=(0,1))
    u_conv_K = np.fft.fftshift(u_conv_K, axes=(0,1))
    u_conv_K = np.reshape(u_conv_K, Nx*Ny)
    u_convolution_K = Function(V)
    u_convolution_K.vector().set_local(np.array(u_conv_K[0:(Nx)*(Ny)]))
    return u_convolution_K


# Define the diffusion coefficient
D = 1


# Interaction range ok the kernel
r = 2


# Define the time step and final time
Nt = 200
T = 10.0
dt = T/Nt
t = 0


# Define the domain
N = 50
L = 2.5
x_l = y_l = -L
x_r = y_r = L
mesh = RectangleMesh(Point(x_l,y_l), Point(x_r,y_r), N, N)
h = mesh.hmin()
x = np.linspace(x_l, x_r-h, N)
y = np.linspace(y_l, y_r-h, N)

# Define periodic boundary conditions and function space
class PeriodicBoundary(SubDomain):
    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        return bool (((near(x[0],x_l) and on_boundary) or (near(x[1],y_l) and on_boundary)) and (not((near(x[0],x_r) and near (x[1],y_l) and on_boundary) or (near(x[0],x_l) and near (x[1],y_r)and on_boundary))))
    # Map right boundary (H) to left boundary (G)
    def map(self, x, y):
        if (near(x[0],x_r) and near(x[1],y_r)):
            y[0] = x[0] - (x_r-x_l)
            y[1] = x[1] - (y_r-y_l)
        elif (near(x[0],x_r)):
            y[0] = x[0] - (x_r-x_l)
            y[1] = x[1]
        else:
            y[0] = x[0]
            y[1] = x[1] - (y_r-y_l)
V = FunctionSpace(mesh, 'P', 1, constrained_domain=PeriodicBoundary())


# Define the initial condition
u0 = np.random.random_sample(N*N)
u_0 = Function(V)
u_0.vector().set_local(u0)
#u0 = Expression('1+sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)', L=L, degree=2)
#u_0 = interpolate(u0,V)


# Plot the initial solution
fig = plt.figure(figsize=(15,8))
plt.subplot(231)
graph = plot(u_0)
plt.title('t = 0.0')

# Define the source fuction
f = Expression('0.0', degree=4)


# Define the kernel for the convolution
K1 = np.zeros((N, N))
K2 = np.zeros((N, N))
E_0 = 1E-30
d = 1
for i in range(N) :
    for j in range(N) :
        """
        if -r <= x[i] < 0 or 0 < x[i] <= r :
            K1[i][j] = x[i]/sqrt(x[i]*x[i]+y[j]*y[j])
        if -r <= y[j] < 0 or 0 < y[j] <= r :
            K1[i][j] = y[i]/sqrt(x[i]*x[i]+y[j]*y[j])
        """
        if x[i] == y[j] == 0 :
            K1[i][j] = K1[i-1][j-1]
            K2[i][j] = K2[i-1][j-1]
        else :
            K1[i][j] = 4*E_0*(pow(d,12)/pow(sqrt(x[i]*x[i]+y[j]*y[j]),12)-pow(d,6)/pow(sqrt(x[i]*x[i]+y[j]*y[j]),6))*x[i]/sqrt(x[i]*x[i]+y[j]*y[j])
            K2[i][j] = 4*E_0*(pow(d,12)/pow(sqrt(x[i]*x[i]+y[j]*y[j]),12)-pow(d,6)/pow(sqrt(x[i]*x[i]+y[j]*y[j]),6))*y[i]/sqrt(x[i]*x[i]+y[j]*y[j])



# Define the test and trial functions
v = TestFunction(V)
u = TrialFunction(V)

alpha = 1

# Define the variational problem
a = u*v*dx + dt*D*dot(grad(u),grad(v))*dx
L = u_0*v*dx + dt*f*v*dx - dt*alpha*u_0*convolution_2D(K1,u_0,V,h,N,N)*v.dx(0)*dx - dt*alpha*u_0*convolution_2D(K2,u_0,V,h,N,N)*v.dx(1)*dx
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
        graph = plot(u)
        plt.title("t = %f" %t)
        k = k+1
plt.savefig("convolution_fourier_2D.pdf")
plt.show()
