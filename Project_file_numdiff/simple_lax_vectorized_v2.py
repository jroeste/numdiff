import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import constants as c
from matplotlib import cm
from time import time


def f2(u_last, u_m):
    f_step = np.zeros(2)
    f_step[:] = 0, c.C **2*u_last[0]/u_m[0]
    return f_step

def f(u_last):
    f_step = np.zeros(2)
    f_step[:] = u_last[0]*u_last[1], (u_last[1]**2)/2
    return f_step


def s(time, position, u_last, delta_t, delta_x, j):
    s_step = np.zeros(2)
    s_step[:] = c.q_in(time)*c.phi(position), (1/c.TAU)*((c.V0*(1-u_last[j,0]/c.RHO_MAX))/(1+c.E*(u_last[j,0]/c.RHO_MAX)**4)
                                                         - u_last[j,1])+c.MY*delta_t*(u_last[j+1,1]-2*u_last[j,1]
                                                         + u_last[j-1,1])/(u_last[j,0]*delta_x**2)
    return s_step

def u_next_simple_lax(u_last, delta_t, delta_x, j, time, position):
    return (u_last[j+1]+u_last[j-1])/2 - delta_t/(2*delta_x)*(f(u_last[j+1])-f(u_last[j-1])) \
           - delta_t/(2*delta_x)*(f2(u_last[j+1], u_last[j])-f2(u_last[j-1], u_last[j]))\
           + delta_t*s(time, position, u_last, delta_t, delta_x, j)

def one_step_simple_lax(u_last, X, delta_t, delta_x ,time):
    u_next = np.zeros((X,2))
    u_next[0,:] = c.RHO_0, c.safe_v(c.RHO_0)
    for j in range(1,X-1):
        position=j*delta_x-c.L/2
        u_next[j] = u_next_simple_lax(u_last, delta_t, delta_x, j, time, position)
        u_next[j][0] = min(c.RHO_MAX, u_next[j][0])
        u_next[j][1] = max(0, u_next[j][1])
    u_next[X-1]=2*u_next[X-2]-u_next[X-3]
    return u_next

def solve_simple_lax(T, X, delta_t, delta_x):
    grid_u = c.initialize_grid(T, X, c.RHO_0)
    for i in range(1, T):
        time=i*delta_t
        grid_u[i]=one_step_simple_lax(grid_u[i-1], X, delta_t, delta_x, time)
    return grid_u

def plot_simple_lax(T, X, delta_x, grid_u):
    x=np.linspace(-X/2*delta_x,X/2*delta_x,X)
    plt.figure()
    plt.plot(x,grid_u[T-1])
    plt.show()


def plot_simple_lax_3d(T,delta_t,X,delta_x,grid_rho,grid_v):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x=np.arange(-X*delta_x/2,X*delta_x/2,delta_x)
    y=np.arange(0,T*delta_t,delta_t)
    x,y=np.meshgrid(x,y)
    ax.plot_surface(x, y, grid_rho,cmap=cm.coolwarm)

    plt.show()

def plot_simple_lax2_3d_v(T,delta_t,X,delta_x,grid_v):
    fig = plt.figure("Speed of cars (m/s)")
    ax = fig.gca(projection='3d')
    x=np.arange(-X*delta_x/2,X*delta_x/2,delta_x)
    y=np.arange(0,T*delta_t,delta_t)
    x,y=np.meshgrid(x,y)
    surf=ax.plot_surface(x,y,grid_v,cmap=cm.coolwarm,linewidth=0)
    ax.text2D(0.05, 0.95, "Speed of cars (m/s)", transform=ax.transAxes)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Time (s)")
    ax.set_zlabel("Speed (m/s)")
    fig.colorbar(surf,shrink=0.5)
    plt.show()


def plot_simple_lax2_3d_rho(T,delta_t,X,delta_x,grid_rho):
    fig = plt.figure("Density of cars (car/m)")
    ax = fig.gca(projection='3d')
    x=np.arange(-X*delta_x/2,X*delta_x/2,delta_x)
    y=np.arange(0,T*delta_t,delta_t)
    x,y=np.meshgrid(x,y)
    surf=ax.plot_surface(x,y,grid_rho,cmap=cm.coolwarm,linewidth=0)
    ax.text2D(0.05, 0.95, "Density of cars (car/m)", transform=ax.transAxes)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Time (s)")
    ax.set_zlabel("Density (car/m)")
    fig.colorbar(surf,shrink=0.5)
    plt.show()

def main():
    grid_u = solve_simple_lax(c.TIME_POINTS, c.SPACE_POINTS, c.delta_t, c.delta_x)
    plot_simple_lax(c.TIME_POINTS, c.SPACE_POINTS, c.delta_x, grid_u[:,:,0])
    #plot_simple_lax(c.TIME_POINTS, c.SPACE_POINTS, c.delta_x, grid_u[:,:,1])

#main()
