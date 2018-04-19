import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import constants as c
import functions as func
from matplotlib import cm
from time import time



def u_next_upwind(u_last, delta_t, delta_x, j, time, position):
    return u_last[j] - delta_t/delta_x*(func.f(u_last[j])-func.f(u_last[j-1])) \
           + delta_t*func.g(u_last, delta_x, j)\
           + delta_t*func.s(time, position, u_last, j)

def one_step_upwind(u_last, X, delta_t, delta_x ,time):
    u_next = np.zeros((X,2))
    #u_next[0, :] = c.RHO_0, func.safe_v(c.RHO_0)
    u_next[0,:] = u_last[1][0], func.safe_v(u_last[1][0])
    for j in range(1,X-1):
        position=j*delta_x-c.L/2
        u_next[j] = u_next_upwind(u_last, delta_t, delta_x, j, time, position)
        u_next[j][0] = min(c.RHO_MAX, u_next[j][0])
        u_next[j][1] = max(0, u_next[j][1])
    u_next[X-1]=2*u_next[X-2]-u_next[X-3]
    return u_next

def solve_upwind(T, X, MAX_TIME):
    grid_u = func.initialize_grid(T, X, c.RHO_0)
    delta_x = c.L/(X-1)
    delta_t = MAX_TIME/(T-1)
    for i in range(1, T):
        time=i*delta_t
        grid_u[i]=one_step_upwind(grid_u[i-1], X, delta_t, delta_x, time)
    return grid_u

def plot_upwind(T, X, grid_u):
    delta_x = c.L/(X-1)
    x=np.linspace(-X*delta_x,X*delta_x,X)
    plt.figure()
    plt.plot(x,grid_u[T-1])
    plt.title("Space points "+str(X))
    plt.show()


def plot_simple_lax_3d(T,delta_t,X,delta_x,grid_rho,grid_v):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x=np.arange(-X*delta_x/2,X*delta_x/2,delta_x)
    y=np.arange(0,T*delta_t,delta_t)
    x,y=np.meshgrid(x,y)
    ax.plot_surface(x, y, grid_rho,cmap=cm.coolwarm)
    #plt.show()
    #plt.figure()
    #plt.imshow(grid_rho,cmap=plt.get_cmap('rainbow'))

    plt.show()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = np.arange(-X * delta_x / 2, X * delta_x / 2, delta_x)
    y = np.arange(0, T * delta_t, delta_t)
    x, y = np.meshgrid(x, y)
    ax.plot_surface(x, y, grid_v, cmap=cm.coolwarm)
    #plt.show()
    #plt.figure()
    #plt.imshow(grid_v, cmap=plt.get_cmap('rainbow'))

    plt.show()

def main():
    grid_u = solve_upwind(c.TIME_POINTS, c.SPACE_POINTS, c.MAX_TIME)
    plot_simple_lax_3d(c.TIME_POINTS,c.delta_t, c.SPACE_POINTS, c.delta_x, grid_u[:,:,0],grid_u[:,:,0])
    #plot_upwind(c.TIME_POINTS, c.SPACE_POINTS, grid_u[:,:,0])
    #plot_upwind(c.TIME_POINTS, c.SPACE_POINTS, grid_u[:,:,1])

#main()
