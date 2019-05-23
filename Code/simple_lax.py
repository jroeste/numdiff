import numpy as np
import matplotlib.pyplot as plt
import constants as c
from time import time


def rho_next_simple_lax(rho_last, v_last, delta_t, delta_x, j, time, position):
    return  1/2*(rho_last[j+1]+rho_last[j-1]) - v_last[j]*delta_t/(2*delta_x)*(rho_last[j+1]-rho_last[j-1])-\
            rho_last[j]*delta_t/(2*delta_x)*(v_last[j+1]- v_last[j-1]) + delta_t*c.q_in(time)*c.phi(position)

def v_next_simple_lax(rho_last, v_last, delta_t, delta_x,j):
    return ((v_last[j+1]+v_last[j-1])/2-v_last[j]*delta_t*(v_last[j+1]-v_last[j-1])/(2*delta_x))\
           +(delta_t/c.TAU)*((c.V0*(1-rho_last[j]/c.RHO_MAX))/(1+c.E*(rho_last[j]/c.RHO_MAX)**4)-v_last[j])-\
           (rho_last[j+1]-rho_last[j-1])*delta_t/(rho_last[j]*2*delta_x)*c.C**2+c.MY*delta_t*(v_last[j+1]-2*v_last[j]+v_last[j-1])/(rho_last[j]*delta_x**2)

def one_step_simple_lax(rho_last, v_last, X, delta_t,delta_x ,time):
    rho_next=np.zeros(X)
    v_next=np.zeros(X)
    rho_next[0]=c.RHO_0
    v_next[0]=c.safe_v(c.RHO_0)
    for j in range(1,X-1):
        position=j*delta_x-c.L/2
        rho_next[j], v_next[j]=rho_next_simple_lax(rho_last,v_last,delta_t,delta_x, j ,time,position),\
                                 v_next_simple_lax(rho_last,v_last,delta_t,delta_x, j)
    rho_next[X-1]=2*rho_next[X-2]-rho_next[X-3]
    v_next[X-1]=2*v_next[X-2]-v_next[X-3]

    return rho_next, v_next

def initialize_grid(time_points, space_points, rho0):
    grid_rho=np.zeros((time_points,space_points))
    grid_v=np.zeros((time_points,space_points))
    grid_rho[0]=np.ones(space_points)*rho0
    grid_v[0]=c.safe_v(grid_rho[0])
    return grid_rho,grid_v

def solve_simple_lax(time_points, space_points, rho0, delta_t,delta_x):
    grid_rho,grid_v = initialize_grid(time_points, space_points, rho0)
    for i in range(1,time_points):
        #print(i)
        time=i*delta_t
        grid_rho[i], grid_v[i]=one_step_simple_lax(grid_rho[i-1],grid_v[i-1],space_points,
                                                   delta_t,delta_x ,time)
    return grid_rho,grid_v

def plot_simple_lax(T,delta_t,X,delta_x,grid_rho,grid_v):
    x=np.linspace(-X*delta_x,X*delta_x,X)
    plt.figure()
    plt.plot(x,grid_rho[T-1])
    plt.show()
    plt.figure()
    plt.plot(x,grid_v[T-1])
    plt.show()

def plot_simple_lax_3d(T,delta_t,X,delta_x,grid_rho,grid_v):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = np.arange(-X * delta_x / 2, X * delta_x / 2, delta_x)
    y = np.arange(0, T * delta_t, delta_t)
    x, y = np.meshgrid(x, y)
    ax.plot_surface(x, y, grid_rho, cmap=cm.coolwarm)
    plt.show()

def main():
    grid_rho,grid_v = solve_simple_lax(c.TIME_POINTS, c.SPACE_POINTS,c.RHO_0,c.delta_t,c.delta_x)
    #plot_simple_lax(c.TIME_POINTS,c.delta_t,c.SPACE_POINTS,c.delta_x,grid_rho,grid_v)
    #plot_simple_lax_3d(c.TIME_POINTS,c.delta_t, c.SPACE_POINTS, c.delta_x, grid_rho, grid_v)




#main()


