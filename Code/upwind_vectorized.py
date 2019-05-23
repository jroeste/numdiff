import numpy as np
import matplotlib.pyplot as plt
import constants as c


def f(u_last):
    f_step = np.zeros(2)
    f_step[:] = u_last[0]*u_last[1], (u_last[1]**2)/2 + c.C**2*np.log(u_last[0])
    return f_step

def s(time, position, u_last, delta_t, delta_x, j):
    s_step = np.zeros(2)
    s_step[:] = c.q_in(time)*c.phi(position), \
                (1/c.TAU)*((c.V0*(1-u_last[j,0]/c.RHO_MAX))/(1+c.E*(u_last[j,0]/c.RHO_MAX)**4)- u_last[j,1])\
                +c.MY*delta_t*(u_last[j+1,1]-2*u_last[j,1]+ u_last[j-1,1])/(u_last[j,0]*delta_x**2)
    return s_step

def u_next_upwind(u_last, delta_t, delta_x, j, time, position):
    return u_last[j] - delta_t/delta_x*(f(u_last[j])-f(u_last[j-1])) \
           + delta_t*s(time, position, u_last, delta_t, delta_x, j)

def one_step_upwind(u_last, X, delta_t, delta_x ,time):
    u_next = np.zeros((X,2))
    u_next[0,:] = c.RHO_0, c.safe_v(c.RHO_0)
    for j in range(1,X-1):
        position=j*delta_x-c.L/2
        u_next[j] = u_next_upwind(u_last, delta_t, delta_x, j, time, position)
        u_next[j][0] = min(c.RHO_MAX, u_next[j][0])
        u_next[j][1] = max(0, u_next[j][1])
    u_next[X-1]=2*u_next[X-2]-u_next[X-3]
    return u_next

def solve_upwind(T, X, delta_t, delta_x):
    grid_u = c.initialize_grid(T, X, c.RHO_0)
    for i in range(1, T):
        time=i*delta_t
        grid_u[i]=one_step_upwind(grid_u[i-1], X, delta_t, delta_x, time)
    return grid_u

def plot_upwind(T, X, delta_x, grid_u):
    x = np.linspace(-X / 2 * delta_x, X / 2 * delta_x, X)
    plt.figure()
    plt.plot(x,grid_u[T-1])
    plt.show()

def main():
    grid_u = solve_upwind(c.TIME_POINTS, c.SPACE_POINTS, c.delta_t, c.delta_x)
    plot_upwind(c.TIME_POINTS, c.SPACE_POINTS, c.delta_x, grid_u[:,:,0])
    #plot_upwind(c.TIME_POINTS, c.SPACE_POINTS, c.delta_x, grid_u[:,:,1])
#main()
