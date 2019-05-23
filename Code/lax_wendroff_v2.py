import numpy as np
import matplotlib.pyplot as plt
import constants as c
import functions as func


def u_next_half_step(u_last, delta_t, delta_x, j, time, position):
    return (u_last[j+1] + u_last[j] - delta_t /delta_x*(func.f(u_last[j+1]) - func.f(u_last[j])) \
           + delta_t*func.g(u_last, delta_x, j)
           + delta_t/2*(func.s(time, position, u_last, j+1)+func.s(time, position, u_last, j)))/2


def u_next_lax_wendroff(u_last, u_halfstep, delta_t, delta_x, j, time, position):
    u_halfstep[j] = u_next_half_step(u_last, delta_t, delta_x, j, time, position)
    return u_last[j] - delta_t/delta_x*(func.f(u_halfstep[j])-func.f(u_halfstep[j-1])) \
           + delta_t*func.g(u_halfstep, delta_x, j)\
           + delta_t/2*(func.s(time, position, u_halfstep, j)\
           + func.s(time, position, u_halfstep, j-1))

def one_step_lax_wendroff(u_last, X, delta_t, delta_x ,time, rho0, L):
    u_next = np.zeros((X,2))
    u_next[0,:] = rho0, func.safe_v(rho0)
    u_halfstep = np.zeros((X,2))
    u_halfstep[0,:] = u_next_half_step(u_last, delta_t, delta_x, 0, time, -L/2)
    for j in range(1,X-1):
        position=j*delta_x-L/2
        u_next[j] = u_next_lax_wendroff(u_last, u_halfstep, delta_t, delta_x, j, time, position)
    u_next[X-1]=2*u_next[X-2]-u_next[X-3]
    return u_next

def solve_lax_wendroff(T, X, MAX_TIME):
    rho0, L= c.RHO_0, c.L
    delta_x = L/(X-1)
    delta_t = MAX_TIME/(T-1)
    grid_u = func.initialize_grid(T, X, rho0)
    for i in range(1, T):
        time=i*delta_t
        grid_u[i]=one_step_lax_wendroff(grid_u[i-1], X, delta_t, delta_x, time, rho0, L)
    return grid_u

def plot_lax_wendroff(T, X, delta_x, grid_u):
    x=np.linspace(-X*delta_x,X*delta_x,X)
    plt.figure()
    plt.plot(x,grid_u[T-1])
    plt.show()

'''def main():
    grid_u = solve_lax_wendroff(c.TIME_POINTS, c.SPACE_POINTS, c.MAX_TIME)
    plot_lax_wendroff(c.TIME_POINTS, c.SPACE_POINTS, c.delta_x, grid_u[:,:,0])
    plot_lax_wendroff(c.TIME_POINTS, c.SPACE_POINTS, c.delta_x, grid_u[:,:,1])
main()'''
