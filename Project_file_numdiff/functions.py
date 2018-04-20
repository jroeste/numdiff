import numpy as np
import constants as c

def initialize_grid(T, X, rho0):
    grid_u = np.zeros((T, X, 2))
    grid_u[0, :, 0] = np.ones(X) * rho0
    grid_u[0, :, 1] = safe_v(grid_u[0, :, 0])
    return grid_u

def safe_v(rho):
    return c.V0 * (1 - rho / c.RHO_MAX) / (1 + c.E * (rho / c.RHO_MAX) ** 4)

def q_in(time):
    if time<30:
        return 0.5
    else:
        return 0

def phi(x):
    return (2*np.pi*c.SIGMA**2)**(-0.5)*np.exp(-x**2/(2*c.SIGMA**2))

def g(u_last, delta_x, j):
    g_step = np.zeros(2)
    g_step[:] = 0, - c.C**2*(u_last[j+1,0]-u_last[j,0])/(delta_x*u_last[j,0]) \
                + c.MY*(u_last[j+1,1]-2*u_last[j,1]+ u_last[j-1,1])/(u_last[j,0]*delta_x**2)
    return g_step

def g2(u_last, delta_x, j):
    g_step = np.zeros(2)
    g_step[:] = 0, - c.C**2*(u_last[j+1,0]-u_last[j-1,0])/(delta_x*u_last[j,0]) \
                + c.MY*(u_last[j+1,1]-2*u_last[j,1]+ u_last[j-1,1])/(u_last[j,0]*delta_x**2)
    return g_step

def g3(u_last, delta_x, j):
    g_step = np.zeros(2)
    g_step[:] = 0, + c.MY*(u_last[j+1,1]-2*u_last[j,1]+ u_last[j-1,1])/(u_last[j,0]*delta_x**2)
    return g_step


def f(u_last):
    f_step = np.zeros(2)
    f_step[:] = u_last[0]*u_last[1], (u_last[1]**2)/2
    return f_step

def f2(u_last):
    f_step = np.zeros(2)
    f_step[:] = u_last[0]*u_last[1], (u_last[1]**2)/2 + c.C**2*np.log(u_last[0])
    return f_step

def s(time, position, u_last, j):
    s_step = np.zeros(2)
    s_step[:] = q_in(time)*phi(position), \
                (1/c.TAU)*((c.V0*(1-u_last[j,0]/c.RHO_MAX))/(1+c.E*(u_last[j,0]/c.RHO_MAX)**4)- u_last[j,1])
    return s_step



#print(safe_v(0.085))