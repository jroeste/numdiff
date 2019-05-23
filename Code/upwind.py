import numpy as np
import matplotlib.pyplot as plt

def safe_v(rho, V0, rho_max, E):
    return V0*(1-rho/rho_max)/(1+E*(rho/rho_max)**4)

def q_in(time):
    return 40

def phi(x, sigma):
    return (2*np.pi*sigma**2)**(-0.5)*np.exp(-x**2/(2*sigma**2))

def rho_next_upwind(rho_last, v_last, delta_t, delta_x, j, time, position,sigma):
    return rho_last[j] - delta_t/delta_x*(rho_last[j]*v_last[j]-rho_last[j-1]*v_last[j-1])+ delta_t*q_in(time)*phi(position,sigma)

def v_next_upwind(rho_last, v_last, delta_t, delta_x,j, tau, V0, rho_max, E, c, my):
    return v_last[j] - delta_t/delta_x*(v_last[j]**2/2-v_last[j-1]**2/2)\
           - delta_t*c**2/(delta_x*rho_last[j])*(rho_last[j+1]-rho_last[j])\
           +(delta_t/tau)*((V0*(1-rho_last[j]/rho_max))/(1+E*(rho_last[j]/rho_max)**4)-v_last[j])\
           + my*delta_t*(v_last[j+1]-2*v_last[j]+v_last[j-1])/(rho_last[j]*delta_x**2)


def one_step_upwind(rho_last, v_last, X, delta_t,delta_x ,time,L,sigma, rho0,V0,rho_max,E,tau,c,my):
    rho_next = np.zeros(X)
    v_next=np.zeros(X)
    rho_next[0]=rho0
    v_next[0]=safe_v(rho0,V0,rho_max,E)
    for j in range(1,X-1):
        position=j*delta_x-L/2
        rho_next[j], v_next[j]=rho_next_upwind(rho_last,v_last,delta_t,delta_x, j ,time,position,sigma),\
                                 v_next_upwind(rho_last,v_last,delta_t,delta_x, j, tau, V0, rho_max, E, c, my)
    rho_next[X-1]=2*rho_next[X-2]-rho_next[X-3]
    v_next[X-1]=2*v_next[X-2]-v_next[X-3]
    return rho_next, v_next

def solve_upwind(grid_rho, grid_v, T, X, rho0,delta_t,delta_x,L,sigma,V0,rho_max,E,tau,c,my):
    for i in range(1,T):
        time=i*delta_t
        grid_rho[i], grid_v[i]=one_step_upwind(grid_rho[i-1], grid_v[i-1], X, delta_t,delta_x ,time,L,sigma, rho0,V0,rho_max,E,tau,c,my)
    return grid_rho, grid_v

def plot_upwind(T,X,delta_x,grid_rho):
    x=np.linspace(-X*delta_x,X*delta_x,X)
    plt.plot(x,grid_rho[T-1])
    plt.show()

def main():
    T = 1000
    X = 2**8
    V0 = 120
    rho_max = 140
    E = 100
    rho0 = 10
    delta_t = 0.01
    delta_x = 20  # meter
    L = delta_x * X  # meter
    sigma = 56.7
    my = 600
    tau = 0.5
    c = 54

    grid_rho=np.zeros((T,X))
    grid_v=np.zeros((T,X))
    grid_rho[0]=np.ones(X)*rho0
    grid_v[0]=safe_v(grid_rho[0], V0, rho_max, E)
    grid_rho, grid_v = solve_upwind(grid_rho, grid_v, T, X, rho0, delta_t, delta_x, L, sigma, V0, rho_max, E, tau, c, my)

    plot_upwind(T,X,delta_x,grid_rho)
#main()
