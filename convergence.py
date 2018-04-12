import numpy as np
import matplotlib.pyplot as plt
import simple_lax as sl

import upwind_vectorized as up_v

import upwind as up
import constants as c

m=8 #2^m, gives number of points in space
max_time=5*60 #seconds
T=1000
X=2**(m)
V0=120
rho_max=140
E=100
rho0=40
L=5000 #meter
delta_t=0.001
delta_x=L/(X-1)  #meter

sigma=56.7
my=600
tau=0.5
c=54
grid_rho=np.zeros((T,X))
grid_v=np.zeros((T,X))
grid_rho[0]=np.ones(X)*rho0
grid_v[0]=sl.safe_v(grid_rho[0], V0, rho_max, E)



def spatial_convergence(solver,grid_rho, grid_v, T, X,delta_t,delta_x):
    convergence_list=np.zeros((2,c.M+1))
    rho_exact,v_exact=solver(grid_rho, grid_v, T, X,delta_t,delta_x)    # FIX THIS
    x_list = np.linspace(-c.L, c.L, len(rho_exact[0]))
    #plt.plot(x_list,rho_exact[-1])
    #plt.plot(x_list, v_exact[-1])
    #plt.show()
    exact_list=np.array([rho_exact[-1],v_exact[-1]])
    #print(exact_list)
    step_length_list=np.zeros(c.M+1)
    for i in range(1,c.M+1):
        len_points = 2 ** i
        new_exact_list=np.zeros((2,len_points))
        ratio=(len(exact_list[0])-1)/(len_points-1)

        for j in range(len_points):
            new_exact_list[:,j]=exact_list[:,int(j*ratio)]

        #print(new_exact_list)
        delta_x = c.L/(len_points-1)    #with endpoints
        print(delta_x)
        step_length_list[i-1]=delta_x

        grid_rho = np.zeros((c.TIME_POINTS, len_points))
        grid_v = np.zeros((c.TIME_POINTS, len_points))
        grid_rho[0] = np.ones(len_points) * c.RHO_0
        grid_v[0] = sl.safe_v(grid_rho[0], c.V0, c.RHO_MAX, c.E)

        rho_i,v_i = solver(grid_rho, grid_v, c.TIME_POINTS, len_points,delta_t,delta_x)
        i_list = np.array([rho_i[-1], v_i[-1]])

        convergence_list[0][i-1]=np.linalg.norm(new_exact_list[0]-i_list[0],np.inf)
        convergence_list[1][i-1] = np.linalg.norm(new_exact_list[1] - i_list[1],np.inf)
        print(len(convergence_list[0]), len(step_length_list))
        x_list2=np.linspace(-L,L,len(new_exact_list[0]))
        plt.plot(x_list,exact_list[0])
        plt.plot(x_list2,i_list[0])
        plt.show()
        #print(convergence_list)

    return convergence_list,step_length_list

def plot_convergence():
    conv_list,step_length_list=spatial_convergence(m,up.solve_upwind,grid_rho, grid_v, c.TIME_POINTS, x.SPACE_POINTS)
    print(len(conv_list),len(step_length_list))
    plt.loglog(step_length_list,conv_list[0])
    plt.loglog(step_length_list,conv_list[1])
    plt.show()

plot_convergence()


#Laget kun for å funke for Simple-Lax foreløpig:
def time_error(solver, X, rho0,delta_x,L,sigma,V0,rho_max,E,tau,c,my,rho_ex,v_ex,T_max,T_ex):
    n = 12 #
    error_list_rho = np.zeros(n)
    error_list_v = np.zeros(n)
    delta_t_list = np.zeros(n)
    for i in range(n):
        T = 2**(i+1) #Number of time points in each iteration
        delta_t = T_max/(T-1) #delta t in each iteration
        print(delta_t)
        #Initialization of a new grid - Skal vi lage en egen funksjon til dette?:
        grid_rho=np.zeros((T,X))
        grid_v=np.zeros((T,X))
        grid_rho[0]=np.ones(X)*rho0
        grid_v[0]=sl.safe_v(grid_rho[0], V0, rho_max, E)
        rho, v = solver(grid_rho, grid_v, T, X, rho0,delta_t,delta_x,L,sigma,V0,rho_max,E,tau,c,my)
        #print(rho_ex[T_ex-1])
        #print(rho[T-1])
        error_rho = rho_ex[T_ex-1]-rho[T-1]
        error_v = v_ex[T_ex-1]-v[T-1]
        error_list_rho[i] = np.sqrt(delta_x*delta_t)*np.linalg.norm(error_rho,2)
        error_list_v[i] = np.sqrt(delta_x*delta_t)*np.linalg.norm(error_v,2)
        delta_t_list[i] = delta_t
    return delta_t_list,error_list_rho,error_list_v

def time_convergence():

    T_max = 5 #Time (minutes?) until we stop the simulation
    T_ex = 10000 #Number of time steps in the reference (exact) solution 
    
    
    #Initialization of exact solution simple lax:
    grid_rho=np.zeros((T_ex,X))
    grid_v=np.zeros((T_ex,X))
    grid_rho[0]=np.ones(X)*rho0
    grid_v[0]=sl.safe_v(grid_rho[0], V0, rho_max, E)
    
    delta_t_min = T_max/(T_ex-1) #The delta T-value used in the exact solution
    
    rho_ex,v_ex = sl.solve_simple_lax(grid_rho, grid_v, T_ex, X, rho0,delta_t_min,delta_x,L,sigma,V0,rho_max,E,tau,c,my)
    print(rho_ex)
    
    delta_t_list,error_rho,error_v = time_error(sl.solve_simple_lax,X,         rho0,delta_x,L,sigma,V0,rho_max,E,tau,c,my,rho_ex,v_ex,T_max, T_ex)
    print(error_rho)
    print(error_v)
    
    print(delta_t_list)
    plt.figure()
    plt.plot(delta_t_list,error_rho,label=r"$\rho$")
    plt.plot(delta_t_list,error_v,label= "v")
    plt.title("Convergence plot in time")
    plt.xlabel(r"$\Delta t$")
    plt.ylabel("Error")
    plt.semilogx()
    plt.semilogy()
    plt.legend()
    plt.show()  
    
    
    grid_rho=np.zeros((T_ex,X))
    grid_v=np.zeros((T_ex,X))
    grid_rho[0]=np.ones(X)*rho0
    grid_v[0]=up.safe_v(grid_rho[0], V0, rho_max, E)
    
    delta_t_min = T_max/(T_ex-1) #The delta T-value used in the exact solution
    
    rho_ex,v_ex = up.solve_upwind(grid_rho, grid_v, T_ex, X, rho0,delta_t_min,delta_x,L,sigma,V0,rho_max,E,tau,c,my)
    print(rho_ex)
    
    delta_t_list,error_rho,error_v = time_error(up.solve_upwind,X,         rho0,delta_x,L,sigma,V0,rho_max,E,tau,c,my,rho_ex,v_ex,T_max, T_ex)
    print(error_rho)
    print(error_v)
    
    print(delta_t_list)
    plt.figure()
    plt.plot(delta_t_list,error_rho,label=r"$\rho$")
    plt.plot(delta_t_list,error_v,label= "v")
    plt.title("Convergence plot in time")
    plt.xlabel(r"$\Delta t$")
    plt.ylabel("Error")
    plt.semilogx()
    plt.semilogy()
    plt.legend()
    plt.show()  
    
time_convergence()
        
        
