import numpy as np
import matplotlib.pyplot as plt
import constants as c
import upwind_vectorized_v2 as up_v2


def time_error(solver, space_points):

    m = 2  #2^m points for first iteration
    n = 6  #2^n points for last iteration
    T_max = 1  # Time seconds until we stop the simulation
    T_ex = 2**(n+1)  # Number of time steps in the reference (exact) solution
    u_ex = up_v2.solve_upwind(T_ex, space_points, T_max)
    delta_t_list = np.zeros(n-m+1)
    delta_x = c.L / (space_points - 1)
    for i in range(m,n+1):
        delta_t_list[i-m] = T_max/(2**(i+1)-1)
        print("CFL-condition: delta_t = ", delta_t_list[i-m], " < ", delta_x / (c.V0+c.C), " = delta_x/(V0+C)")
    error_list_rho = np.zeros(n-m)
    error_list_v = np.zeros(n-m)
    delta_t_list = np.zeros(n-m)
    for i in range(m,n):
        time_points = 2**(i+1) #Number of time points in each iteration
        delta_t = T_max/(time_points-1) #delta t in each iteration
        u = solver(time_points, space_points, T_max)
        print(u_ex[-1,:,0])
        error_rho = u_ex[-1,:,0]-u[-1,:,0]
        x_list=np.linspace(-c.L/2,c.L/2,space_points)
        plt.figure()
        plt.plot(x_list,u_ex[-1,:,0],label="exact")
        plt.plot(x_list, u[-1, :, 0], label="approx")
        plt.legend()
        plt.show()
        error_v = u_ex[-1,:,1]-u[-1,:,1]
        error_list_rho[i-m] = np.sqrt(delta_t)*np.linalg.norm(error_rho,2)
        error_list_v[i-m] = np.sqrt(delta_t)*np.linalg.norm(error_v,2)
        delta_t_list[i-m] = delta_t
        #x = np.linspace(-2500,2500, space_points)
        #plt.figure()
        #plt.plot(x, u[-1,:,0], label='Test')
        #plt.plot(x, u_ex[-1,:, 0], label='Exact')
        #plt.legend()
        #plt.show()

    return delta_t_list,error_list_rho,error_list_v
   
    
def plot_time_convergence(solver):
    space_points=2**12
    delta_t_list, error_rho, error_v = time_error(solver, space_points)
    plt.figure()
    plt.plot(delta_t_list, error_rho, label=r"$\rho$")
    plt.plot(delta_t_list, error_v, label= "v")
    plt.title("Convergence plot in time")
    plt.xlabel(r"$\Delta t$")
    plt.ylabel("Error")
    plt.semilogx()
    plt.semilogy()
    plt.legend()
    plt.show()

def plot_time_convergence_2(solver1,  solver2, solver3, solver4):
    delta_t_list1, error_rho1, error_v1 = time_error(solver1, c.SPACE_POINTS)
    delta_t_list2, error_rho2, error_v2 = time_error(solver2, c.SPACE_POINTS)
    delta_t_list3, error_rho3, error_v3 = time_error(solver3, c.SPACE_POINTS)
    delta_t_list4, error_rho4, error_v4 = time_error(solver4, c.SPACE_POINTS)

    plt.figure()
    plt.loglog(delta_t_list1, error_rho1, label= r"Lax-Friedrichs")
    plt.loglog(delta_t_list2, error_rho2, label= r"Upwind")
    plt.loglog(delta_t_list3, error_rho3, label= r"Lax-Wendroff")
    plt.loglog(delta_t_list4, error_rho4, label= r"MacCormack")
    plt.title(r"Convergence plot of $\rho$ in time with Upwind as reference solution")
    plt.xlabel(r"$\Delta t$")
    plt.ylabel("Error")
    plt.legend()
    plt.grid()
    plt.savefig("conv_rho_time_up.pdf")
    plt.show()

    plt.figure()
    plt.loglog(delta_t_list1, error_v1, label=r"Lax-Friedrichs")
    plt.loglog(delta_t_list2, error_v2, label=r"Upwind")
    plt.loglog(delta_t_list3, error_v3, label=r"Lax-Wendroff")
    plt.loglog(delta_t_list4, error_v4, label=r"MacCormac")
    plt.title("Convergence plot of " + r'$v$' + " in time with Upwind as reference solution")
    plt.xlabel(r'$\Delta t$')
    plt.ylabel("Error")
    plt.grid()
    plt.legend()
    plt.savefig("conv_v_time_up.pdf")
    plt.show()


