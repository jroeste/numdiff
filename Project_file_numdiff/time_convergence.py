import numpy as np
import matplotlib.pyplot as plt
import constants as c


def time_error(solver, space_points):

    m = 4  #2^m points for first iteration
    n = 12  #2^n points for last iteration
    T_max = 1 * 10  # Time seconds until we stop the simulation
    T_ex = 2**(n+1)  # Number of time steps in the reference (exact) solution
    u_ex = solver(T_ex, space_points, T_max)
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
        error_rho = u_ex[-1,:,0]-u[-1,:,0]
        error_v = u_ex[-1,:,1]-u[-1,:,1]
        error_list_rho[i-m] = np.sqrt(delta_t)*np.linalg.norm(error_rho,2)
        error_list_v[i-m] = np.sqrt(delta_t)*np.linalg.norm(error_v,2)
        delta_t_list[i-m] = delta_t

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
    plt.loglog(delta_t_list2, error_rho2, label= r"Lax-Friedrichs v2")
    plt.loglog(delta_t_list3, error_rho3, label= r"Lax-Wendroff")
    plt.loglog(delta_t_list4, error_rho4, label= r"MacCormack")
    plt.title(r"Convergence plot of $\rho$ in time")
    plt.xlabel(r"$\Delta t$")
    plt.ylabel("Error")
    plt.legend()
    plt.grid()
    plt.savefig("conv_rho_time.pdf")
    plt.show()

    plt.figure()
    plt.loglog(delta_t_list1, error_v1, label=r"Lax-Friedrichs")
    plt.loglog(delta_t_list2, error_v2, label=r"Upwind")
    plt.loglog(delta_t_list3, error_v3, label=r"Lax-Wendroff")
    plt.loglog(delta_t_list4, error_v4, label=r"MacCormac")
    plt.title("Convergence plot of " + r'$v$' + " in time")
    plt.xlabel(r'$\Delta t$')
    plt.ylabel("Error")
    plt.grid()
    plt.legend()
    plt.savefig("conv_v_time.pdf")
    plt.show()


