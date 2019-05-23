import numpy as np
import matplotlib.pyplot as plt
import constants as c
import simple_lax_vectorized as sl_v
import upwind_vectorized as up_v
import upwind_vectorized_v2 as up_v2
import mac_cormack as mc

"""
def spatial_convergence_vec(solver, T, X, delta_t, delta_x):
    startnumber = 3
    convergence_list = np.zeros((2, c.M-startnumber-1))
    u_exact = solver(T, X, delta_t, delta_x)
    exact_list = u_exact[-1]
    step_length_list = np.zeros(c.M -startnumber-1)
    x_list = np.linspace(-c.L / 2, c.L / 2, len(exact_list))
    plt.plot(x_list,exact_list[:,0])
    plt.show()
"""


def spatial_convergence_vec(solver, T, X, MAX_TIME, M, startnumber):
    convergence_list = np.zeros((2, M-startnumber-1))
    u_exact = solver(T, X, MAX_TIME)
    exact_list = u_exact[-1]
    step_length_list = np.zeros(M -startnumber-1)

    for j in range(startnumber,M-1):
        x_points = 2**(j+1)
        new_exact_list = np.zeros((x_points,2))
        ratio = (len(exact_list) - 1) / (x_points - 1)
        for h in range(x_points):
            new_exact_list[h] = exact_list[int(h * ratio)]

        delta_x = c.L / (x_points - 1)
        step_length_list[j - startnumber] = delta_x
        u = solver(T, x_points, MAX_TIME)
        j_list=u[-1]
        convergence_list[0][j-startnumber] = np.sqrt(delta_x) * np.linalg.norm(new_exact_list[:,0] - j_list[:,0], 2)
        convergence_list[1][j-startnumber] = np.sqrt(delta_x) * np.linalg.norm(new_exact_list[:,1] - j_list[:,1], 2)
        #x = np.linspace(-2500,2500, x_points)
        #plt.figure()
        #plt.plot(x, j_list[:,0], label='Test')
        #plt.plot(x, new_exact_list[:, 0], label='Exact')
        #plt.legend()
        #plt.show()
        print("Points: ", x_points)

    return step_length_list,convergence_list

'''def plot_convergence(method):
    conv_list, step_length_list = spatial_convergence_vec(method, c.TIME_POINTS, c.SPACE_POINTS, c.delta_t, c.delta_x)


    plt.loglog(step_length_list,conv_list[0],label='rho')
    plt.show()
    plt.figure()
    plt.loglog(step_length_list,conv_list[1],label='v')
    plt.xlabel("Steplength ($\Delta x$)")
    plt.ylabel("Error")
    plt.grid()
    plt.legend()
    plt.show()'''

def plot_spatial_convergence_lax(solver1, solver2):
    M=10
    time_points=1000
    space_points=2**M
    delta_t=0.01
    MAX_TIME = delta_t*time_points

    delta_x=c.L/(space_points-1)
    #delta_x_list1, conv_1 = spatial_convergence_vec(solver1, time_points,space_points,delta_t, MAX_TIME,M)
    delta_x_list2, conv_2 = spatial_convergence_vec(solver2, time_points, space_points, MAX_TIME, M)

    plt.figure()
    #plt.loglog(delta_x_list1, conv_1[0], label= r"Lax-Fredrich")
    plt.loglog(delta_x_list2, conv_2[0], label= r"Lax-Fredrich v2")
    plt.title("Convergence plot of "+ r'$\rho$' +" in space")
    plt.xlabel(r'$\Delta x$')
    plt.ylabel("Error")
    plt.legend()
    plt.savefig("conv_rho_space_lax.pdf")
    plt.show()

    plt.figure()
    #plt.loglog(delta_x_list1, conv_1[1], label=r"Lax-Fredrich")
    plt.loglog(delta_x_list2, conv_2[1], label=r"Lax-Fredrich v2")
    plt.title("Convergence plot of " + r'$v$' + " in space")
    plt.xlabel(r'$\Delta x$')
    plt.ylabel("Error")
    plt.savefig("conv_v_space_lax.pdf")
    plt.legend()
    plt.show()

def plot_spatial_convergence(solver1, solver2, solver3,solver4):
    #M = 10
    #MAX_TIME = 5
    #time_points = 500
    #space_points = 2 ** M
    #m = 3
    #delta_t = MAX_TIME/(time_points-1)
    #delta_x = c.L/(space_points-1)
    #print("delta_t: ", delta_t)
    #print("delta_x: ", delta_x)
    #print("CFL-condition: delta_t = ", delta_t, " < ", delta_x/(c.V0+c.C), " = delta_x/(V0+C)")

    M = 14
    m = 6
    MAX_TIME = 1
    time_points = 100
    space_points = 2 ** M

    delta_x_list1, conv_1 = spatial_convergence_vec(solver1, time_points, space_points, MAX_TIME, M, m)

    M = 12
    m = 3
    MAX_TIME = 40
    time_points = 2000
    space_points = 2**M

    delta_x_list2, conv_2 = spatial_convergence_vec(solver2, time_points, space_points, MAX_TIME, M, m)

    M = 10
    m = 3
    MAX_TIME = 160
    time_points = 1000
    space_points = 2 ** M

    delta_x_list3, conv_3 = spatial_convergence_vec(solver3, time_points, space_points, MAX_TIME, M, m)

    M = 15
    m = 3
    MAX_TIME = 1
    time_points = 800
    space_points = 2 ** M

    delta_x_list4, conv_4 = spatial_convergence_vec(solver4, time_points, space_points, MAX_TIME, M, m)

    plt.figure()
    plt.loglog(delta_x_list1, conv_1[0], label=r"Lax-Friedrichs")
    plt.loglog(delta_x_list2, conv_2[0], label=r"Upwind")
    plt.loglog(delta_x_list3, conv_3[0], label= r"Lax-Wendroff")
    plt.loglog(delta_x_list4, conv_4[0], label= r"MacCormack")
    plt.title("Convergence plot of "+ r'$\rho$' +" in space")
    plt.xlabel(r'$\Delta x$')
    plt.ylabel("Error")
    plt.legend()
    plt.grid()
    plt.savefig("conv_rho_space.pdf")
    plt.show()

    plt.figure()
    plt.loglog(delta_x_list1, conv_1[1], label=r"Lax-Friedrichs")
    plt.loglog(delta_x_list2, conv_2[1], label=r"Upwind")
    plt.loglog(delta_x_list3, conv_3[1], label=r"Lax-Wendroff")
    plt.loglog(delta_x_list4, conv_4[1], label=r"MacCormack")
    plt.title("Convergence plot of " + r'$v$' + " in space")
    plt.xlabel(r'$\Delta x$')
    plt.ylabel("Error")
    plt.legend()
    plt.grid()
    plt.savefig("conv_v_space.pdf")
    plt.show()

