import numpy as np
import matplotlib.pyplot as plt

import constants as c
import simple_lax_vectorized as sl_v
import upwind_vectorized as up_v


def spatial_convergence_vec(solver, T, X, delta_t, delta_x):
    convergence_list = np.zeros((2, c.M + 1))
    u_exact = solver(T, X, delta_t, delta_x)
    exact_list = u_exact[-1]
    step_length_list = np.zeros(c.M + 1)
    print(exact_list)

    x_list = np.linspace(-c.L / 2, c.L / 2, len(exact_list))
    plt.plot(x_list,exact_list[:,0])
    plt.show()

    for j in range(c.M):
        print("inside for loop")
        x_points = 2 ** (j + 1)
        new_exact_list = np.zeros((x_points,2))

        ratio = (len(exact_list[0]) - 1) / (x_points - 1)
        for h in range(x_points):
            new_exact_list[h] = exact_list[int(h * ratio)]


        delta_x = c.L / (2*(x_points - 1))
        step_length_list[j - 1] = delta_x
        u = solver(c.TIME_POINTS, x_points, delta_t, delta_x)
        j_list=u[-1]
        #j_list = np.array([u[:, :, 0][-1], [u[:, :, 1][-1]]])
        #rho_j=u[:, :, 0][-1]
        #v_j=u[:, :, 1][-1]

        convergence_list[0][j - 1] = np.sqrt(delta_x * delta_t) * np.linalg.norm(new_exact_list[:,0] - j_list[:,0], 2)
        convergence_list[1][j - 1] = np.sqrt(delta_x * delta_t) * np.linalg.norm(new_exact_list[:,1] - j_list[:,1], 2)

        x_list = np.linspace(-c.L/2, c.L/2, len(new_exact_list[:,0]))
        x_list2 = np.linspace(-c.L/2, c.L/2, len(j_list[:,0]))
        plt.plot(x_list,new_exact_list[:,0],label="exact")
        plt.plot(x_list2,j_list[:,0],label="not exact")
        plt.legend()
        plt.show()

    return convergence_list, step_length_list

def plot_convergence():
    conv_list,step_length_list=spatial_convergence_vec(sl_v.solve_simple_lax, c.TIME_POINTS, c.SPACE_POINTS,c.delta_t,c.delta_x)
    print(conv_list[0])
    print(conv_list[1])
    plt.loglog(step_length_list,conv_list[0],label='rho')
    plt.loglog(step_length_list,conv_list[1],label='v')
    plt.legend()
    plt.show()

plot_convergence()