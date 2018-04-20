import numpy as np
import matplotlib.pyplot as plt

import constants as c
import convergence as conv
import lax_wendroff as lw
import simple_lax as sl
import simple_lax_vectorized as sl_v
import lax_friedrichs as lf
import spatial_convergence as sc
import time_convergence as tc

import upwind as up
import upwind_vectorized as up_v
import upwind_vectorized_v2 as up_v2
import mac_cormack_v2 as mc_v2
import general_convergence as gc

if __name__ == "__main__":

    Master_Flag = {
                    0: 'Lax-Friedrichs',
                    1: 'Upwind',
                    2: 'Lax-Wendroff',
                    3: 'MacCormack',
                    4: 'Time Convergence',
                    5: 'Spatial Convergence',
                    6: '3d plot'



            }[4]        #<-------Write number of the function you want to test.

    if Master_Flag =='Lax-Friedrichs':
        #sc.plot_spatial_convergence_lax(4,lw.solve_lax_wendroff)
        grid_u = lf.solve_lax_friedrichs(c.TIME_POINTS, c.SPACE_POINTS, c.MAX_TIME)
        #lf.plot_lax_friedrichs(c.TIME_POINTS,c.SPACE_POINTS,grid_u)
        lf.plot_lax_friedrichs_3d_rho(c.TIME_POINTS,c.SPACE_POINTS, c.MAX_TIME, grid_u[:, :, 0])
        lf.plot_lax_friedrichs_3d_v(c.TIME_POINTS, c.SPACE_POINTS, c.MAX_TIME, grid_u[:, :, 1])


    elif Master_Flag=='Upwind':
        grid_u = up_v2.solve_upwind(c.TIME_POINTS, c.SPACE_POINTS, c.MAX_TIME)
        up_v2.plot_upwind(c.TIME_POINTS, c.SPACE_POINTS, grid_u[:, :, 0])
        up_v2.plot_upwind_3d_rho(c.TIME_POINTS, c.SPACE_POINTS, c.MAX_TIME, grid_u[:, :, 0])
        up_v2.plot_upwind_3d_v(c.TIME_POINTS, c.SPACE_POINTS, c.MAX_TIME, grid_u[:, :, 1])
        plt.show()

    elif Master_Flag=='Lax-Wendroff':
        grid_u = lw.solve_lax_wendroff(c.TIME_POINTS, c.SPACE_POINTS, c.MAX_TIME)
        lw.plot_lax_wendroff(c.TIME_POINTS, c.SPACE_POINTS, grid_u[:, :, 0])
        lw.plot_lax_wendroff(c.TIME_POINTS, c.SPACE_POINTS, grid_u[:, :, 1])
        lw.plot_lax_wendroff_3d_rho(c.TIME_POINTS,c.SPACE_POINTS,c.MAX_TIME,grid_u[:,:,0])
        lw.plot_lax_wendroff_3d_v(c.TIME_POINTS, c.SPACE_POINTS, c.MAX_TIME,grid_u[:, :, 1])

    elif Master_Flag == 'MacCormack':
        grid_u = mc_v2.solve_mac_cormack(c.TIME_POINTS, c.SPACE_POINTS, c.MAX_TIME)
        mc_v2.plot_mac_cormack(c.TIME_POINTS, c.SPACE_POINTS, c.delta_x,grid_u[:, :, 0])
        mc_v2.plot_mac_cormack(c.TIME_POINTS, c.SPACE_POINTS, c.delta_x,grid_u[:, :, 1])

    elif Master_Flag=='Time Convergence':
        print("Time Convergence")
        tc.plot_time_convergence_2(lf.solve_lax_friedrichs, up_v2.solve_upwind, lw.solve_lax_wendroff, mc_v2.solve_mac_cormack)


    elif Master_Flag=='Spatial Convergence':
        sc.plot_spatial_convergence(lf.solve_lax_friedrichs, up_v2.solve_upwind, lw.solve_lax_wendroff, mc_v2.solve_mac_cormack)

    elif Master_Flag=='General Convergence':
        gc.plot_general_convergence(up_v2.solve_upwind)

