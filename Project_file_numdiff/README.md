# numdiff
This is numerical simulation of the traffic equations using Lax-Friedrichs, Upwind, Lax-Wendroff and MacCormack schemes.

## Running code
Run all files through main.py using the masterflag function.
0 to 3 runs individual plots for the different schemes.
4 run the time convergence plot for all equations with step size from m=2^4 to n=2^12 and reference solution at 2^(n+1).
5 run spatial convergence plot for all equations. Here the step size in space and time, as well as simulation time can be manipulated in spatial_convergence.py.
6 plots 3D plots for Lax-Friedrichs scheme.

Constants can be changed in constants.py, but for the convergce-files, some of the constants are in the files.

