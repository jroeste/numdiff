import numpy as np

M = 8


#MAX_TIME = 60 #seconds
TIME_POINTS = 1000
SPACE_POINTS = 2**(M)
#SPACE_POINTS=1000
L=5000 #meter

delta_t=0.1
MAX_TIME=(TIME_POINTS-1)*delta_t
delta_x=L/(SPACE_POINTS-1)  #meter

V0=33.33 #m/s
RHO_MAX=0.140 #vehicles/m
E=100 #
RHO_0=0.05 #vehicles/m

SIGMA=300 #
MY=0 #
TAU=30 #sec
C=15 #m/s
