import numpy as np

M = 12

MAX_TIME = 60 #seconds
TIME_POINTS = 50
#SPACE_POINTS = 2**(M)
SPACE_POINTS=10
L=5000 #meter


delta_t=0.1
MAX_TIME=(TIME_POINTS-1)*delta_t
delta_x=L/(SPACE_POINTS-1)  #meter

V0=33.33
RHO_MAX=0.140
E=100
RHO_0=0.05

SIGMA=300
MY=0
TAU=30
C=15
