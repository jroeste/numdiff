import numpy as np

a = np.zeros(2)
a[0] = 3
a[1] = 4
b = np.zeros(2)
b[:] = 1,10
c = np.zeros(2)
c[0] = 2
c[1] = 1
print(a,b,c)
k = a - b+c
print(k)
