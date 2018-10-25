import numpy as np
from scipy.optimize import linprog

a = np.array([[1,2],[3,4]])
b= np.array([[1,2],[3,3]])
c = (a.T[0]*b.T[0])
print(c)