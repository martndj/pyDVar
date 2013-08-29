import numpy as np
from dataAssLib import *



csp=np.array([45, 34+5j, 24.+2j, 24.-2j,34-5j])
rsp=c2r(csp)
print(np.dot(r2c(rsp), csp))
print(np.dot(rsp, r2c_Adj(csp)))

