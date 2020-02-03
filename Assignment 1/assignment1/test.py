import numpy as np
import math
import time

a = np.array([(1, 0), (2, 0)])

print(a[np.argmin(a[:, 0])][1])