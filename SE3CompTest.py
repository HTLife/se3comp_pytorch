import torch 

import numpy as np
from SE3Comp import *

batchSize = 2
Tg = torch.zeros(batchSize, 7, 1)
xi_vec = torch.zeros(batchSize, 6, 1)

Tg[0, 0] = 1
Tg[0, 1] = 2
Tg[0, 2] = 3
Tg[0, 3] = 0.6324555
Tg[0, 4] = 0.3162278
Tg[0, 5] = 0.6324555
Tg[0, 6] = 0.3162278

# Tg[0, 0] = 0
# Tg[0, 1] = 0
# Tg[0, 2] = 0
# Tg[0, 3] = 0
# Tg[0, 4] = 0
# Tg[0, 5] = 0
# Tg[0, 6] = 0
xi_vec[0,0] = 3.48075536975048
xi_vec[0,1] = 2.34718513399708
xi_vec[0,2] = 0.874839482491055
xi_vec[0,3] = 1.83448188157597
xi_vec[0,4] = 0.366896492801180
xi_vec[0,5] = 2.20137808316218

# xi_vec[0,0] = 0
# xi_vec[0,1] = 0
# xi_vec[0,2] = 0
# xi_vec[0,3] = 0
# xi_vec[0,4] = 0
# xi_vec[0,5] = 0

model = SE3Comp()


ans = model.forward(Tg, xi_vec)
print('The result should be:')
print('  [3.71428578273958 0.5714288880782 5.14285714642664 0.4383   -0.0000   -0.1992   -0.8765]')
print('  [0 0 0 1 0 0 0]')
print('=======Result=======')
print(ans)
#3.71428578273958 0.5714288880782 5.14285714642664 0.4383   -0.0000   -0.1992   -0.8765
