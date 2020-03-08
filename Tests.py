import numpy as np
diffs = np.array([3,4,12,5])
tmp = diffs ** 2
mod_diffs = np.sqrt(np.repeat([sum(tmp[i:i + 2]) for i in range(0, len(tmp), 2)], 2))
print(mod_diffs)