import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

a = 40.0
b = 3


@njit
def pd(d):
    return 1 / (1+np.power(d/a, b))

n_trails = 1000000

@njit
def generate():
    count = np.zeros(n_trails, np.int32)
    for i in prange(n_trails):
        while (np.random.rand() < pd(count[i])):
            count[i] += 1
    return count

count = generate()
print(count.mean())
print(count.max())
print(count.min())
counts = plt.hist(count, bins=np.arange(count.max() + 1), density=True)
plt.vlines(count.mean(), 0, counts[0].max(), label="mean", colors='r')
plt.legend()
plt.show()
