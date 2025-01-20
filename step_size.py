import numpy as np
from scipy.stats import norm

'''
The following procedure is used to calculate the step size for the superresolution experiment
'''
Lm=128; Ln=128; #size of the original image
Scale_k=4; # the scale of size

# settings of blur kernel
r = 4
sigma = 1

def scale_matrix(m, n, k):
    s = np.zeros((m * n // (k ** 2), m * n))
    s = s.astype(np.float32)
    #print(s.shape)

    for i in range(0, m, k):
        small_i = i // k
        for j in range(0, n, k):
            small_j = j // k
            i0 = i + k // 2 - 1
            j0 = j + k // 2 - 1
            flag = j0 * m + i0
            row_s = small_j * m // k + small_i
            s[row_s , flag] = 1

    return s

ScM = scale_matrix(Lm, Ln, Scale_k)

size = 2 * r + 1
kx = np.linspace(-r, r, size)
GK = norm.pdf(kx, loc=0, scale=sigma)
GK = GK / GK.sum()

GK = np.outer(GK, GK)
# print(GK);


m=Lm
n=Ln

x = np.zeros((m, n))

A = np.zeros((m * n, m * n))
A = A.astype(np.float32)

Y = np.pad(x, ((r, r), (r, r)), mode='constant')

for i in range(r, m + r):
    for j in range(r, n + r):
        for k in range(i - r, i + r + 1):
            for L in range(j - r, j + r + 1):
                Y[k, L] = GK[k - (i - r), L - (j - r)]

        CY = Y[r:m + r, r:n + r]
        LY = CY.flatten()
        i0, j0 = i - r, j - r
        A[j0 * m + i0, :] = LY

        Y = np.pad(x, ((r, r), (r, r)), mode='constant')

temp3 = A.T @ ScM.T @  ScM @ A
del A
del ScM


def estimate_matrix_2_norm(A, num_iterations=20):
    v = np.random.rand(A.shape[1])

    for _ in range(num_iterations):
        Av = A @ v
        v = Av / np.linalg.norm(Av, ord=2)
        print(v @ A @ v)

    norm_estimate = (v @ A @ v)**0.5
    print(norm_estimate)
    return norm_estimate
temp4 = temp3.T @ temp3
del temp3
estimated_norm = estimate_matrix_2_norm(temp4)
result = 2/estimated_norm
print(estimated_norm)
print(result)

