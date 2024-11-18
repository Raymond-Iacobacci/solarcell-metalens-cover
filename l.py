import numpy as np

# Define a square matrix A
A = np.array([[2, 0],
              [0, 3]])

# Compute eigenvalues and eigenvectors
w, v = np.linalg.eig(A)

# Eigenvalues
print("Eigenvalues:", w)  # Output: [2. 3.]

# Eigenvectors
print("Eigenvectors:\n", v)
# Output:
# [[1. 0.]
#  [0. 1.]]

# Verify the relationship A @ v[:, i] == w[i] * v[:, i]
for i in range(len(w)):
    left_side = A @ v[:, i]
    right_side = w[i] * v[:, i]
    print(f"Verification for eigenvalue {w[i]}:")
    print("A @ v[:, {}] = {}".format(i, left_side))
    print("w[{}] * v[:, {}] = {}".format(i, i, right_side))
    assert np.allclose(left_side, right_side), "The eigenvalue equation does not hold."

