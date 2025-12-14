import numpy as np

# Defined given values
A = np.array([
    [3, 2, 0, -1],
    [-1, 3, 0, 2],
    [0, -4, -2, 7]
], dtype = float)

b = np.array([3, 1, -4], dtype = float)

x = np.array([1, 1, 1, 1], dtype = float)

alpha = 0.01
gamma = 2.0

# Return gradient of f(x)
def gradient(x):
    return A.T @ (A @ x - b) + gamma * x

# Gradient descent, return result of x and k by iterations
def grad_desc(X):
    k = 0
    x_list = [x.copy()]

    while True:
        grad = gradient(x_list[-1])
        # Stops when gradient is smaller than 0.001
        if np.linalg.norm(grad) < 0.001:
            break

        new_x = x_list[-1] - alpha * grad
        x_list.append(new_x)
        k = k + 1

    return np.array(x_list), k

# Print result
x_result, iteration = grad_desc(x)

print(f"There are {iteration} iterations before stop\n")

if iteration >= 5:
    print("First 5 steps (k = 5 inclusive):")
    for k in range(6): # k = 5 inclusive
        print(f"k = {k}, x{k} = {np.round(x_result[k], 4)}")

    print("\nLast 5 steps:")
    for k in range(len(x_result) - 5, len(x_result)):
        print(f"k = {k}, x{k} = {np.round(x_result[k], 4)}")

if iteration < 5:
    for k in range(len(x_result)):
        print(f"k = {k}, x{k} = {np.round(x_result[k], 4)}")