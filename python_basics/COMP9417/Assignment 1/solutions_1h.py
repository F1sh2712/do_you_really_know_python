import numpy as np
import matplotlib.pyplot as plt
t_var = np.load("t_var.npy")
y_var = np.load("y_var.npy")
p = y_var.size
# plt.plot(t_var, y_var)
# plt.show()

def create_W(p):
   ## generate W which is a p-2 x p matrix as defined in the question
    W = np.zeros((p-2, p))
    b = np.array([1, -2, 1])
    for i in range(p-2):
        W[i, i:i+3] = b 
    return W 

W = create_W(p)

# Defined given values
lambda_1h = 0.001
alpha = 1.0
matrix = np.identity(p) + 2 * lambda_1h * p * (W.T @ W) # Got from deduction
beta_hat = np.linalg.solve(matrix, y_var) # Solve liner matrix equation ax = b


def loss(beta, y, W, L):
    ## compute loss for a given vector beta for data y, matrix W, 
    # regularization parameter L (lambda)
    # your code here
    residual = y - beta
    loss_val = 0.5/p * (residual @ residual) + L * np.linalg.norm(W @ beta) ** 2
    return loss_val

def gradient(beta, y, W, L):
    gradient_val = (beta - y)/p + 2 * L * (W.T @ (W @ beta))
    return gradient_val

# Get coordinates
def coordinate(beta, y, L, j, p):
    # Due to the equation we get, need to consider that j-2, j-1, j+1, j+2 are out of range
    if j - 2 >= 0:
        beta_j_minus_2 = beta[j - 2]
    else:
        beta_j_minus_2 = 0
    
    if j - 1 >= 0:
        beta_j_minus_1 = beta[j - 1]
    else:
        beta_j_minus_1 = 0

    if j + 1 < p:
        beta_j_plus_1 = beta[j + 1]
    else:
        beta_j_plus_1 = 0

    if j + 2 < p:
        beta_j_plus_2 = beta[j + 2]
    else:
        beta_j_plus_2 = 0

    coordinate_val = ((y[j] / p) + L * (2*beta_j_minus_2 - 
                                        8*beta_j_minus_1 - 
                                        8*beta_j_plus_1 + 
                                        2*beta_j_plus_2)) / (1 / p + 12 * L)
    
    return coordinate_val

# Calculate L(beta_hat)
L_beta_hat = loss(beta_hat, y_var, W, lambda_1h)

# Ends after 1000 steps
GD_beta = np.ones(p) # Starting point beta0
coordinate_beta = np.ones(p)
GD_list = [] # Values of delta for GD schema
coordinate_list = [] # Values for coordinate schema

for k in range(1000):
    grad = gradient(GD_beta, y_var, W, lambda_1h)
    GD_beta = GD_beta - alpha * grad
    GD_delta = loss(GD_beta, y_var, W, lambda_1h) - L_beta_hat
    GD_list.append(GD_delta)
 
    j = k % p # Cycle through coordinate level updates
    coordinate_beta[j] = coordinate(coordinate_beta, y_var, lambda_1h, j, p)
    coordinate_delta = loss(coordinate_beta, y_var, W, lambda_1h) - L_beta_hat
    coordinate_list.append(coordinate_delta)

# Generate plot
plt.plot(coordinate_list, color='blue', label='Coordinate')
plt.plot(GD_list, color='green', label="GD")
plt.title('k vs ∆(k)')
plt.legend(loc='best')
plt.xlabel('Iteration(k)')
plt.ylabel('∆ L')

## your code here
plt.tight_layout()
plt.show()