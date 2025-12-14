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
lambda_1e = 0.001
matrix = np.identity(p) + 2 * lambda_1e * p * (W.T @ W) # Got from deduction
beta_hat = np.linalg.solve(matrix, y_var) # Solve liner matrix equation ax = b
alpha_list = [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.6, 1.2, 2]

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

# Calculate L(beta_hat)
L_beta_hat = loss(beta_hat, y_var, W, lambda_1e)

# Gnerate plots with different alphas
fig, axs = plt.subplots(3, 3) # 3 * 3 grid

for i in range(len(alpha_list)):
    alpha = alpha_list[i]
    beta = np.ones(p) # Starting point beta0
    delta_list = [] # Values of delta k and plot later

    for k in range(1000):
        grad = gradient(beta, y_var, W, lambda_1e)
        beta = beta - alpha * grad
        delta = loss(beta, y_var, W, lambda_1e) - L_beta_hat
        delta_list.append(delta)

    # Plots
    ax = axs[i // 3, i % 3] # Define (col, row) cordinates for each figure
    ax.plot(delta_list)
    ax.set_title(f"alpha = {alpha}")

## your code here
plt.tight_layout()
plt.show()