from sklearn.tree import DecisionTreeRegressor
import numpy as np 
import matplotlib.pyplot as plt

# true function
def f(x):
    t1 = np.sqrt(x * (1-x))
    t2 = (2.1 * np.pi) / (x + 0.05)
    t3 = np.sin(t2)
    return t1*t3

def f_sampler(f, n=100, sigma=0.05):    
    # sample points from function f with Gaussian noise (0,sigma**2)
    xvals = np.random.uniform(low=0, high=1, size=n)
    yvals = f(xvals) + sigma * np.random.normal(0,1,size=n)

    return xvals, yvals

def gradient_combination(X, y, T, is_adaptive):
    fixed_alpha = 0.1
    depth = 1
    n = X.shape[0]
    f_t = np.zeros(n)
    models_list = []

    for t in range(T):
        # GC1, compute the pseudo-residual at iteration t
        residual = y - f_t

        # GC2, construct new pseudo data set D
        model = DecisionTreeRegressor(max_depth=depth)
        # GC3, fit a model to D, a base learner h
        model.fit(X, residual)
        h_t = model.predict(X)

        # GC4, choose a step size
        if is_adaptive:
            # From the result of question 2b
            numerator = np.dot(residual, h_t)
            denominator = np.dot(h_t, h_t)

            if denominator != 0:
                alpha_t = numerator / denominator
            else:
                alpha_t = 0
        else:
            # Use fixed step size
            alpha_t = fixed_alpha

        # GC5, update f_t
        f_t = f_t + alpha_t * h_t
        models_list.append((alpha_t, model))

    return models_list

# (III) return f_T
def predicted_result(xx, model_list):
    predicted_y = np.zeros(xx.shape[0])

    for alpha_t, model in model_list:
        predicted_y = predicted_y + alpha_t * model.predict(xx)

    return predicted_y

np.random.seed(123)
X, y = f_sampler(f, 160, sigma=0.2)
X = X.reshape(-1,1)

# Draw plots
fig_adpat, axs_adapt = plt.subplots(5, 2, figsize=(12, 26))
fig_fixed, axs_fixed = plt.subplots(5, 2, figsize=(12, 26))
base_learners = 5
plot_index = 0

while base_learners <= 50:
    row = plot_index //2 
    col = plot_index % 2

    models_list_adaptive = gradient_combination(X, y, base_learners, is_adaptive=True)
    models_list_fixed = gradient_combination(X, y, base_learners, is_adaptive=False)

    xx = np.linspace(0,1,1000).reshape(-1, 1)

    pred_result_adaptive = predicted_result(xx, models_list_adaptive)
    pred_result_fixed = predicted_result(xx, models_list_fixed)

    axs_adapt[row, col].plot(xx, f(xx), alpha=0.5, color='red', label='truth')
    axs_adapt[row, col].scatter(X, y, marker='x', alpha=0.5, color='blue',  label='observed')
    axs_adapt[row, col].plot(xx, pred_result_adaptive, color='green', label=f'T={base_learners}')
    # axs_adapt[row, col].set_title(f'{base_learners} base learners', fontsize=10)
    axs_adapt[row, col].legend(fontsize=8)

    axs_fixed[row, col].plot(xx, f(xx), alpha=0.5, color='red', label='truth')
    axs_fixed[row, col].scatter(X, y, marker='x', alpha=0.5, color='blue',  label='observed')
    axs_fixed[row, col].plot(xx, pred_result_fixed, color='green', label=f'T={base_learners}')
    # axs_fixed[row, col].set_title(f'{base_learners} base learners', fontsize=10)
    axs_fixed[row, col].legend(fontsize=8)

    base_learners = base_learners + 5
    plot_index = plot_index + 1

fig_adpat.tight_layout()    
fig_adpat.suptitle(f'GD with adaptive step size', fontsize=15)
fig_adpat.subplots_adjust(top=0.96)
fig_adpat.subplots_adjust(hspace=0.1)
fig_adpat.savefig("GD_adaptive.png", dpi=400)

fig_fixed.tight_layout()
fig_fixed.suptitle(f'GD with fixed step size', fontsize=15)
fig_fixed.subplots_adjust(top=0.96)
fig_fixed.subplots_adjust(hspace=0.1)
fig_fixed.savefig("GD_fixed.png", dpi=400)

plt.show()
# fig = plt.figure(figsize=(7,7))
# dt = DecisionTreeRegressor(max_depth=2).fit(X,y)
# xx = np.linspace(0,1,1000)
# plt.plot(xx, f(xx), alpha=0.5, color='red', label='truth')
# plt.scatter(X,y, marker='x', color='blue', label='observed')
# plt.plot(xx, dt.predict(xx.reshape(-1,1)), color='green', label='dt')
# plt.legend()
# #plt.savefig("example.png", dpi=400)        
# plt.show()