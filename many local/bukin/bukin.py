import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits import mplot3d

def f(x1, x2):
    ab = np.fabs(x2 - 0.01 * x1 * x1)
    a = 100 * np.sqrt(ab) + 0.01 * np.fabs(x1 + 10)
    return a

def gradient(x):
    epsilon = 1e-8
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += epsilon
        grad[i] = (f(x_plus[0], x_plus[1]) - f(x[0], x[1])) / epsilon
    return grad

# Adam Optimizer Implementation
def adam_optimizer(f_grad, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, steps=200):
    x = np.random.uniform(-15, 15, size=(2,)) 
    m = np.zeros_like(x)  
    v = np.zeros_like(x)  
    path = [x.copy()]  

    for t in range(1, steps + 1):
        grad = f_grad(x)

        
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2

       
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)

     
        x -= lr * m_hat / (np.sqrt(v_hat) + epsilon) 
        path.append(x.copy())

    return np.array(path), f(x[0], x[1])

# Run the optimizer for 30 times
runs = 30
results = []
paths = []

for _ in range(runs):
    path, value = adam_optimizer(gradient, steps=200)
    results.append(value)
    paths.append(path)


best = np.min(results)  # Best value (global minimum)
worst = np.max(results)  # Worst value (highest value found during optimization)
average = np.mean(results)  # Average value


df = pd.DataFrame({
    "Run": [i + 1 for i in range(runs)],
    "Final X1": [path[-1, 0] for path in paths],
    "Final X2": [path[-1, 1] for path in paths],
    "Function Value": results,
})


print(df)


print(f"\nBest value: {best}")
print(f"Worst value: {worst}")
print(f"Average value: {average}")

# Visualizing the Bukin Function N.6
x1 = np.linspace(-15, -5, 100)
x2 = np.linspace(-3, 3, 100)
X1, X2 = np.meshgrid(x1, x2)
Z = f(X1, X2)


fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Z, cmap='viridis', edgecolor='none', alpha=0.7)
ax.set_title('Bukin Function N.6 (3D Surface)')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
ax.view_init(elev=40, azim=40)

# 2D Contour Plot with Optimization Paths
fig, ax = plt.subplots(figsize=(12, 8))
contour = ax.contourf(X1, X2, Z, levels=50, cmap='viridis')
plt.colorbar(contour)
for path in paths:
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], color='red', alpha=0.6)
ax.set_title('Bukin Function N.6 with Optimization Paths')
ax.set_xlabel('x1')
ax.set_ylabel('x2')

# Show the plots
plt.show()
