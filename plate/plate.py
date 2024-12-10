import numpy as np
import matplotlib.pyplot as plt
import time  


def matyas_function(x, y):
    return 0.26 * ((x**2) + (y**2)) - (0.48 * x * y)


def matyas_gradient(x):
    x1, x2 = x
    grad_x1 = 0.52 * x1 - 0.48 * x2
    grad_x2 = 0.52 * x2 - 0.48 * x1
    return np.array([grad_x1, grad_x2])

# Adam optimizer
def adam_optimizer(f_grad, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, steps=200):
    x = np.random.uniform(-10, 10, size=(2,))  # Random starting point
    m = np.zeros_like(x)  # First moment vector
    v = np.zeros_like(x)  # Second moment vector
    path = [x.copy()]  # Record optimization path

    for t in range(1, steps + 1):
        grad = f_grad(x)

        # Update biased first and second moments
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2

        # Compute bias-corrected moments
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)

        # Update parameters
        x -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
        path.append(x.copy())

    return np.array(path), matyas_function(*x)

# Run Adam optimizer 30 times and collect results
runs = 30
results = []
times = []
paths = []

for i in range(runs):
    start_time = time.time()  # Start timing
    path, value = adam_optimizer(matyas_gradient, steps=200)
    end_time = time.time()  # End timing
    
    results.append(value)
    times.append(end_time - start_time)
    paths.append(path)

best = np.min(results)
worst = np.max(results)
average = np.mean(results)
average_time = np.mean(times)


import pandas as pd

data = {
    'Run': range(1, runs + 1),
    'Final X1': [path[-1][0] for path in paths],
    'Final X2': [path[-1][1] for path in paths],
    'Function Value': results,
    'Time (seconds)': times
}

df = pd.DataFrame(data)

print(df)

# Visualize the Matyas function
x1 = np.linspace(-10, 10, 100)
x2 = np.linspace(-10, 10, 100)
x1, x2 = np.meshgrid(x1, x2)
z = matyas_function(x1, x2)

# 3D Surface Plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1, x2, z, cmap='viridis', edgecolor='none', alpha=0.8)
ax.set_title("Matyas Function (3D Surface)")
ax.view_init(40, 40)
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("f(x1, x2)")

# 2D Contour Plot with Optimization Paths
fig, ax = plt.subplots(figsize=(10, 8))
contour = ax.contourf(x1, x2, z, levels=50, cmap='viridis')
plt.colorbar(contour)
for path in paths:
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], color="red", alpha=0.6)
ax.set_title("Matyas Function with Optimization Paths")
ax.set_xlabel("X1")
ax.set_ylabel("X2")

# Display results
print(f"Best value: {best}")
print(f"Worst value: {worst}")
print(f"Average value: {average}")
print(f"Average solution time: {average_time:.4f} seconds")

plt.show()
