import numpy as np
import matplotlib.pyplot as plt

def holder_function(x):
    x1, x2 = x
    return -np.abs(np.exp(np.abs(1 - np.sqrt(x1**2 + x2**2) / np.pi)) * np.sin(x1) * np.cos(x2))

def holder_gradient(x):
    epsilon = 1e-8
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += epsilon
        grad[i] = (holder_function(x_plus) - holder_function(x)) / epsilon
    return grad

# Adam optimizer
def adam_optimizer(f_grad, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, steps=200):
    x = np.random.uniform(-10, 10, size=(2,))  # Random starting point
    m = np.zeros_like(x)  # First moment vector
    v = np.zeros_like(x)  # Second moment vector
    path = [x.copy()]  # Record optimization path

    for t in range(1, steps + 1):
        grad = f_grad(x)

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2

        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)

        x -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
        path.append(x.copy())

    return np.array(path), holder_function(x)

# Run Adam optimizer 30 times and collect results
runs = 30
results = []
paths = []

for _ in range(runs):
    path, value = adam_optimizer(holder_gradient, steps=200)
    results.append(value)
    paths.append(path)

# Compute statistics
best = np.min(results)
worst = np.max(results)
average = np.mean(results)


x1 = np.linspace(-10, 10, 100)
x2 = np.linspace(-10, 10, 100)
x1, x2 = np.meshgrid(x1, x2)
z = -np.abs(np.exp(np.abs(1 - np.sqrt(x1**2 + x2**2) / np.pi)) * np.sin(x1) * np.cos(x2))

# 3D Surface Plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1, x2, z, cmap='viridis', edgecolor='none', alpha=0.8)
ax.set_title("Holder Function (3D Surface)")
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
ax.set_title("holder Function with Optimization Paths")
ax.set_xlabel("X1")
ax.set_ylabel("X2")

# Display results
print(f"Best value: {best}")
print(f"Worst value: {worst}")
print(f"Average value: {average}")

plt.show()
