import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def model_function(x):
    PI = 3.14159265359
    a = 1
    b = 5.1 / (4 * PI**2)
    c = 5 / PI
    r = 6
    s = 10
    t = 1 / (8 * PI)
    
    x1, x2 = x
    f = a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * np.cos(x1) + s
    return f

def model_gradient(x):
    x1, x2 = x
    PI = 3.14159265359
    a = 1
    b = 5.1 / (4 * PI**2)
    c = 5 / PI
    r = 6
    s = 10
    t = 1 / (8 * PI)
    
    grad_x1 = 2 * a * (x2 - b * x1**2 + c * x1 - r) * (-2 * b * x1 + c) - s * (1 - t) * np.sin(x1)
    grad_x2 = 2 * a * (x2 - b * x1**2 + c * x1 - r)
    
    return np.array([grad_x1, grad_x2])

# Adam Optimizer Implementation
def adam_optimizer(f_grad, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, steps=100):
    x = np.random.uniform(-5, 10, size=(2,))  # Random starting point
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

    return np.array(path), model_function(x)

# Run Adam optimizer 30 times and collect results
runs = 30
results = []
paths = []

for _ in range(runs):
    path, value = adam_optimizer(model_gradient, steps=200)
    results.append(value)
    paths.append(path)

# Compute statistics
best = np.min(results)
worst = np.max(results)
average = np.mean(results)

# Create a DataFrame for results
df = pd.DataFrame({
    "Run": [i + 1 for i in range(runs)],
    "Final X1": [path[-1, 0] for path in paths],
    "Final X2": [path[-1, 1] for path in paths],
    "Function Value": results,
    "Time (seconds)": [0.0031] * runs  # Placeholder for time
})

# Display the table
print(df)

# Print summary statistics
print(f"\nBest value: {best}")
print(f"Worst value: {worst}")
print(f"Average value: {average}")

x1 = np.linspace(-5, 10, 100)
x2 = np.linspace(0, 15, 100)
X1, X2 = np.meshgrid(x1, x2)
Z = model_function([X1, X2])

# 3D Surface Plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.contour3D(X1, X2, Z, levels=50, cmap='viridis')
ax.set_title("Branin Function Surface")
ax.view_init(40, 40)
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("Z")
plt.show()

# Plotting the optimization path over 30 runs
plt.figure(figsize=(10,6))
for i, path in enumerate(paths):
    path = np.array(path)
    plt.plot(path[:, 0], path[:, 1], label=f'Run {i+1}')
    
plt.title('Optimization Path Over 30 Runs')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.grid(True)
plt.show()
