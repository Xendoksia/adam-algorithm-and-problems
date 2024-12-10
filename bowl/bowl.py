import matplotlib.pyplot as plt
import numpy as np
import time  
import pandas as pd  

def perm_function(x1, x2):
    return (11 * (x1 - 1) + 12 * (x2 - 0.5))**2 + (11 * (x1**2 - 1) + 12 * (x2**2 - 0.25))**2


def perm_gradient(x):
    x1, x2 = x
    grad_x1 = 2 * 11 * (11 * (x1 - 1) + 12 * (x2 - 0.5)) + 4 * 11 * x1 * (11 * (x1**2 - 1) + 12 * (x2**2 - 0.25))
    grad_x2 = 2 * 12 * (11 * (x1 - 1) + 12 * (x2 - 0.5)) + 4 * 12 * x2 * (11 * (x1**2 - 1) + 12 * (x2**2 - 0.25))
    return np.array([grad_x1, grad_x2])

# Adam optimizer
def adam_optimizer(f_grad, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, steps=100):
    x = np.random.uniform(-2, 2, size=(2,))  #
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

    return np.array(path), perm_function(*x)

runs = 30
results = []
times = [] 
final_values = []  

for _ in range(runs):
    start_time = time.time()  
    path, value = adam_optimizer(perm_gradient, steps=200)
    end_time = time.time()  
    results.append(value)
    final_values.append(path[-1])  #
    times.append(end_time - start_time)  

best = np.min(results)
worst = np.max(results)
average = np.mean(results)
average_time = np.mean(times)  

# Create a table of results using pandas
df = pd.DataFrame({
    'Run': np.arange(1, runs + 1),
    'Final X1': [fv[0] for fv in final_values],
    'Final X2': [fv[1] for fv in final_values],
    'Function Value': results,
    'Time (seconds)': times
})

# Display the table
print(df)

# Visualize the bowl function
x1 = np.linspace(-2, 2, 100)
x2 = np.linspace(-2, 2, 100)
x1, x2 = np.meshgrid(x1, x2)
z = perm_function(x1, x2)


fig1 = plt.figure(figsize=(10, 10))
ax1 = fig1.add_subplot(111, projection='3d')
ax1.contour3D(x1, x2, z, levels=50, cmap='viridis')
ax1.set_title("Perm Function (3D Surface)")
ax1.view_init(40, 40)
ax1.set_xlabel("X1")
ax1.set_ylabel("X2")
ax1.set_zlabel("Z")
plt.show()


fig2 = plt.figure(figsize=(10, 10))
ax2 = fig2.add_subplot(111)
cp = ax2.contourf(x1, x2, z, levels=50, cmap='viridis')
fig2.colorbar(cp, ax=ax2)
ax2.set_title("Perm Function (2D Contour)")
ax2.set_xlabel("X1")
ax2.set_ylabel("X2")
plt.show()

# Print statistics
print(f"Best value: {best}")
print(f"Worst value: {worst}")
print(f"Average value: {average}")
print(f"Average solution time: {average_time:.4f} seconds")
