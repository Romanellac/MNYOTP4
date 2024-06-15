import numpy as np
import matplotlib.pyplot as plt

# Parámetros
n = 5
d = 100

# Generación de matriz A y vector b aleatorios
np.random.seed(0)  # Para reproducibilidad
A = np.random.randn(n, d)
b = np.random.randn(n)

# Funciones de costo
def F(x):
    return 0.5 * np.linalg.norm(A @ x - b)**2

def F2(x, delta2): 
    return 0.5 * np.linalg.norm(A @ x - b)**2 + 0.5 * delta2 * np.linalg.norm(x)**2

# Gradientes
def grad_F(x): # *** chequear si es así el gradiente,
    return A.T @ (A @ x - b)

def grad_F2(x, delta2): # *** chequear si es así el gradiente, sobretodo lo de delta
    return A.T @ (A @ x - b) + delta2 * x

# Máximo valor propio de A^T A *** es así?
lambda_max = np.linalg.norm(A.T @ A, 2)

# Valor máximo singular de A *** es así?
sigma_max = np.linalg.norm(A, 2)

# Parámetro de regularización
delta2 = 10**-2 * sigma_max

# Tamaño de paso
s = 1 / lambda_max

# Inicialización aleatoria del vector x *** lo inicializo con cualquiera o desde el 00? no sé
x0 = np.random.randn(d)

# Gradiente descendente para minimizar F y F2 (Si delta2 es none usa solo F, sino usa F2! la versión con delta2)
def gradient_descent(grad_F, x0, s, iterations, delta2=None):
    x = x0
    history = [x0]
    for _ in range(iterations):
        if delta2 is not None:
            grad = grad_F(x, delta2)
        else:
            grad = grad_F(x)
        x = x - s * grad
        history.append(x)
    return np.array(history)

# Ejecutar gradiente descendente
iterations = 1000
# Me fijo los casos con y sin delta:
history_F = gradient_descent(grad_F, x0, s, iterations)
history_F2 = gradient_descent(grad_F2, x0, s, iterations, delta2=delta2)


# Solución usando SVD
'''
matrix = U@S@Vt
pseudoinv = np.linalg.pinv(matrix)
x = np.dot(pseudoinv, y)
'''
U, Sigma, Vt = np.linalg.svd(A, full_matrices=False)
Sigma_inv = np.diag(1 / Sigma)
x_svd = Vt.T @ Sigma_inv @ U.T @ b

# Solución con regularización usando SVD
Sigma2_inv = np.diag(1 / (Sigma**2 + delta2))
x_svd_reg = Vt.T @ Sigma2_inv @ U.T @ b

# Evaluar las soluciones
F_svd = F(x_svd)
F2_svd_reg = F2(x_svd_reg, delta2)
F_history = [F(x) for x in history_F]
F2_history = [F2(x, delta2) for x in history_F2]

# Gráficas de la evolución del costo
plt.figure(figsize=(12, 6))
plt.plot(F_history, label='F(x) sin regularización')
plt.plot(F2_history, label='F2(x) con regularización')
plt.axhline(F_svd, color='r', linestyle='--', label='F(x) con SVD')
plt.axhline(F2_svd_reg, color='g', linestyle='--', label='F2(x) con SVD')
plt.yscale('log')
plt.xlabel('Iteraciones')
plt.ylabel('Costo')
plt.legend()
plt.title('Evolución del Costo en Gradiente Descendente')
plt.show()

# Comparación de resultados
print(f'Solución con gradiente descendente F(x): {F_history[-1]:.4e}')
print(f'Solución con SVD F(x): {F_svd:.4e}')
print(f'Solución con gradiente descendente F2(x): {F2_history[-1]:.4e}')
print(f'Solución con SVD F2(x): {F2_svd_reg:.4e}')

# *** GRAFICAMOS HACIA DÓNDE VA EL MÍNIMO O LOS X DE POR SI? O SEA GRAFICAMOS F(X) O X??? X NO PORQUE ES DE ALTA DIMENSIÓN.

'''
# Evaluar la evolución de la norma de x en las iteraciones
norm_history_F = [np.linalg.norm(x) for x in history_F]
norm_history_F2 = [np.linalg.norm(x) for x in history_F2]

# Gráficas de la evolución de la norma de x
plt.figure(figsize=(12, 6))
plt.plot(norm_history_F, label='F(x) sin regularización')
plt.plot(norm_history_F2, label='F2(x) con regularización')
plt.axhline(np.linalg.norm(x_svd), color='r', linestyle='--', label='Norma de x con SVD')
plt.axhline(np.linalg.norm(x_svd_reg), color='g', linestyle='--', label='Norma de x con SVD y regularización')
plt.yscale('log')
plt.xlabel('Iteraciones')
plt.ylabel('Norma de x')
plt.legend()
plt.title('Evolución de la Norma de x en Gradiente Descendente')
plt.show()

# Comparación de resultados
print(f'Norma de x con gradiente descendente F(x): {norm_history_F[-1]:.4e}')
print(f'Norma de x con SVD F(x): {np.linalg.norm(x_svd):.4e}')
print(f'Norma de x con gradiente descendente F2(x): {norm_history_F2[-1]:.4e}')
print(f'Norma de x con SVD y regularización F2(x): {np.linalg.norm(x_svd_reg):.4e}')

'''

