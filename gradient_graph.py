import numpy as np
import matplotlib.pyplot as plt

# Definición de la función y sus derivadas parciales
def f(x, y):
    return np.sin(x) + np.sin(y) + 0.5 * x**2 - 0.5 * y**2

def df_dx(x):
    return np.cos(x) + x

def df_dy(y):
    return np.cos(y) - y

# Función para graficar el progreso del algoritmo
def plot_progress(x_vals, y_vals, f_vals):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(f_vals)), f_vals, label='Valor de la función')
    plt.scatter(range(len(f_vals)), f_vals, c='red')
    plt.xlabel('Iteración')
    plt.ylabel('Valor de la función')
    plt.title('Progreso del descenso del gradiente')
    plt.legend()
    plt.grid(True)
    plt.show()

# Inicialización de parámetros
x = 0
y = 0
alpha = 0.1  # Tasa de aprendizaje
max_iter = 100  # Número máximo de iteraciones
tolerance = 1e-6  # Tolerancia para la convergencia

# Listas para almacenar los valores en cada iteración
x_vals = [x]
y_vals = [y]
f_vals = [f(x, y)]

# Iteraciones del descenso del gradiente
for i in range(max_iter):
    # Calculamos las derivadas parciales en el punto actual
    df_dx_val = df_dx(x)
    df_dy_val = df_dy(y)
    
    # Actualizamos los parámetros
    new_x = x - alpha * df_dx_val
    new_y = y - alpha * df_dy_val
    
    # Calculamos la diferencia en el valor de la función entre iteraciones consecutivas
    diff = np.abs(f(new_x, new_y) - f(x, y))
    
    # Actualizamos los parámetros para la próxima iteración
    x = new_x
    y = new_y
    
    # Almacenamos los valores en cada iteración
    x_vals.append(x)
    y_vals.append(y)
    f_vals.append(f(x, y))
    
    # Verificamos la condición de parada
    if diff < tolerance:
        print(f"Convergencia alcanzada en la iteración {i+1}")
        break

# Evaluación del resultado
min_value = f(x, y)
print(f"El mínimo local encontrado es: ({x}, {y}) con un valor de {min_value}")

# Graficamos el progreso del descenso del gradiente
plot_progress(x_vals, y_vals, f_vals)
