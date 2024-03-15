import numpy as np

# Definición de la función y sus derivadas parciales
def f(x, y):
    return np.sin(x) + np.sin(y) + 0.5 * x**2 - 0.5 * y**2

def df_dx(x):
    return np.cos(x) + x

def df_dy(y):
    return np.cos(y) - y

# Inicialización de parámetros
x = 0
y = 0
alpha = 0.1  # Tasa de aprendizaje
max_iter = 100  # Número máximo de iteraciones
tolerance = 1e-6  # Tolerancia para la convergencia

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
    
    # Verificamos la condición de parada
    if diff < tolerance:
        print(f"Convergencia alcanzada en la iteración {i+1}")
        break

# Evaluación del resultado
min_value = f(x, y)
print(f"El mínimo local encontrado es: ({x}, {y}) con un valor de {min_value}")
