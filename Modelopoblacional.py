import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt

# Datos históricos proporcionados
years_full = np.array([1770, 1778, 1825, 1835, 1843, 1851, 1864, 1870, 1905, 1912, 1918, 1928, 1938, 1951, 1964, 1973, 1985, 1993, 2005, 2018])
population_full = np.array([507209, 747641, 2583799, 1687109, 1932279, 2243730, 2441300, 2681637, 4533777, 5472604, 5855077, 7851110, 8697041, 11548172, 17484510, 20666920, 27853432, 33109839, 41468384, 48258494])

# Datos para el ajuste inicial (1912, 1938 y 1993)
years_initial = np.array([1912, 1938, 1993])
population_initial = np.array([5472604, 8697041, 33109839])

# Definición del modelo logístico
def logistic_model(t, r, K):
    P0 = 5472604  # Fijamos la población inicial
    return K / (1 + ((K - P0) / P0) * np.exp(-r * (t - years_initial[0])))

# Valores iniciales aproximados para r y K
initial_values = [0.03, 100000000]  # [r, K]

# Ajuste del modelo logístico a los datos iniciales
params, cov = curve_fit(logistic_model, years_initial, population_initial, p0=initial_values)
r, K = params

# Mostrar los parámetros ajustados
print(f"Tasa de crecimiento (r): {r}")
print(f"Capacidad de carga (K): {K}")

# Parte 2: Comparar la población real con la predicha por el modelo

# Población predicha por el modelo
predicted_population = logistic_model(years_full, r, K)

# Calcular error y error porcentual
error = population_full - predicted_population
percentage_error = (error / population_full) * 100

# Crear una tabla comparativa
comparison_table = pd.DataFrame({
    'Año': years_full,
    'Población Real': population_full,
    'Población Predicha': predicted_population.astype(int),
    'Error': error.astype(int),
    'Error Porcentual (%)': percentage_error
})

# Mostrar la tabla
print(comparison_table)

# Parte 3: Proyección de la población para el año 2024
year_2024 = 2024
predicted_population_2024 = logistic_model(year_2024, r, K)
print(f"Población proyectada para 2024: {int(predicted_population_2024)}")

# Visualización
plt.figure(figsize=(10, 6))e
plt.scatter(years_full, population_full, label='Datos históricos', color='red')
plt.plot(np.arange(1700, 2030), logistic_model(np.arange(1700, 2030), r, K), label='Modelo logístico')
plt.scatter([year_2024], [predicted_population_2024], color='blue', label=f'Proyección para 2024: {int(predicted_population_2024)}')
plt.xlabel('Año')
plt.ylabel('Población')
plt.title('Modelo Logístico de Crecimiento Poblacional en Colombia')
plt.legend()
plt.grid(True)
plt.show()