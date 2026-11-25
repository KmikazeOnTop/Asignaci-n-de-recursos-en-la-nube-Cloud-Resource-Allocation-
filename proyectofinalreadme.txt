# Asignación de recursos en la nube: tiempo de respuesta vs. consumo energético

Proyecto final de Ingeniería en Computación, Centro Universitario de la Costa,
Universidad de Guadalajara.

## ¿De qué trata?

Un proveedor de nube debe decidir en qué servidor se ejecuta cada tarea. Esa
decisión afecta dos cosas que se contraponen: cuánto tarda el usuario en
recibir su respuesta (**makespan**) y cuánta energía gasta el centro de datos.
Los servidores rápidos terminan antes, pero consumen mucha más energía.

Este proyecto compara cuatro métodos para resolver el problema:

| Método | Descripción |
|--------|-------------|
| **BASE** | Algoritmo voraz determinista (punto de comparación) |
| **GA** | Algoritmo genético (torneo, cruce uniforme, mutación y elitismo) |
| **PSO** | Optimización por enjambre de partículas, en versión discreta |
| **HIB** | Híbrido GA-PSO: un GA con un paso de atracción hacia la mejor solución global |

## Modelo

- Una solución es una lista donde el índice es la tarea y el valor es el servidor asignado.
- La potencia de cada servidor crece con el cuadrado de su velocidad (P = 20·v²).
- Función de aptitud normalizada: `F = w·makespan/R₁ + (1 − w)·energía/R₂`
  (menor es mejor). Con `w = 0.5`, tiempo y energía pesan igual.

## Experimentos

- 3 escenarios: **A** (10 tareas, 3 servidores), **B** (30, 6) y **C** (100, 10).
- 30 corridas independientes por algoritmo y escenario, con el mismo presupuesto de evaluaciones.
- Prueba de Wilcoxon por pares y análisis de sensibilidad del peso `w`.

## Resultados principales

- Las tres metaheurísticas mejoran al baseline en fitness: 4.7 % (A), 1.4 % (B) y 6.2 % (C).
- El híbrido converge un poco más rápido al inicio, pero **no supera** al GA ni al PSO
  en calidad final; ninguna diferencia sobrevive a la corrección de Bonferroni.
- El compromiso tiempo-energía cambia de forma escalonada al variar `w`.

## Requisitos

- Python 3.9 o superior
- `matplotlib`
- `scipy`

```bash
pip install matplotlib scipy
```

## Uso

```bash
python PROYECTOFINAL_documentado.py
```

Por defecto ejecuta el análisis de sensibilidad. Para correr los experimentos
completos y generar tablas y gráficas, consulta los comentarios del bloque
`__main__` al final del archivo:

```python
resultados = correr_experimentos(30, "resultados.csv")
```

## Estructura del código

1. Modelo del problema (instancia, evaluación y fitness)
2. Operadores comunes (población, cruce, mutación, torneo, acercar)
3. Algoritmos (GA, PSO, baseline voraz, híbrido)
4. Configuración y ejecución de experimentos
5. Estadísticas y resúmenes
6. Gráficas
7. Pruebas estadísticas y sensibilidad

## Limitaciones

- Los parámetros no se afinaron.
- Una sola instancia sintética por escenario.
- Modelo simple: tareas independientes, sin plazos ni memoria.

## Autor

Miguel Ignacio González
Mtro. Santo Rubio Pano, Universidad de Guadalajara, CUCosta