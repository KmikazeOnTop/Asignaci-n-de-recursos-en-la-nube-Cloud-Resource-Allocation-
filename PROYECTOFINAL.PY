"""
PROYECTO FINAL: Asignación de recursos en la nube
optimizando tiempo de respuesta y consumo energético.

Universidad de Guadalajara - Centro Universitario de la Costa
Ingeniería en Computación
MIGUEL IGNACIO GONZALEZ GOMEZ

Descripción general
-------------------
Se tienen N tareas (cada una con una carga de trabajo) y M servidores (cada
uno con una velocidad y una potencia). Hay que decidir en qué servidor se
ejecuta cada tarea para minimizar a la vez:
    - el makespan (tiempo que tarda el servidor más cargado en terminar), y
    - la energía total consumida.

Se comparan cuatro métodos:
    - BASE: algoritmo voraz determinista (punto de comparación).
    - GA:   algoritmo genético.
    - PSO:  optimización por enjambre de partículas, en versión discreta.
    - HIB:  híbrido GA-PSO (un GA con un paso de atracción hacia la mejor
            solución global, tomado de PSO).

Representación de una solución
------------------------------
Una solución ("asignacion") es una lista de N enteros: el índice es la
tarea y el valor es el servidor al que se asigna.
    Ejemplo: [2, 0, 1] -> tarea 0 al servidor 2, tarea 1 al 0, tarea 2 al 1.
Cualquier lista con valores entre 0 y M-1 es válida, por lo que no se
necesitan reparaciones.

Organización del archivo
------------------------
    1. Modelo del problema (instancia, evaluación y fitness)
    2. Operadores comunes (población, cruce, mutación, torneo, acercar)
    3. Algoritmos (GA, PSO, baseline voraz, híbrido)
    4. Configuración y ejecución de experimentos
    5. Estadísticas y resúmenes
    6. Gráficas
    7. Pruebas estadísticas y análisis de sensibilidad
    8. Programa principal
"""

import csv
import math
import pickle
import random
import time

import matplotlib.pyplot as plt
from scipy.stats import wilcoxon


# ============================================================================
# 1. MODELO DEL PROBLEMA
# ============================================================================

def generar_instancia(num_tareas, num_servidores, semilla):
    """Genera los datos sintéticos del problema.

    Parámetros:
        num_tareas:     cantidad de tareas (N).
        num_servidores: cantidad de servidores (M).
        semilla:        semilla del generador, para que la instancia sea
                        reproducible.
    Regresa:
        cargas:      lista con la carga de cada tarea (enteros de 10 a 100).
        velocidades: lista con la velocidad de cada servidor (enteros de 1 a 5).
        potencias:   lista con la potencia de cada servidor, P = 20 * v^2.
                     Esto hace que los servidores rápidos gasten mucha más
                     energía por unidad de trabajo, y es lo que genera el
                     conflicto entre tiempo y energía.
    """
    generador = random.Random(semilla)

    cargas = []
    for i in range(num_tareas):
        cargas.append(generador.randint(10, 100))

    velocidades = []
    potencias = []
    for j in range(num_servidores):
        velocidad = generador.randint(1, 5)
        velocidades.append(velocidad)
        potencias.append(20 * velocidad ** 2)

    return cargas, velocidades, potencias


def evaluar(asignacion, cargas, velocidades, potencias):
    """Calcula el makespan y la energía de una asignación.

    Para cada servidor j:
        tiempo_j = (suma de cargas de sus tareas) / velocidad_j
    Entonces:
        makespan = máximo de tiempo_j
        energía  = suma de potencia_j * tiempo_j
    Un servidor sin tareas tiene tiempo 0 y no consume energía.

    Regresa:
        (makespan, energia)
    """
    num_servidores = len(velocidades)
    carga_servidor = [0] * num_servidores

    # Acumular la carga que recibe cada servidor.
    for i in range(len(asignacion)):
        servidor = asignacion[i]
        carga_servidor[servidor] = carga_servidor[servidor] + cargas[i]

    # Calcular tiempo por servidor, makespan y energía total.
    makespan = 0
    energia = 0
    for j in range(num_servidores):
        tiempo = carga_servidor[j] / velocidades[j]
        if tiempo > makespan:
            makespan = tiempo
        energia = energia + potencias[j] * tiempo

    return makespan, energia


def crear_instancia(num_tareas, num_servidores, semilla):
    """Crea una instancia completa, incluyendo las cotas de referencia.

    Además de los datos básicos, calcula dos cotas inferiores que sirven para
    normalizar el fitness (así tiempo y energía quedan en escala parecida):
        ref_tiempo  = (suma de cargas) / (suma de velocidades)
        ref_energia = (mínimo de P_j / v_j) * (suma de cargas)
    Ninguna solución puede superar estas cotas, por lo que cada término del
    fitness vale al menos 1.

    Regresa:
        Diccionario con las claves: "cargas", "velocidades", "potencias",
        "ref_tiempo" y "ref_energia".
    """
    cargas, velocidades, potencias = generar_instancia(num_tareas, num_servidores, semilla)

    ref_tiempo = sum(cargas) / sum(velocidades)

    # Energía por unidad de carga de cada servidor; la mejor es la mínima.
    eficiencias = []
    for j in range(len(velocidades)):
        eficiencias.append(potencias[j] / velocidades[j])
    ref_energia = min(eficiencias) * sum(cargas)

    return {"cargas": cargas,
            "velocidades": velocidades,
            "potencias": potencias,
            "ref_tiempo": ref_tiempo,
            "ref_energia": ref_energia}


def fitness(asignacion, instancia, peso):
    """Función de aptitud: combina tiempo y energía en un solo número.

        F = w * makespan / ref_tiempo + (1 - w) * energia / ref_energia

    Parámetros:
        peso: w, entre 0 y 1. Con w = 1 solo importa el tiempo; con w = 0,
              solo la energía; con w = 0.5 pesan igual.
    Regresa:
        Valor de F (siempre >= 1). Menor es mejor.
    """
    makespan, energia = evaluar(asignacion, instancia["cargas"],
                                instancia["velocidades"], instancia["potencias"])
    return (peso * makespan / instancia["ref_tiempo"]
            + (1 - peso) * energia / instancia["ref_energia"])


# ============================================================================
# 2. OPERADORES COMUNES
# ============================================================================

def crear_poblacion(tam_poblacion, num_tareas, num_servidores, generador):
    """Crea una población inicial de asignaciones completamente aleatorias.

    Regresa:
        Lista de 'tam_poblacion' asignaciones.
    """
    poblacion = []
    for i in range(tam_poblacion):
        asignacion = []
        for t in range(num_tareas):
            asignacion.append(generador.randrange(num_servidores))
        poblacion.append(asignacion)
    return poblacion


def cruce_uniforme(padre1, padre2, generador):
    """Cruce uniforme: cada gen del hijo viene de un padre al azar (50 %).

    Como cualquier combinación de genes es una asignación válida, no hace
    falta ninguna corrección.

    Regresa:
        Una nueva asignación (hijo); no modifica a los padres.
    """
    hijo = []
    for i in range(len(padre1)):
        if generador.random() < 0.5:
            hijo.append(padre1[i])
        else:
            hijo.append(padre2[i])
    return hijo


def mutacion(asignacion, prob_gen, num_servidores, generador):
    """Mutación por gen: con probabilidad 'prob_gen', cada tarea cambia a un
    servidor aleatorio.

    Nota: modifica la lista recibida (en sitio) y también la regresa.
    """
    for i in range(len(asignacion)):
        if generador.random() < prob_gen:
            asignacion[i] = generador.randrange(num_servidores)
    return asignacion


def evaluar_poblacion(poblacion, instancia, peso):
    """Calcula el fitness de cada individuo de la población.

    Regresa:
        Lista de aptitudes, en el mismo orden que la población.
    """
    aptitudes = []
    for asignacion in poblacion:
        aptitudes.append(fitness(asignacion, instancia, peso))
    return aptitudes


def posicion_del_mejor(aptitudes):
    """Regresa la posición del menor valor de la lista (el mejor fitness)."""
    mejor = 0
    for i in range(len(aptitudes)):
        if aptitudes[i] < aptitudes[mejor]:
            mejor = i
    return mejor


def seleccion_torneo(poblacion, aptitudes, generador, k=3):
    """Selección por torneo: se eligen k individuos al azar (sin repetir) y
    gana el de menor fitness.

    Regresa:
        El individuo ganador (no una copia).
    """
    posiciones = generador.sample(range(len(poblacion)), k)
    mejor_posicion = posiciones[0]
    for posicion in posiciones:
        if aptitudes[posicion] < aptitudes[mejor_posicion]:
            mejor_posicion = posicion
    return poblacion[mejor_posicion]


def acercar(asignacion, objetivo, prob, generador):
    """Adaptación discreta del movimiento de PSO.

    Cada gen que difiere del 'objetivo' se copia con probabilidad 'prob'. Así
    la asignación se "acerca" al objetivo sin salirse del espacio de
    asignaciones válidas.

    Regresa:
        Una copia modificada; no altera la asignación original.
    """
    copia = asignacion[:]
    for i in range(len(copia)):
        if copia[i] != objetivo[i]:
            if generador.random() < prob:
                copia[i] = objetivo[i]
    return copia


# ============================================================================
# 3. ALGORITMOS
# ============================================================================
# Los tres algoritmos estocásticos regresan lo mismo:
#     (mejor_solucion, mejor_fitness, historial)
# donde 'historial' es una lista de pares (evaluaciones, mejor_fitness) que
# sirve para dibujar las curvas de convergencia.

def algoritmo_genetico(instancia, peso, tam_poblacion, max_evaluaciones,
                       prob_gen, semilla, k=3):
    """Algoritmo genético (GA).

    Cada generación conserva al mejor individuo (elitismo) y completa la
    población con hijos creados por torneo, cruce uniforme y mutación.

    Parámetros:
        instancia:        diccionario de crear_instancia().
        peso:             peso w del fitness.
        tam_poblacion:    número de individuos por generación.
        max_evaluaciones: presupuesto de evaluaciones de fitness.
        prob_gen:         probabilidad de mutación por gen.
        semilla:          semilla de esta corrida.
        k:                tamaño del torneo.
    Nota:
        Evalúa en bloques de 'tam_poblacion', por lo que puede pasarse del
        presupuesto hasta en tam_poblacion - 1 evaluaciones.
    """
    generador = random.Random(semilla)
    num_tareas = len(instancia["cargas"])
    num_servidores = len(instancia["velocidades"])

    # Población inicial y su mejor individuo.
    poblacion = crear_poblacion(tam_poblacion, num_tareas, num_servidores, generador)
    aptitudes = evaluar_poblacion(poblacion, instancia, peso)
    evaluaciones = tam_poblacion

    pos = posicion_del_mejor(aptitudes)
    mejor = poblacion[pos][:]
    mejor_fit = aptitudes[pos]
    historial = [(evaluaciones, mejor_fit)]

    while evaluaciones < max_evaluaciones:
        # Elitismo: el mejor pasa intacto a la siguiente generación.
        nueva_poblacion = [mejor[:]]
        while len(nueva_poblacion) < tam_poblacion:
            padre1 = seleccion_torneo(poblacion, aptitudes, generador, k)
            padre2 = seleccion_torneo(poblacion, aptitudes, generador, k)
            hijo = cruce_uniforme(padre1, padre2, generador)
            hijo = mutacion(hijo, prob_gen, num_servidores, generador)
            nueva_poblacion.append(hijo)

        poblacion = nueva_poblacion
        aptitudes = evaluar_poblacion(poblacion, instancia, peso)
        evaluaciones = evaluaciones + tam_poblacion

        # Actualizar la mejor solución encontrada hasta ahora.
        pos = posicion_del_mejor(aptitudes)
        if aptitudes[pos] < mejor_fit:
            mejor = poblacion[pos][:]
            mejor_fit = aptitudes[pos]
        historial.append((evaluaciones, mejor_fit))

    return mejor, mejor_fit, historial


def pso(instancia, peso, num_particulas, c1, c2, prob_gen,
        max_evaluaciones, semilla):
    """Optimización por enjambre de partículas (PSO) discreto.

    En cada paso, cada partícula se acerca a su mejor posición personal
    (con probabilidad c1), luego a la mejor global (con probabilidad c2) y
    finalmente recibe una mutación de exploración.

    Parámetros:
        num_particulas: tamaño del enjambre.
        c1:             probabilidad de copiar genes de su mejor personal.
        c2:             probabilidad de copiar genes de la mejor global.
        prob_gen:       probabilidad de mutación por gen.
    Nota:
        Evalúa una partícula a la vez, pero registra el historial una vez por
        ronda completa del enjambre (cada 'num_particulas' evaluaciones).
    """
    generador = random.Random(semilla)
    num_tareas = len(instancia["cargas"])
    num_servidores = len(instancia["velocidades"])

    particulas = crear_poblacion(num_particulas, num_tareas, num_servidores, generador)
    aptitudes = evaluar_poblacion(particulas, instancia, peso)
    evaluaciones = num_particulas

    # La mejor posición personal inicial es la posición de arranque.
    mejores_personales = [p[:] for p in particulas]
    mejores_fit = aptitudes[:]

    pos = posicion_del_mejor(aptitudes)
    mejor_global = particulas[pos][:]
    mejor_global_fit = aptitudes[pos]
    historial = [(evaluaciones, mejor_global_fit)]

    while evaluaciones < max_evaluaciones:
        for i in range(num_particulas):
            nueva = acercar(particulas[i], mejores_personales[i], c1, generador)
            nueva = acercar(nueva, mejor_global, c2, generador)
            nueva = mutacion(nueva, prob_gen, num_servidores, generador)
            particulas[i] = nueva

            f = fitness(nueva, instancia, peso)
            evaluaciones = evaluaciones + 1

            if f < mejores_fit[i]:
                mejores_personales[i] = nueva[:]
                mejores_fit[i] = f
            if f < mejor_global_fit:
                mejor_global = nueva[:]
                mejor_global_fit = f
        historial.append((evaluaciones, mejor_global_fit))

    return mejor_global, mejor_global_fit, historial


def baseline_voraz(instancia, peso):
    """Baseline determinista: algoritmo voraz en línea.

    Procesa las tareas en orden y asigna cada una al servidor que da el menor
    fitness de la asignación parcial construida hasta el momento. No usa
    azar y utiliza N * M evaluaciones.

    Regresa:
        (asignacion, fitness_de_la_asignacion)
    """
    num_servidores = len(instancia["velocidades"])
    num_tareas = len(instancia["cargas"])
    asignacion = []

    for t in range(num_tareas):
        mejor_servidor = 0
        mejor_fit = None
        # Probar la tarea en cada servidor y quedarse con el mejor.
        for s in range(num_servidores):
            candidata = asignacion + [s]
            f = fitness(candidata, instancia, peso)
            if mejor_fit is None or f < mejor_fit:
                mejor_fit = f
                mejor_servidor = s
        asignacion.append(mejor_servidor)

    return asignacion, fitness(asignacion, instancia, peso)


def hibrido(instancia, peso, tam_poblacion, max_evaluaciones,
            prob_gen, c_social, semilla, k=3):
    """Híbrido GA-PSO.

    Es el algoritmo genético con un paso adicional tomado de PSO: después del
    cruce y antes de la mutación, cada hijo se acerca a la mejor solución
    global con probabilidad 'c_social' por gen. Así, el cruce explora y la
    atracción social explota.

    Casos extremos de c_social:
        0 -> el híbrido se reduce a un GA puro.
        1 -> todos los hijos serían copias de la mejor solución.
    """
    generador = random.Random(semilla)
    num_tareas = len(instancia["cargas"])
    num_servidores = len(instancia["velocidades"])

    poblacion = crear_poblacion(tam_poblacion, num_tareas, num_servidores, generador)
    aptitudes = evaluar_poblacion(poblacion, instancia, peso)
    evaluaciones = tam_poblacion

    pos = posicion_del_mejor(aptitudes)
    mejor = poblacion[pos][:]
    mejor_fit = aptitudes[pos]
    historial = [(evaluaciones, mejor_fit)]

    while evaluaciones < max_evaluaciones:
        nueva_poblacion = [mejor[:]]  # elitismo
        while len(nueva_poblacion) < tam_poblacion:
            padre1 = seleccion_torneo(poblacion, aptitudes, generador, k)
            padre2 = seleccion_torneo(poblacion, aptitudes, generador, k)
            hijo = cruce_uniforme(padre1, padre2, generador)
            hijo = acercar(hijo, mejor, c_social, generador)  # paso tomado de PSO
            hijo = mutacion(hijo, prob_gen, num_servidores, generador)
            nueva_poblacion.append(hijo)

        poblacion = nueva_poblacion
        aptitudes = evaluar_poblacion(poblacion, instancia, peso)
        evaluaciones = evaluaciones + tam_poblacion

        pos = posicion_del_mejor(aptitudes)
        if aptitudes[pos] < mejor_fit:
            mejor = poblacion[pos][:]
            mejor_fit = aptitudes[pos]
        historial.append((evaluaciones, mejor_fit))

    return mejor, mejor_fit, historial


# ============================================================================
# 4. CONFIGURACIÓN Y EJECUCIÓN DE EXPERIMENTOS
# ============================================================================

# Escenarios: nombre -> (num_tareas, num_servidores, presupuesto_de_evaluaciones)
ESCENARIOS = {"A": (10, 3, 3000),
              "B": (30, 6, 10000),
              "C": (100, 10, 20000)}

# Algoritmos estocásticos que se comparan (el baseline se maneja aparte: "BASE").
ALGORITMOS = ["GA", "PSO", "HIB"]

# Peso w de los experimentos principales (tiempo y energía pesan igual).
PESO = 0.5


def ejecutar(nombre, instancia, peso, presupuesto, semilla):
    """Corre un algoritmo con los parámetros fijos del proyecto y mide su tiempo.

    Parámetros fijos: población/enjambre de 30, c1 = c2 = c_social = 0.3 y
    mutación de 1/N por gen.

    Parámetros:
        nombre: "GA", "PSO" o "HIB".
    Regresa:
        (mejor_solucion, mejor_fitness, historial, tiempo_en_segundos)
    """
    inicio = time.perf_counter()
    n = len(instancia["cargas"])
    if nombre == "GA":
        mejor, fit, hist = algoritmo_genetico(instancia, peso, 30, presupuesto, 1 / n, semilla)
    elif nombre == "PSO":
        mejor, fit, hist = pso(instancia, peso, 30, 0.3, 0.3, 1 / n, presupuesto, semilla)
    else:
        mejor, fit, hist = hibrido(instancia, peso, 30, presupuesto, 1 / n, 0.3, semilla)
    tiempo = time.perf_counter() - inicio
    return mejor, fit, hist, tiempo


def correr_experimentos(num_corridas, archivo):
    """Ejecuta todos los experimentos y guarda los resultados en un CSV.

    Para cada escenario: corre el baseline una vez y cada algoritmo
    'num_corridas' veces (semillas 1 a num_corridas). Todas las corridas de un
    escenario comparten la misma instancia (semilla 42).

    Parámetros:
        num_corridas: corridas por algoritmo y escenario (30 en el proyecto).
        archivo:      nombre del CSV de salida.
    Regresa:
        Lista de tuplas, una por corrida, con el formato:
            (escenario, algoritmo, semilla, solucion, fitness, historial, tiempo)
        Las demás funciones de análisis acceden a ellas por posición:
            [0] escenario, [1] algoritmo, [2] semilla, [3] solución,
            [4] fitness final, [5] historial, [6] tiempo.
    """
    resultados = []
    with open(archivo, "w", newline="") as f:
        escritor = csv.writer(f)
        escritor.writerow(["escenario", "algoritmo", "semilla", "fitness",
                           "makespan", "energia", "evaluaciones", "tiempo"])

        for esc in ESCENARIOS:
            num_tareas, num_servidores, presupuesto = ESCENARIOS[esc]
            instancia = crear_instancia(num_tareas, num_servidores, 42)

            # Baseline voraz: determinista, solo se corre una vez.
            inicio = time.perf_counter()
            sol, fit = baseline_voraz(instancia, PESO)
            tiempo = time.perf_counter() - inicio
            evaluaciones = num_tareas * num_servidores
            mk, en = evaluar(sol, instancia["cargas"], instancia["velocidades"],
                             instancia["potencias"])
            escritor.writerow([esc, "BASE", 0, fit, mk, en, evaluaciones, tiempo])
            resultados.append((esc, "BASE", 0, sol, fit, [(evaluaciones, fit)], tiempo))

            # Metaheurísticas: varias corridas con semillas distintas.
            for alg in ALGORITMOS:
                for semilla in range(1, num_corridas + 1):
                    sol, fit, hist, tiempo = ejecutar(alg, instancia, PESO,
                                                      presupuesto, semilla)
                    mk, en = evaluar(sol, instancia["cargas"],
                                     instancia["velocidades"], instancia["potencias"])
                    escritor.writerow([esc, alg, semilla, fit, mk, en,
                                       hist[-1][0], tiempo])
                    resultados.append((esc, alg, semilla, sol, fit, hist, tiempo))

    return resultados


# ============================================================================
# 5. ESTADÍSTICAS Y RESÚMENES
# ============================================================================

def corridas_de(resultados, escenario, algoritmo):
    """Filtra las corridas (tuplas completas) de un escenario y algoritmo."""
    seleccion = []

    for corrida in resultados:
        if corrida[0] == escenario and corrida[1] == algoritmo:
            seleccion.append(corrida)

    return seleccion


def fitness_de(resultados, escenario, algoritmo):
    """Lista de fitness finales de todas las corridas de un escenario y algoritmo."""
    valores = []

    for corrida in resultados:
        if corrida[0] == escenario and corrida[1] == algoritmo:
            valores.append(corrida[4])

    return valores


def promedio(lista):
    """Media aritmética de una lista de números."""
    return sum(lista) / len(lista)


def desviacion(lista):
    """Desviación estándar muestral (divide entre n - 1).

    Regresa 0 si la lista tiene menos de 2 elementos (por ejemplo, el
    baseline, que solo tiene una corrida).
    """
    if len(lista) < 2:
        return 0
    media = promedio(lista)
    suma = 0
    for x in lista:
        suma = suma + (x - media) ** 2
    return math.sqrt(suma / (len(lista) - 1))


def generar_resumen(resultados, archivo):
    """Escribe un CSV con las métricas por escenario y algoritmo.

    Columnas: mejor, peor, media y desviación del fitness final; makespan,
    energía, evaluaciones y tiempo promedio.
    """
    with open(archivo, "w", newline="") as f:
        escritor = csv.writer(f)
        escritor.writerow(["escenario", "algoritmo", "mejor", "peor", "media",
                           "desviacion", "makespan_prom", "energia_prom",
                           "evaluaciones_prom", "tiempo_prom"])
        for esc in ESCENARIOS:
            num_tareas, num_servidores, presupuesto = ESCENARIOS[esc]
            instancia = crear_instancia(num_tareas, num_servidores, 42)
            for alg in ["BASE"] + ALGORITMOS:
                corridas = corridas_de(resultados, esc, alg)
                fits = [c[4] for c in corridas]
                tiempos = [c[6] for c in corridas]
                evals = [c[5][-1][0] for c in corridas]
                makespans = []
                energias = []
                for c in corridas:
                    mk, en = evaluar(c[3], instancia["cargas"],
                                     instancia["velocidades"], instancia["potencias"])
                    makespans.append(mk)
                    energias.append(en)
                escritor.writerow([esc, alg, min(fits), max(fits), promedio(fits),
                                   desviacion(fits), promedio(makespans),
                                   promedio(energias), promedio(evals),
                                   promedio(tiempos)])


def valor_en(historial, evaluaciones):
    """Mejor fitness registrado en el historial hasta un número de evaluaciones.

    Permite comparar algoritmos que registran su progreso en puntos distintos
    (por ejemplo, el GA registra cada generación y el PSO cada ronda del
    enjambre) sobre una misma malla de evaluaciones.
    """
    valor = historial[0][1]
    for ev, fit in historial:
        if ev <= evaluaciones:
            valor = fit
    return valor


def curva_promedio(resultados, escenario, algoritmo, malla):
    """Promedia, sobre todas las corridas, el mejor fitness en cada punto de la malla.

    Parámetros:
        malla: lista de números de evaluaciones donde se calcula el promedio.
    Regresa:
        Lista de promedios, uno por punto de la malla.
    """
    historiales = []
    for corrida in resultados:
        if corrida[0] == escenario and corrida[1] == algoritmo:
            historiales.append(corrida[5])

    promedios = []
    for e in malla:
        suma = 0
        for historial in historiales:
            suma = suma + valor_en(historial, e)
        promedios.append(suma / len(historiales))
    return promedios


# ============================================================================
# 6. GRÁFICAS
# ============================================================================

def grafica_boxplot(resultados, escenario, archivo):
    """Diagrama de cajas del fitness final de los tres algoritmos en un escenario."""
    datos = []

    for alg in ALGORITMOS:
        datos.append(fitness_de(resultados, escenario, alg))

    plt.figure(figsize=(6, 5))
    plt.boxplot(datos)
    plt.xticks([1, 2, 3], ALGORITMOS)
    plt.title("Escenario " + escenario + ": fitness final (30 corridas)")
    plt.ylabel("Mejor fitness")
    plt.savefig(archivo)
    plt.close()


def grafica_convergencia(resultados, escenario, presupuesto, archivo):
    """Curva de convergencia promedio de cada algoritmo; la línea punteada es
    el fitness del baseline.

    La malla de evaluaciones va de 30 en 30 hasta el presupuesto.
    """
    malla = list(range(30, presupuesto + 1, 30))
    plt.figure(figsize=(8, 5))
    for alg in ALGORITMOS:
        curva = curva_promedio(resultados, escenario, alg, malla)
        plt.plot(malla, curva, label=alg)

    base = corridas_de(resultados, escenario, "BASE")[0][4]
    plt.axhline(base, linestyle="--", color="gray", label="BASE")

    plt.title("Escenario " + escenario + ": convergencia promedio (30 corridas)")
    plt.xlabel("Evaluaciones")
    plt.ylabel("Mejor fitness")
    plt.legend()
    plt.savefig(archivo)
    plt.close()


def grafica_pareto(resultados, escenario, archivo):
    """Dispersión de makespan contra energía de cada corrida, por algoritmo.

    Muestra el conflicto entre los dos objetivos: más rápido implica más
    energía.
    """
    num_tareas, num_servidores, presupuesto = ESCENARIOS[escenario]
    instancia = crear_instancia(num_tareas, num_servidores, 42)

    plt.figure(figsize=(7, 5))
    for alg in ["BASE"] + ALGORITMOS:
        mks = []
        ens = []
        for c in corridas_de(resultados, escenario, alg):
            mk, en = evaluar(c[3], instancia["cargas"],
                             instancia["velocidades"], instancia["potencias"])
            mks.append(mk)
            ens.append(en)
        plt.scatter(mks, ens, label=alg)

    plt.title("Escenario " + escenario + ": tiempo contra energía")
    plt.xlabel("Makespan (tiempo de respuesta)")
    plt.ylabel("Energía")
    plt.legend()
    plt.savefig(archivo)
    plt.close()


def grafica_sensibilidad(filas, archivo):
    """Gráfica del compromiso tiempo-energía al variar el peso w.

    Parámetros:
        filas: lista de tuplas (peso, makespan_prom, energia_prom), tal como
               la regresa sensibilidad_peso().
    """
    mks = [f[1] for f in filas]
    ens = [f[2] for f in filas]
    plt.figure(figsize=(7, 5))
    plt.plot(mks, ens, marker="o")
    for f in filas:
        plt.annotate("w=" + str(f[0]), (f[1], f[2]))
    plt.title("Sensibilidad al peso w (GA, escenario B)")
    plt.xlabel("Makespan promedio")
    plt.ylabel("Energía promedio")
    plt.savefig(archivo)
    plt.close()


# ============================================================================
# 7. PRUEBAS ESTADÍSTICAS Y SENSIBILIDAD
# ============================================================================

def comparar_wilcoxon(resultados, escenario, alg1, alg2):
    """Prueba de Wilcoxon entre los fitness finales de dos algoritmos.

    Se rechaza la igualdad si p < 0.05 (con Bonferroni para 9 pruebas, el
    umbral sería 0.05 / 9 = 0.0056).

    Regresa:
        El valor p, o None si todas las diferencias son cero (en ese caso
        scipy lanza ValueError).
    Nota:
        La semilla identifica la corrida pero no genera pares reales entre
        algoritmos; Mann-Whitney sería una alternativa válida.
    """
    fits1 = [c[4] for c in corridas_de(resultados, escenario, alg1)]
    fits2 = [c[4] for c in corridas_de(resultados, escenario, alg2)]
    try:
        estadistico, p = wilcoxon(fits1, fits2)
    except ValueError:
        return None
    return p


def sensibilidad_peso(escenario, pesos, num_corridas):
    """Análisis de sensibilidad: corre el GA con distintos pesos w.

    Parámetros:
        escenario:    nombre del escenario ("A", "B" o "C").
        pesos:        lista de valores de w a probar.
        num_corridas: corridas del GA por cada valor de w.
    Regresa:
        Lista de tuplas (peso, makespan_promedio, energia_promedio).
    """
    num_tareas, num_servidores, presupuesto = ESCENARIOS[escenario]
    instancia = crear_instancia(num_tareas, num_servidores, 42)
    filas = []
    for peso in pesos:
        mks = []
        ens = []
        for semilla in range(1, num_corridas + 1):
            sol, fit, hist = algoritmo_genetico(instancia, peso, 30, presupuesto,
                                                1 / num_tareas, semilla)
            mk, en = evaluar(sol, instancia["cargas"],
                             instancia["velocidades"], instancia["potencias"])
            mks.append(mk)
            ens.append(en)
        filas.append((peso, promedio(mks), promedio(ens)))
    return filas


# ============================================================================
# 8. PROGRAMA PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    # Análisis de sensibilidad al peso w (GA, escenario B, 10 corridas por valor).
    filas = sensibilidad_peso("B", [0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 1], 10)
    for f in filas:
        print(f)
    grafica_sensibilidad(filas, "final_sensibilidad_B.png")

    # ------------------------------------------------------------------
    # Partes documentadas:
    #
    #     resultados = correr_experimentos(30, "resultados.csv")
    #     # o bien, si ya se guardaron en un archivo:
    #     with open("resultados.pkl", "rb") as f:
    #         resultados = pickle.load(f)
    #
    # Gráficas por escenario (convergencia, cajas y tiempo contra energía):
    #
    #     for esc in ESCENARIOS:
    #         presupuesto = ESCENARIOS[esc][2]
    #         grafica_convergencia(resultados, esc, presupuesto,
    #                              "final_convergencia_" + esc + ".png")
    #         grafica_boxplot(resultados, esc, "final_boxplot_" + esc + ".png")
    #         grafica_pareto(resultados, esc, "final_pareto_" + esc + ".png")
    #
    # Pruebas de Wilcoxon por pares de algoritmos en cada escenario:
    #
    #     print("escenario  par         p")
    #     for esc in ESCENARIOS:
    #         for alg1, alg2 in [("HIB", "GA"), ("HIB", "PSO"), ("GA", "PSO")]:
    #             p = comparar_wilcoxon(resultados, esc, alg1, alg2)
    #             if p is None:
    #                 print(esc, alg1, "vs", alg2, "sin diferencias")
    #             else:
    #                 print(esc, alg1, "vs", alg2, round(p, 4),
    #                       "significativa" if p < 0.05 else "no significativa")
    # ------------------------------------------------------------------