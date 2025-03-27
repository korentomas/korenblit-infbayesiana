# ==============================================
# Práctica 1
# Especificación y evaluación de argumentos causales.
# Docente: Gustavo Landfried
# Inferencia Bayesiana Causal 1
# 1er cuatrimestre 2025
# UNSAM
# Alumno: Tomás Pablo Korenblit
# ==============================================

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ==============================================
# 1.1 Definir distribuciones condicionales
# ==============================================
H = np.arange(3)  # Posibles valores para r, c, s (0,1,2)

def pr(r): 
    """Distribución marginal P(r) - igual para todos los modelos"""
    return 1/3

def pc(c): 
    """Distribución marginal P(c) - igual para todos los modelos"""
    return 1/3

def ps_rM0(s, r):
    """P(s|r) para Modelo Base (M0)"""
    return 0 if s == r else 1/2

def ps_rcM1(s, r, c):
    """P(s|r,c) para Modelo Monty Hall (M1)"""
    opciones_validas = [x for x in H if x != r and x != c]
    return 1/len(opciones_validas) if s in opciones_validas else 0

def prcs_M(r, c, s, m):
    """Distribución conjunta P(r,c,s|M)"""
    if m == 0:  # Modelo Base
        return pr(r) * pc(c) * ps_rM0(s, r)
    else:       # Modelo Monty Hall
        return pr(r) * pc(c) * ps_rcM1(s, r, c)

# ==============================================
# 1.2 Simular datos con modelo Monty Hall
# ==============================================
def simular_datos(T=16, semilla=0):
    """Genera datos según el modelo Monty Hall (M1)"""
    np.random.seed(semilla)
    datos = []
    for _ in range(T):
        r = np.random.choice(3, p=[pr(h) for h in H])
        c = np.random.choice(3, p=[pc(h) for h in H])
        probs_s = [ps_rcM1(s, r, c) for s in H]
        s = np.random.choice(3, p=probs_s)
        datos.append((c, s, r))
    return datos

datos_MH = simular_datos()

# ==============================================
# 1.3 Predicción a priori de cada modelo
# ==============================================
def pDatos_M(datos, m, log=False):
    """Calcula P(Datos|M) para un modelo específico"""
    log_prob = 0.0
    for c, s, r in datos:
        prob = prcs_M(r, c, s, m)
        if prob <= 0: return -np.inf if log else 0
        log_prob += np.log(prob)
    return log_prob if log else np.exp(log_prob)

# ==============================================
# 1.4 Predicción con contribución de todos los modelos
# ==============================================
def pDatos(datos, modelos=[0,1], log=False):
    """Calcula P(Datos) marginalizando sobre los modelos"""
    log_probs = [pDatos_M(datos, m, log=True) + np.log(0.5) for m in modelos]
    max_log = np.max(log_probs)
    log_total = max_log + np.log(sum(np.exp(lp - max_log) for lp in log_probs))
    return log_total if log else np.exp(log_total)

# ==============================================
# 1.5 Posterior de los modelos
# ==============================================
def pM_Datos(m, datos, modelos=[0,1]):
    """Calcula P(M|Datos) usando Bayes"""
    log_posterior = pDatos_M(datos, m, log=True) + np.log(0.5) - pDatos(datos, log=True)
    return np.exp(log_posterior)

# ==============================================
# 1.6 Graficar evolución del posterior
# ==============================================
def graficar_evolucion(datos):
    """Muestra cómo evoluciona P(M|Datos) con más datos"""
    posteriors = {0: [], 1: []}
    for t in range(1, len(datos)+1):
        sub_datos = datos[:t]
        for m in [0,1]:
            posteriors[m].append(pM_Datos(m, sub_datos))
    
    plt.figure(figsize=(10,6))
    plt.plot(posteriors[0], 'b-', label='Modelo Base (M0)')
    plt.plot(posteriors[1], 'r-', label='Modelo Monty Hall (M1)')
    plt.xlabel('Número de episodios'), plt.ylabel('P(M|Datos)')
    plt.legend(), plt.title('Evolución del Posterior')
    plt.show()

graficar_evolucion(datos_MH)

# ==============================================
# 2.1 Posterior sobre probabilidad p (Modelo Alternativo)
# ==============================================
Posibles_p = np.linspace(0, 1, 11)  # Discretización de p

def pp_Datos(p, datos):
    """Calcula log P(p|Datos) para el modelo alternativo"""
    log_likelihood = 0.0
    for c, s, r in datos:
        prob = (1-p)*ps_rM0(s,r) + p*ps_rcM1(s,r,c)
        if prob <= 0: return -np.inf
        log_likelihood += np.log(prob * pr(r) * pc(c))
    return log_likelihood + np.log(1/len(Posibles_p))  # Prior uniforme

# ==============================================
# 2.2 Predicción de episodio dado datos anteriores
# ==============================================
def predecir_episodio(episodio, datos_previos):
    """Calcula P(episodio|datos_previos) para MA"""
    if not datos_previos:  # Sin datos previos, usar prior uniforme
        posterior_p = np.ones(len(Posibles_p))/len(Posibles_p)
    else:
        log_post = np.array([pp_Datos(p, datos_previos) for p in Posibles_p])
        posterior_p = np.exp(log_post - np.max(log_post))  # Estabilidad numérica
        posterior_p /= posterior_p.sum()
    
    c, s, r = episodio
    prob = sum([((1-p)*ps_rM0(s,r) + p*ps_rcM1(s,r,c)) * posterior_p[i] 
               for i, p in enumerate(Posibles_p)])
    return prob * pr(r) * pc(c)

# ==============================================
# 2.3 Predicción completa del modelo alternativo
# ==============================================
def pDatos_MA(datos, log=True):
    """Calcula P(Datos|MA) usando actualización secuencial"""
    log_prob = 0.0
    for t in range(len(datos)):
        prob = predecir_episodio(datos[t], datos[:t])
        log_prob += np.log(prob) if prob > 0 else -np.inf
    return log_prob if log else np.exp(log_prob)

# Cargar datos reales
df = pd.read_csv('./practicas/data/NoMontyHall.csv')
datos_reales = list(zip(df.c, df.s, df.r))[:2000]

# ==============================================
# 2.4 Comparación de desempeño (log Bayes Factor)
# ==============================================
log_m0 = pDatos_M(datos_reales, 0, log=True)
log_m1 = pDatos_M(datos_reales, 1, log=True)
log_ma = pDatos_MA(datos_reales)

print("\nComparación de modelos (log10 Bayes Factors):")
print(f"M1 vs M0: {log_m1 - log_m0:.2f}")
print(f"MA vs M0: {log_ma - log_m0:.2f}")
print(f"MA vs M1: {log_ma - log_m1:.2f}")

# ==============================================
# 2.5 Predicción típica (media geométrica)
# ==============================================
def prediccion_tipica(log_total, n_episodios):
    """Calcula la media geométrica de las predicciones"""
    return np.exp(log_total / n_episodios) if n_episodios > 0 else 0

print("\nPredicciones típicas:")
print(f"M0: {prediccion_tipica(log_m0, len(datos_reales)):.3f}")
print(f"M1: {prediccion_tipica(log_m1, len(datos_reales)):.3f}")
print(f"MA: {prediccion_tipica(log_ma, len(datos_reales)):.3f}")

# ==============================================
# 2.6 Graficar posteriores para primeros episodios
# ==============================================
def graficar_posteriores(datos, n_episodios=60):
    """Compara P(M|Datos) para los tres modelos"""
    posteriors = {'M0': [], 'M1': [], 'MA': []}
    
    for t in range(1, n_episodios+1):
        sub_datos = datos[:t]
        # Normalizar para que sumen 1
        log_probs = [
            pDatos_M(sub_datos, 0, log=True) + np.log(1/3),
            pDatos_M(sub_datos, 1, log=True) + np.log(1/3),
            pDatos_MA(sub_datos) + np.log(1/3)
        ]
        max_log = np.max(log_probs)
        probs = np.exp([lp - max_log for lp in log_probs])
        probs /= probs.sum()
        
        posteriors['M0'].append(probs[0])
        posteriors['M1'].append(probs[1])
        posteriors['MA'].append(probs[2])
    
    plt.figure(figsize=(10,6))
    for modelo, color in zip(['M0', 'M1', 'MA'], ['blue', 'red', 'green']):
        plt.plot(range(1,n_episodios+1), posteriors[modelo], color=color, label=modelo)
    plt.xlabel('Episodio'), plt.ylabel('P(Modelo|Datos)')
    plt.title('Comparación de Modelos en Primeros Episodios')
    plt.legend(), plt.show()

graficar_posteriores(datos_reales)